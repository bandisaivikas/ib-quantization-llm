import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME   = "gpt2"
N_CALIBRATION = 20          # calibration prompts
MAX_LENGTH    = 64
DEVICE        = "mps" if torch.backends.mps.is_available() else "cpu"
SEED          = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Calibration prompts — reasoning-style sentences
CALIBRATION_PROMPTS = [
    "If a train travels 60 miles per hour for 2 hours, how far does it travel?",
    "What is the capital of France and why is it historically significant?",
    "Solve for x: 3x + 7 = 22. Show your reasoning step by step.",
    "A store offers 20% discount on a $50 item. What is the final price?",
    "If all roses are flowers and some flowers fade quickly, what can we conclude?",
    "The sum of two consecutive integers is 37. What are the integers?",
    "A rectangle has a perimeter of 40cm. If length is 12cm, find the width.",
    "Explain why the sky appears blue during the day.",
    "If you invest $1000 at 5% annual interest, how much after 2 years?",
    "A car uses 8 liters per 100km. How much fuel for a 250km trip?",
    "What is the probability of rolling an even number on a six-sided die?",
    "If 5 workers complete a job in 8 days, how long for 10 workers?",
    "Convert 0.75 to a fraction and explain the conversion process.",
    "A triangle has angles 45° and 60°. What is the third angle?",
    "If the temperature drops 3°C every hour, what is the drop after 5 hours?",
    "Two trains start 300km apart and move toward each other at 60 and 90 km/h.",
    "What is the area of a circle with radius 7cm? Use π ≈ 3.14.",
    "A recipe needs 2.5 cups of flour for 12 cookies. How much for 30 cookies?",
    "The ratio of boys to girls is 3:2. If there are 30 students, how many boys?",
    "If 40% of a number is 80, what is the number?",
]

print("=" * 60)
print("IB T-Selection Experiment — Loading model...")
print(f"Device: {DEVICE}")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 1: Load model and tokenizer
# ─────────────────────────────────────────────
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model_full = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model_full = model_full.to(DEVICE)
model_full.eval()

print(f"Model loaded: {MODEL_NAME}")
print(f"Number of transformer layers: {len(model_full.transformer.h)}")
n_layers = len(model_full.transformer.h)

# ─────────────────────────────────────────────
# STEP 2: Tokenize calibration prompts
# ─────────────────────────────────────────────
encodings = tokenizer(
    CALIBRATION_PROMPTS,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH
).to(DEVICE)

print(f"Calibration set: {len(CALIBRATION_PROMPTS)} prompts tokenized.")

# ─────────────────────────────────────────────
# UTILITY: KL-Divergence between two distributions
# ─────────────────────────────────────────────
def compute_kl_divergence(model_p, model_q, encodings):
    """
    Compute mean KL-Divergence between full-precision (p) and quantized (q) models
    across all calibration prompts at the token-level output distribution.
    KL(P || Q) = sum_x P(x) * log(P(x)/Q(x))
    """
    kl_values = []
    with torch.no_grad():
        for i in range(len(CALIBRATION_PROMPTS)):
            input_ids = encodings["input_ids"][i].unsqueeze(0)
            attn_mask = encodings["attention_mask"][i].unsqueeze(0)

            logits_p = model_p(input_ids=input_ids, attention_mask=attn_mask).logits
            logits_q = model_q(input_ids=input_ids, attention_mask=attn_mask).logits

            # Softmax to get probability distributions
            p = torch.softmax(logits_p, dim=-1)  # [1, seq_len, vocab]
            q = torch.softmax(logits_q, dim=-1)  # [1, seq_len, vocab]

            # KL-D averaged over sequence positions
            kl = (p * (p.log() - q.log())).sum(dim=-1)
            kl_values.append(kl.mean().item())

    return np.mean(kl_values)

# ─────────────────────────────────────────────
# UTILITY: INT8 Quantization of a single Linear layer
# ─────────────────────────────────────────────
def quantize_layer_int8(layer):
    """
    Quantize a Linear layer's weights to INT8 and back (simulated quantization).
    Scale = max(|W|) / 127
    This is symmetric per-tensor quantization — identical to what bitsandbytes does at the core.
    """
    with torch.no_grad():
        W = layer.weight.data.float()
        scale = W.abs().max() / 127.0
        W_q = (W / scale).round().clamp(-127, 127)
        W_dequant = W_q * scale
        layer.weight.data = W_dequant.to(layer.weight.dtype)
    return layer

# ─────────────────────────────────────────────
# STEP 3: Ground Truth KL-Divergence Map
# Quantize each layer one at a time, measure KL-D increase
# ─────────────────────────────────────────────
print("\n[Stage 1] Computing Ground Truth KL-D Map (layer-wise quantization)...")
kl_ground_truth = []

for layer_idx in range(n_layers):
    model_q = deepcopy(model_full)
    model_q.eval()

    # Quantize ONLY the MLP layers in this transformer block (c_fc and c_proj)
    block = model_q.transformer.h[layer_idx]
    quantize_layer_int8(block.mlp.c_fc)
    quantize_layer_int8(block.mlp.c_proj)

    kl = compute_kl_divergence(model_full, model_q, encodings)
    kl_ground_truth.append(kl)
    print(f"  Layer {layer_idx:2d}: KL-D = {kl:.6f}")

kl_ground_truth = np.array(kl_ground_truth)
print(f"Ground truth computed. Max degradation at layer: {np.argmax(kl_ground_truth)}")

# ─────────────────────────────────────────────
# STEP 4: Option A Score — Weight Frobenius Norm
# Proxy for weight-level information content
# ─────────────────────────────────────────────
print("\n[Stage 2A] Computing Option A Scores (Weight Frobenius Norm)...")
option_a_scores = []

for layer_idx in range(n_layers):
    block = model_full.transformer.h[layer_idx]
    W_fc   = block.mlp.c_fc.weight.data.float()
    W_proj = block.mlp.c_proj.weight.data.float()
    # Frobenius norm — measures total weight magnitude
    score = (torch.norm(W_fc, 'fro') + torch.norm(W_proj, 'fro')).item()
    option_a_scores.append(score)
    print(f"  Layer {layer_idx:2d}: Weight Norm = {score:.2f}")

option_a_scores = np.array(option_a_scores)

# ─────────────────────────────────────────────
# STEP 5: Option B Score — Activation Mutual Information I(Q(h); Y)
# Measure: how much information about the task output (Y = output logits)
# survives in the quantized activation of each layer?
# ─────────────────────────────────────────────
print("\n[Stage 2B] Computing Option B Scores (Activation Mutual Information)...")

def get_layer_activations(model, encodings, layer_idx):
    """Extract the MLP output activations at a specific layer."""
    activations = []
    hooks = []

    def hook_fn(module, input, output):
        activations.append(output.detach().float().cpu())

    hook = model.transformer.h[layer_idx].mlp.register_forward_hook(hook_fn)
    hooks.append(hook)

    with torch.no_grad():
        for i in range(len(CALIBRATION_PROMPTS)):
            input_ids = encodings["input_ids"][i].unsqueeze(0)
            attn_mask = encodings["attention_mask"][i].unsqueeze(0)
            model(input_ids=input_ids, attention_mask=attn_mask)

    for h in hooks:
        h.remove()

    # Stack: [n_prompts, seq_len, hidden_dim]
    return torch.cat(activations, dim=0)


def quantize_activation_int8(activation):
    """Simulate INT8 quantization on activation tensor."""
    scale = activation.abs().max() / 127.0
    act_q = (activation / scale).round().clamp(-127, 127) * scale
    return act_q


def estimate_mutual_information(X, Y, n_bins=10):
    """
    Estimate I(X; Y) using histogram-based mutual information.
    X, Y are 1D numpy arrays (scalar projections of activations and outputs).
    Uses: I(X;Y) = H(X) + H(Y) - H(X,Y)
    """
    # Normalize to [0,1] for binning
    X_norm = (X - X.min()) / (X.max() - X.min() + 1e-8)
    Y_norm = (Y - Y.min()) / (Y.max() - Y.min() + 1e-8)

    # Joint histogram
    joint_hist, _, _ = np.histogram2d(X_norm, Y_norm, bins=n_bins)
    joint_hist = joint_hist / joint_hist.sum()

    # Marginals
    p_x = joint_hist.sum(axis=1)
    p_y = joint_hist.sum(axis=0)

    # Entropy H(X), H(Y), H(X,Y)
    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    H_X  = entropy(p_x)
    H_Y  = entropy(p_y)
    H_XY = entropy(joint_hist.flatten())

    mi = H_X + H_Y - H_XY
    return max(mi, 0.0)  # MI is always non-negative


# Get full-precision output logits for all prompts (our Y)
output_logprobs = []
with torch.no_grad():
    for i in range(len(CALIBRATION_PROMPTS)):
        input_ids = encodings["input_ids"][i].unsqueeze(0)
        attn_mask = encodings["attention_mask"][i].unsqueeze(0)
        logits = model_full(input_ids=input_ids, attention_mask=attn_mask).logits
        # Summarize Y as mean log-prob of top token (task-relevant scalar per prompt)
        top_logprob = logits.softmax(dim=-1).max(dim=-1).values.mean().item()
        output_logprobs.append(top_logprob)

Y_task = np.array(output_logprobs)

option_b_scores = []
for layer_idx in range(n_layers):
    # Get full-precision activations
    acts_full = get_layer_activations(model_full, encodings, layer_idx)
    # Quantize activations to INT8
    acts_quant = quantize_activation_int8(acts_full)

    # Project to scalar per prompt using L2 norm (reduce seq_len and hidden_dim)
    # This gives one scalar per prompt: ||Q(h_l)||_2
    X_quant = acts_quant.norm(dim=-1).mean(dim=-1).numpy()  # [n_prompts]

    # Estimate I(Q(h_l); Y)
    mi_score = estimate_mutual_information(X_quant, Y_task)
    option_b_scores.append(mi_score)
    print(f"  Layer {layer_idx:2d}: I(Q(h); Y) = {mi_score:.4f} bits")

option_b_scores = np.array(option_b_scores)

# ─────────────────────────────────────────────
# STEP 6: Spearman Rank Correlation — The Verdict
# ─────────────────────────────────────────────
print("\n[Stage 3] Spearman Rank Correlation Analysis...")

spearman_a, p_val_a = stats.spearmanr(option_a_scores, kl_ground_truth)
spearman_b, p_val_b = stats.spearmanr(option_b_scores, kl_ground_truth)

print(f"\n  Option A (Weight Norm):      ρ = {spearman_a:.4f}  (p = {p_val_a:.4f})")
print(f"  Option B (Activation MI):    ρ = {spearman_b:.4f}  (p = {p_val_b:.4f})")
print(f"\n  Winner: {'Option B — Activation MI ✓' if spearman_b > spearman_a else 'Option A — Weight Norm'}")

# ─────────────────────────────────────────────
# STEP 7: Visualization — Publication-Style
# ─────────────────────────────────────────────
print("\n[Stage 4] Generating figures...")

fig = plt.figure(figsize=(18, 12), facecolor="#0d0d0d")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

C_BG   = "#0d0d0d"
C_CARD = "#1a1a1a"
C_A    = "#ff6b6b"   # Option A — red
C_B    = "#00d4aa"   # Option B — teal (winner)
C_GT   = "#f5f5f5"   # Ground truth — white
C_GRID = "#2a2a2a"
FONT   = {"family": "monospace"}

layers = np.arange(n_layers)

def style_ax(ax, title):
    ax.set_facecolor(C_CARD)
    ax.tick_params(colors="#888888", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_GRID)
    ax.grid(True, color=C_GRID, linewidth=0.5, alpha=0.8)
    ax.set_title(title, color="#cccccc", fontsize=10, pad=8, **FONT)
    ax.xaxis.label.set_color("#888888")
    ax.yaxis.label.set_color("#888888")

# ── Plot 1: Ground Truth KL-D Map ──────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Ground Truth: KL-D per Layer")
ax1.bar(layers, kl_ground_truth, color=C_GT, alpha=0.85, width=0.7)
ax1.set_xlabel("Layer Index", **FONT)
ax1.set_ylabel("KL-Divergence ↑ = more degradation", **FONT)
ax1.axhline(kl_ground_truth.mean(), color="#ffcc44", linewidth=1,
            linestyle="--", label="mean KL-D")
ax1.legend(fontsize=8, facecolor=C_CARD, labelcolor="#cccccc")

# ── Plot 2: Option A Scores ─────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Option A: Weight Frobenius Norm")
ax2.bar(layers, option_a_scores / option_a_scores.max(), color=C_A, alpha=0.85, width=0.7)
ax2.set_xlabel("Layer Index", **FONT)
ax2.set_ylabel("Normalized Score", **FONT)
ax2.text(0.05, 0.92, f"ρ = {spearman_a:.3f}", transform=ax2.transAxes,
         color=C_A, fontsize=11, **FONT, fontweight="bold")

# ── Plot 3: Option B Scores ─────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, "Option B: Activation MI — I(Q(h); Y)")
ax3.bar(layers, option_b_scores / (option_b_scores.max() + 1e-8),
        color=C_B, alpha=0.85, width=0.7)
ax3.set_xlabel("Layer Index", **FONT)
ax3.set_ylabel("Normalized Score", **FONT)
ax3.text(0.05, 0.92, f"ρ = {spearman_b:.3f}", transform=ax3.transAxes,
         color=C_B, fontsize=11, **FONT, fontweight="bold")

# ── Plot 4: Scatter — Option A vs KL-D ─────────────────
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4, "Option A vs Ground Truth KL-D")
ax4.scatter(option_a_scores, kl_ground_truth, color=C_A, alpha=0.85, s=60, zorder=3)
m, b = np.polyfit(option_a_scores, kl_ground_truth, 1)
x_line = np.linspace(option_a_scores.min(), option_a_scores.max(), 100)
ax4.plot(x_line, m * x_line + b, color=C_A, linewidth=1.5, alpha=0.6, linestyle="--")
ax4.set_xlabel("Weight Norm (Option A)", **FONT)
ax4.set_ylabel("KL-Divergence", **FONT)
ax4.text(0.05, 0.9, f"ρ = {spearman_a:.3f}\np = {p_val_a:.3f}",
         transform=ax4.transAxes, color=C_A, fontsize=9, **FONT)

# ── Plot 5: Scatter — Option B vs KL-D ─────────────────
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5, "Option B vs Ground Truth KL-D")
ax5.scatter(option_b_scores, kl_ground_truth, color=C_B, alpha=0.85, s=60, zorder=3)
m2, b2 = np.polyfit(option_b_scores, kl_ground_truth, 1)
x_line2 = np.linspace(option_b_scores.min(), option_b_scores.max(), 100)
ax5.plot(x_line2, m2 * x_line2 + b2, color=C_B, linewidth=1.5, alpha=0.6, linestyle="--")
ax5.set_xlabel("Activation MI — I(Q(h);Y)  (Option B)", **FONT)
ax5.set_ylabel("KL-Divergence", **FONT)
ax5.text(0.05, 0.9, f"ρ = {spearman_b:.3f}\np = {p_val_b:.3f}",
         transform=ax5.transAxes, color=C_B, fontsize=9, **FONT)

# ── Plot 6: Summary Bar — Spearman Comparison ──────────
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6, "Verdict: Spearman ρ Comparison")
bars = ax6.bar(
    ["Option A\nWeight Norm\n(T = Q(W))", "Option B\nActivation MI\n(T = Q(h))"],
    [abs(spearman_a), abs(spearman_b)],
    color=[C_A, C_B], alpha=0.9, width=0.5
)
ax6.set_ylabel("|Spearman ρ| with KL-D Ground Truth", **FONT)
ax6.set_ylim(0, 1.1)
ax6.axhline(0.5, color="#ffcc44", linewidth=1, linestyle="--", alpha=0.7, label="threshold ρ=0.5")
ax6.legend(fontsize=8, facecolor=C_CARD, labelcolor="#cccccc")
for bar, val in zip(bars, [abs(spearman_a), abs(spearman_b)]):
    ax6.text(bar.get_x() + bar.get_width()/2, val + 0.03,
             f"ρ = {val:.3f}", ha="center", color="#ffffff", fontsize=10, **FONT, fontweight="bold")

# Winner annotation
winner_label = "T = Q(h) is the valid IB bottleneck" if spearman_b > spearman_a else "Unexpected: review assumptions"
winner_color = C_B if spearman_b > spearman_a else C_A
ax6.text(0.5, -0.18, winner_label, transform=ax6.transAxes,
         ha="center", color=winner_color, fontsize=9, **FONT, fontstyle="italic")

# ── Main Title ──────────────────────────────────────────
fig.text(0.5, 0.97,
         "IB T-Selection Experiment: Which Compressed Representation Best Predicts Silent Degradation?",
         ha="center", color="#ffffff", fontsize=13, fontweight="bold", **FONT)
fig.text(0.5, 0.945,
         f"Model: GPT-2 Small  |  Calibration: {N_CALIBRATION} reasoning prompts  |  Quantization: INT8 per-tensor symmetric",
         ha="center", color="#666666", fontsize=9, **FONT)

plt.savefig("/mnt/user-data/outputs/ib_t_selection_results.png",
            dpi=150, bbox_inches="tight", facecolor=C_BG)
plt.close()
print("Figure saved.")

# ─────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)
print(f"  Ground truth layers: {n_layers}")
print(f"  Max KL-D layer:      Layer {np.argmax(kl_ground_truth)} (KL = {kl_ground_truth.max():.6f})")
print(f"  Option A ρ: {spearman_a:.4f} (p={p_val_a:.4f})  — Weight Frobenius Norm")
print(f"  Option B ρ: {spearman_b:.4f} (p={p_val_b:.4f})  — Activation Mutual Information")
print()
if spearman_b > spearman_a:
    print("  CONCLUSION: T = Q(h) (quantized activations) is the valid IB bottleneck.")
    print("  Option B shows higher Spearman correlation with ground truth KL-D,")
    print("  confirming the theoretical result that I(T;X) requires T to be")
    print("  input-dependent — which weights are not, but activations are.")
else:
    print("  UNEXPECTED RESULT: Review MI estimation or calibration set.")
print("=" * 60)
