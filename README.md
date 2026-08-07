# IB-Guided LLM Quantization

**Information Bottleneck-guided selective quantization for LLMs — investigating silent inference degradation in Llama 3.1 8B.**

> *Bandi Saivikas · IIIT Naya Raipur · B.Tech Data Science & AI*  
> Target: EMNLP 2026

---

## Motivation

Standard quantization methods (INT4, INT8) treat all layers equally. This research shows that **KL-divergence sensitivity varies by 121× across Llama 3.1 8B's 32 layers** — so uniform quantization silently destroys reasoning quality in the most sensitive layers while wasting precision on robust ones.

The Information Bottleneck (IB) framework provides a principled way to measure each layer's information compression, identify which layers act as critical bottlenecks, and allocate bits accordingly.

---

## Key Results (Llama 3.1 8B Instruct, 7-bit budget)

| Method | HellaSwag Acc. | WikiText-2 PPL | KL Divergence |
|---|---|---|---|
| FP32 Baseline | 54.5% | 8.9 | — |
| Uniform INT8 | 56.0% | 9.7 | baseline |
| GPTQ-style 7-bit | 33.5% | 2,780 | high |
| AWQ-style 7-bit | 29.0% | 59,409 | very high |
| Random 7-bit | 35.0% | — | — |
| **IB-7bit (ours)** | **46.5%** | **42.3** | **lowest** |
| Uniform INT4 | 23.0% | 1,007,348 | extreme |

IB-guided allocation cuts KL divergence by **81.5% vs GPTQ** and **90.9% vs AWQ** at the same 7-bit memory budget.

---

## Experiments

### Stage 1 — Silent Degradation Discovery (`gpt2_kl_accuracy_all_configs.ipynb`)
Establishes that perplexity and accuracy metrics can appear stable while token-distribution KL divergence degrades significantly — the "silent" failure mode that motivates this work.

### Stage 2 — IB Framework Validation (`stage2_ib_validation.ipynb`, `stage2_ib_validation.py`)
Validates the IB framework on GPT-2. Confirms that quantized weight matrices can serve as valid IB bottlenecks (T-selection: T = Q(W) weights, not activations). Produces layer-wise KL-divergence maps and information plane visualizations.

### Stage 3 — IB Algorithm on GPT-2 (`stage3_ib_algorithm.ipynb`)
Implements the full β-score allocation algorithm on GPT-2. Each layer receives a bit-width proportional to its IB sensitivity score. Validates the Pareto frontier between compression and information loss.

### Stage 4 — Llama 3.1 8B Pipeline (`llama31_ib_complete_pipeline.ipynb`)
Scales the full pipeline to Llama 3.1 8B Instruct on Lightning AI. Runs all four experiments end-to-end:
- **EXP 1**: Ground-truth layer-wise KL-divergence map (32 layers × 500 calibration prompts from GSM8K)
- **EXP 4**: Layer information plane displacement (I(X;T) vs I(T;Y) under INT4 quantization)
- **EXP 5**: T-selection validation (confirms T = Q(W) is the correct bottleneck)
- **7-bit Showdown**: IB vs GPTQ vs AWQ vs Random at identical memory

---

## Repository Structure

```
├── notebooks/
│   ├── llama31_ib_complete_pipeline.ipynb   # Main pipeline: Llama 3.1 8B (Lightning AI)
│   ├── llama3_ib_pipeline.ipynb             # Earlier Llama 3 8B version (local Apple Silicon)
│   ├── llama_windows_gpu.ipynb              # Windows/CUDA port
│   ├── stage2_ib_validation.ipynb           # Stage 2: GPT-2 IB framework validation
│   ├── stage3_ib_algorithm.ipynb            # Stage 3: GPT-2 full IB algorithm
│   ├── gpt2_kl_accuracy_all_configs.ipynb   # Stage 1: Silent degradation discovery
│   ├── gpt2_demo_for_professor.ipynb        # Clean demo: GPT-2 IB vs uniform quantization
│   └── ib_vs_original.ipynb                 # IB vs baseline comparison
├── scripts/
│   ├── ib_t_selection_experiment.py         # T-selection validation script (GPT-2)
│   └── stage2_ib_validation.py              # Stage 2 validation script
├── figures/
│   ├── plots/                               # Llama 3.1 8B result figures (5 panels)
│   ├── plotsfig1_silent_degradation.png     # Silent degradation demo
│   ├── plotsfig2_layer_kl_map.png           # Layer-wise KL-D heatmap
│   ├── plotsfig3_ib_vs_original.png         # IB vs original information plane
│   ├── plotsfig4_pareto.png                 # Pareto frontier
│   ├── plotsfig5_beta_scores.png            # β-score layer allocation
│   ├── gpt2_all_quantization_results.png    # GPT-2 quantization sweep
│   ├── ib_quantization_proofs.png           # Mathematical framework
│   └── stage2_ib_framework_validation.png   # Stage 2 validation summary
├── results/
│   ├── llama31_results.json                 # Full Llama 3.1 8B experiment outputs
│   └── stage3_results.json                  # GPT-2 Stage 3 outputs
└── llm-memory-research/
    └── requirements.txt                     # Dependencies
```

---

## Method

The IB-guided quantization assigns each layer a β-score:

```
β_l = α · KL_l^(EXP1) + (1-α) · Δplane_l^(EXP4)
```

where:
- `KL_l^(EXP1)` — KL divergence when layer *l* alone is quantized (direct sensitivity)
- `Δplane_l^(EXP4)` — information plane displacement under full INT4 quantization
- `α = 1.0` — optimal weighting found by ablation (EXP1 signal dominates at this scale)

Layers with high β are kept at INT8; layers with low β are quantized to INT4. The mixed-precision allocation targets an average of 7 bits with the same total memory as GPTQ/AWQ.

---

## Reproducing the Results

### Requirements

```bash
pip install transformers datasets scipy matplotlib torch huggingface_hub
```

For Apple Silicon (MLX):
```bash
pip install mlx-lm
```

### Running the pipeline

**GPT-2 experiments (local, ~20 min total):**
```bash
# T-selection validation
python scripts/ib_t_selection_experiment.py

# Full Stage 2 validation
python scripts/stage2_ib_validation.py
```

Then open the notebooks in order: `stage2_ib_validation.ipynb` → `stage3_ib_algorithm.ipynb`.

**Llama 3.1 8B (requires ≥16 GB RAM or Lightning AI GPU):**

Open `llama31_ib_complete_pipeline.ipynb` and set your HuggingFace token:
```python
login(token="YOUR_HUGGINGFACE_TOKEN")
```
Run cells top to bottom. Full pipeline takes ~90 min on an A10G (Lightning AI).

> **Note:** Llama 3.1 8B requires HuggingFace access approval at [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

---

## Findings

1. **Layer 0 is the critical bottleneck** — it exhibits the highest KL-divergence sensitivity (4.2×, vs 0.1× for stable mid-layers). Quantizing it to INT4 alone accounts for most of the model's quality degradation.

2. **121× KL range across layers** — the variation in sensitivity dwarfs what any uniform scheme can account for, motivating selective allocation.

3. **Silent degradation is real** — standard perplexity and accuracy metrics fail to detect significant KL-divergence degradation until it is catastrophic. The KL map catches it early.

4. **IB-7bit dominates at the same memory** — 46.5% HellaSwag vs 33.5% (GPTQ) and 29% (AWQ), with perplexity 42 vs 2,780 and 59,409.

5. **T = Q(W) is the correct bottleneck** — quantized weights, not activations, are the valid IB bottleneck variable. Activation-based T-selection gives statistically insignificant mutual information estimates.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{bandi2026ibquantization,
  title   = {Information Bottleneck-Guided Selective Quantization for LLMs},
  author  = {Bandi, Saivikas},
  year    = {2026},
  note    = {Under submission, EMNLP 2026}
}
```
