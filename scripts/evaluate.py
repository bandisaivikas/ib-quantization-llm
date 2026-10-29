"""
Shared evaluation harness.

EVERY method (FP16 baseline, uniform GPTQ, IB-allocated GPTQ, random-allocated
GPTQ, sensitivity-allocated GPTQ) must call evaluate() from this file.

If baselines are evaluated even slightly differently from your method, the
comparison is worthless. This is the single highest-risk source of a wrong
number in the whole project. Do not inline eval logic in notebooks.
"""

import json
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from datasets import load_dataset


# ----------------------------------------------------------------------------
# Config: pin these once, never vary them across methods.
# ----------------------------------------------------------------------------

EVAL_CONFIG = {
    "wikitext2_split": "test",
    "wikitext2_seqlen": 2048,
    "hellaswag_split": "validation",
    "hellaswag_n": 1000,      # set to None for full split
    "kl_n_prompts": 500,      # must match your beta-score calibration count
    "seed": 0,
}


@dataclass
class EvalResult:
    method: str
    avg_bits: float
    size_mb: float
    hellaswag_acc: float
    wikitext2_ppl: float
    kl_vs_fp16: float
    quant_seconds: float = 0.0

    def to_json(self):
        return json.dumps(asdict(self), indent=2)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

@torch.no_grad()
def wikitext2_perplexity(model, tokenizer, seqlen=2048, device="cuda"):
    """Standard sliding-window perplexity on WikiText-2 test."""
    data = load_dataset("wikitext", "wikitext-2-raw-v1",
                        split=EVAL_CONFIG["wikitext2_split"])
    enc = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    n_chunks = input_ids.numel() // seqlen
    nlls = []
    for i in range(n_chunks):
        chunk = input_ids[:, i * seqlen:(i + 1) * seqlen]
        out = model(chunk, labels=chunk)
        # loss is mean over (seqlen - 1) predicted tokens
        nlls.append(out.loss.float() * (seqlen - 1))

    total_tokens = n_chunks * (seqlen - 1)
    return float(torch.exp(torch.stack(nlls).sum() / total_tokens))


@torch.no_grad()
def hellaswag_accuracy(model, tokenizer, n=None, device="cuda"):
    """
    Length-normalised log-likelihood scoring: for each of the 4 endings, score
    only the continuation tokens and normalise by continuation length.
    """
    data = load_dataset("Rowan/hellaswag", split=EVAL_CONFIG["hellaswag_split"])
    if n is not None:
        data = data.select(range(min(n, len(data))))

    correct = 0
    for ex in data:
        ctx = ex["ctx"]
        scores = []
        for ending in ex["endings"]:
            ctx_ids = tokenizer(ctx, return_tensors="pt").input_ids
            full_ids = tokenizer(ctx + " " + ending, return_tensors="pt").input_ids
            full_ids = full_ids.to(device)
            n_ctx = ctx_ids.shape[1]

            logits = model(full_ids).logits
            # predict token t from position t-1
            logprobs = F.log_softmax(logits[:, :-1].float(), dim=-1)
            targets = full_ids[:, 1:]
            token_lp = logprobs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

            cont_lp = token_lp[:, n_ctx - 1:]          # continuation only
            scores.append(float(cont_lp.sum()) / max(cont_lp.numel(), 1))

        if int(scores.index(max(scores))) == int(ex["label"]):
            correct += 1

    return correct / len(data)


@torch.no_grad()
def kl_vs_reference(model, ref_model, tokenizer, prompts, device="cuda"):
    """
    Mean token-level KL(reference || quantised) over calibration prompts.
    This is the metric your silent-degradation finding is built on, so it must
    be computed identically for every method.
    """
    total_kl, total_tokens = 0.0, 0
    for text in prompts:
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=512).input_ids.to(device)

        ref_logits = ref_model(ids).logits.float()
        q_logits = model(ids).logits.float()

        ref_lp = F.log_softmax(ref_logits, dim=-1)
        q_lp = F.log_softmax(q_logits, dim=-1)

        kl = F.kl_div(q_lp, ref_lp, log_target=True, reduction="sum")
        total_kl += float(kl)
        total_tokens += ids.numel()

    return total_kl / max(total_tokens, 1)


def model_size_mb(model):
    """Reported footprint. For packed quantised models prefer on-disk size."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / (1024 ** 2)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def evaluate(model, tokenizer, ref_model, kl_prompts, method: str,
             avg_bits: float, quant_seconds: float = 0.0,
             device="cuda") -> EvalResult:
    model.eval()
    torch.manual_seed(EVAL_CONFIG["seed"])

    return EvalResult(
        method=method,
        avg_bits=avg_bits,
        size_mb=model_size_mb(model),
        hellaswag_acc=hellaswag_accuracy(
            model, tokenizer, n=EVAL_CONFIG["hellaswag_n"], device=device),
        wikitext2_ppl=wikitext2_perplexity(
            model, tokenizer, seqlen=EVAL_CONFIG["wikitext2_seqlen"], device=device),
        kl_vs_fp16=kl_vs_reference(
            model, ref_model, tokenizer, kl_prompts, device=device),
        quant_seconds=quant_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(
        "Import evaluate() from this module; do not run it standalone.\n"
        "First task: reproduce your existing published numbers with this "
        "harness before adding any baseline."
    )
