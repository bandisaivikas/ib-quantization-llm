# IB-Quantization-LLM

Information-Bottleneck guided mixed-precision quantization for LLMs.

## Key Finding

**KL-divergence sensitivity varies 121× across Llama 3.1 8B's 32 layers.** Layer 0 dominates at 4.2× while mid-layers sit at ~0.1×. Uniform quantization ignores this; IB-guided allocation exploits it.

## The Problem: Silent Degradation

Standard metrics (perplexity, accuracy) stay stable while token-distribution KL divergence blows up. A model can look fine on benchmarks while its output distribution has silently drifted. This repo detects and addresses that.

## Status

| Component | Status |
|-----------|--------|
| Silent degradation discovery | ✅ Validated |
| 121× layer-sensitivity map | ✅ Validated on Llama 3.1 8B |
| β-score allocation algorithm | ⚠️ Bug found — see below |
| GPTQ/AWQ baseline comparison | ❌ Not yet run |

### Critical Bug Found (Sept 2026)

The MI estimator used to compute `arrow_len` (the information-plane term) was broken:

```
mutual_information(X, Y, n_bins=5)  # with n=20 samples
```

**Result:** Reported MI = 0.64 on pure independent noise (true = 0). The ablation correctly zeroed the term — but because the estimate was garbage, not because information-plane geometry is useless.

**Fix:** `scripts/mi_estimator.py` now uses:
- KSG (k-nearest-neighbour) estimator instead of histogram
- Adaptive bin count when using histogram
- Mandatory shuffle noise floor — if not 2σ above shuffle baseline, treat as zero

**Next step:** Rerun Stage 2 with the fixed estimator and 500+ calibration samples.

## Repository Structure

```
├── notebooks/
│   ├── llama31_ib_complete_pipeline.ipynb   # Main: Llama 3.1 8B
│   ├── stage2_ib_validation.ipynb           # GPT-2 IB framework validation
│   ├── stage3_ib_algorithm.ipynb            # GPT-2 full IB algorithm
│   ├── gpt2_kl_accuracy_all_configs.ipynb   # Stage 1: silent degradation
│   ├── gpt2_demo_for_professor.ipynb        # Clean demo
│   ├── ib_vs_original.ipynb                 # IB vs baseline comparison
│   └── ib_mi_diagnostics.ipynb              # MI estimator bug diagnosis
├── scripts/
│   ├── mi_estimator.py                      # Fixed MI estimator with noise floor
│   ├── allocate.py                          # Bit allocation for framing B
│   ├── evaluate.py                          # Shared evaluation harness
│   ├── ib_t_selection_experiment.py         # T-selection validation
│   └── stage2_ib_validation.py              # Stage 2 validation
├── figures/
│   └── plots/                               # Llama 3.1 8B result figures
├── results/
│   ├── llama31_results.json                 # Llama 3.1 8B experiment outputs
│   └── stage3_results.json                  # GPT-2 Stage 3 outputs
├── legacy/                                  # Superseded notebooks
└── tests/                                   # pytest checks run by GitHub Actions
```

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

For Apple Silicon (MLX) — optional:
```bash
pip install -r requirements-apple-silicon.txt
```

### Run the diagnostics

```bash
python scripts/mi_estimator.py
```

This reproduces the MI estimator failure and verifies the fix.

### Run GPT-2 experiments (~20 min)

```bash
python scripts/ib_t_selection_experiment.py
python scripts/stage2_ib_validation.py
```

Then open `notebooks/stage2_ib_validation.ipynb` → `notebooks/stage3_ib_algorithm.ipynb`.

### Llama 3.1 8B (requires ≥16 GB RAM or Lightning AI GPU)

Open `notebooks/llama31_ib_complete_pipeline.ipynb` and set your HuggingFace token.

## Results

| Model | Method | Avg Bits | HellaSwag | WikiText-2 PPL | KL vs FP32 |
|-------|--------|----------|-----------|----------------|------------|
| Llama 3.1 8B | FP32 Baseline | 32 | 54.5% | 8.9 | — |
| Llama 3.1 8B | Uniform INT8 | 8 | 56.0% | 9.7 | baseline |
| Llama 3.1 8B | Random 7-bit | 7 | 35.0% | — | — |
| Llama 3.1 8B | **IB-7bit** | 7 | **46.5%** | **42.3** | **lowest** |
| Llama 3.1 8B | Uniform INT4 | 4 | 23.0% | 1,007,348 | extreme |

> **Read by matched bit-budget.** INT8 beats IB-7bit because it uses more bits — not a fair comparison. At matched 7-bit budget, IB allocation beats random by 11.5 points.

> **Not yet compared against GPTQ or AWQ.** That's the next step after fixing the MI estimator.

## Planned Work

1. **Rerun Stage 2 with fixed MI estimator** — use `estimate_mi()` from `scripts/mi_estimator.py` with 500+ samples
2. **Redo α sweep** — check if α moves off 1.0 now that the plane term isn't garbage
3. **GPTQ/AWQ comparison (Framing B)** — use β-scores to *allocate* bitwidths with GPTQ as the underlying quantizer, not compete against it

## Citation

If you use this work, please cite:

```bibtex
@misc{bandi2026ibquant,
  author = {Bandi, Sai Vikas},
  title = {Information-Bottleneck Guided LLM Quantization},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/bandisaivikas/IB-Quantization-LLM}
}
```

## License

MIT
