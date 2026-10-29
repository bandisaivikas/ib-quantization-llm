# Accessibility

## Commitment

This project aims to be usable by as many people as possible, including those with limited hardware.

## Supported Environments

- **CPU-only:** GPT-2 experiments run on CPU (~20 min).
- **Apple Silicon:** MPS backend and optional MLX path.
- **Free cloud GPUs:** Google Colab and Kaggle notebooks are supported for Llama experiments.
- **NVIDIA GPUs:** CUDA path for full Llama 3.1 8B pipeline.

## Known Limitations

- Llama 3.1 8B needs at least 16 GB RAM or a GPU, and HuggingFace access approval.
- Figures use colour to show differences; values are also printed in tables and JSON in `results/`.
- Notebooks may be hard to navigate with screen readers; reusable logic is also available as plain Python in `scripts/`.

## Reporting Barriers

If something in this project is hard to use or access, please open an issue describing the problem.
