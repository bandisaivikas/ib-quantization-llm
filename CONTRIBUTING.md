# Contributing

Thanks for your interest in IB-Quantization-LLM.

## Ways to Contribute

- **Report a bug** — open an issue using the Bug Report template.
- **Suggest an experiment or feature** — open an issue using the Feature Request template.
- **Submit code** — fork the repo, make your change on a branch, and open a pull request.

## Development Setup

```bash
git clone https://github.com/bandisaivikas/IB-Quantization-LLM.git
cd IB-Quantization-LLM
pip install -r requirements.txt
```

Check the MI estimator works:

```bash
python scripts/mi_estimator.py
```

## Rules for Experimental Code

This is a research repo, so correctness of numbers matters more than anything else.

1. **Use the shared evaluation harness.** Every method must be evaluated with `scripts/evaluate.py`. Do not write inline eval code in notebooks.
2. **Never report a raw MI value.** Use `estimate_mi()` from `scripts/mi_estimator.py`, which returns a noise floor and a significance flag.
3. **Compare at matched bit-budgets.** A method using more bits is not a fair comparison point.
4. **Include the random-allocation control** when proposing a new allocation method.
5. **Do not add results you did not compute.** Every number in the README must trace back to a file in `results/`.

## Pull Request Checklist

- [ ] Code runs from a clean environment with `requirements.txt`
- [ ] New results are saved as JSON in `results/`
- [ ] README updated if results or structure changed
- [ ] No API keys, tokens, or model weights committed

## Style

- Python 3.10+
- Clear function names and docstrings
- Keep notebooks for exploration; move reusable logic into `scripts/`
