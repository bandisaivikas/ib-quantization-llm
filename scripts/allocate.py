"""
Bit allocation: turn per-layer scores into per-layer bitwidths hitting a
target average.

This is the heart of framing B. The allocator is shared; only the *scores*
differ between methods:

  ib          -> your beta scores       (results/llama31_results.json)
  random      -> shuffled beta scores   (control: same bit histogram, wrong layers)
  sensitivity -> layer-wise KL only     (prior-art style heuristic baseline)
  uniform     -> all layers equal       (standard GPTQ)

The random control is the most important experiment you have. If IB does not
beat random at a matched average, the premise needs rethinking.
"""

import json
import random
from collections import Counter


def _weighted_avg_bits(bits_per_layer, params_per_layer=None):
    """Average bits weighted by parameter count (this is what memory tracks)."""
    if params_per_layer is None:
        return sum(bits_per_layer) / len(bits_per_layer)
    total_params = sum(params_per_layer)
    return sum(b * p for b, p in zip(bits_per_layer, params_per_layer)) / total_params


def allocate_bits(scores, target_avg_bits, bit_choices=(3, 4, 5),
                  params_per_layer=None):
    """
    Assign bitwidths so that higher-score layers get more bits, subject to
    hitting target_avg_bits (parameter-weighted if params_per_layer given).

    Strategy: walk layers in descending score order and give each the highest
    bitwidth it can afford, while reserving enough budget to keep every
    not-yet-assigned layer at the minimum. This concentrates precision on the
    top-scoring layers rather than levelling everything up uniformly (which
    would collapse to a single bitwidth and defeat the point).
    """
    bit_choices = sorted(bit_choices)
    lo = bit_choices[0]
    n = len(scores)

    if params_per_layer is None:
        params_per_layer = [1] * n

    budget = target_avg_bits * sum(params_per_layer)
    min_cost = sum(lo * p for p in params_per_layer)
    if min_cost > budget:
        raise ValueError(
            f"target_avg_bits={target_avg_bits} is below the minimum "
            f"achievable {lo} with bit_choices={bit_choices}"
        )

    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    bits = [lo] * n
    spent = 0.0

    for pos, i in enumerate(order):
        remaining = order[pos + 1:]
        reserve = sum(lo * params_per_layer[j] for j in remaining)
        # Highest choice affordable without starving the remaining layers
        chosen = lo
        for b in bit_choices:
            if spent + b * params_per_layer[i] + reserve <= budget:
                chosen = b
        bits[i] = chosen
        spent += chosen * params_per_layer[i]

    return bits


def load_beta_scores(path="results/llama31_results.json"):
    with open(path) as f:
        return json.load(f)["beta"]


def make_allocation(method, beta_scores, kl_scores, target_avg_bits,
                    bit_choices=(3, 4, 5), params_per_layer=None, seed=0):
    """Return per-layer bitwidths for the named method."""
    n = len(beta_scores)

    if method == "uniform":
        b = int(round(target_avg_bits))
        return [b] * n

    if method == "ib":
        scores = beta_scores

    elif method == "sensitivity":
        # Prior-art style: raw layer-wise KL sensitivity, no IB derivation.
        # This is the baseline that makes the IB contribution meaningful.
        scores = kl_scores

    elif method == "random":
        # Same score *distribution*, shuffled across layers. Produces the same
        # bit histogram as `ib` but assigned to the wrong layers.
        scores = list(beta_scores)
        random.Random(seed).shuffle(scores)

    else:
        raise ValueError(f"unknown method: {method}")

    return allocate_bits(scores, target_avg_bits, bit_choices, params_per_layer)


def summarise(bits, params_per_layer=None):
    return {
        "avg_bits": round(_weighted_avg_bits(bits, params_per_layer), 4),
        "histogram": dict(sorted(Counter(bits).items())),
        "per_layer": bits,
    }


if __name__ == "__main__":
    import sys
    res_path = sys.argv[1] if len(sys.argv) > 1 else "results/llama31_results.json"
    with open(res_path) as f:
        res = json.load(f)

    beta, kl = res["beta"], res["kl_gt"]
    target = 4.0

    print(f"target average bits: {target}\n")
    for method in ("uniform", "ib", "sensitivity", "random"):
        bits = make_allocation(method, beta, kl, target)
        s = summarise(bits)
        print(f"{method:12s} avg={s['avg_bits']:.3f}  hist={s['histogram']}")
        print(f"             {s['per_layer']}\n")
