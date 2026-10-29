import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from mi_estimator import estimate_mi, mi_histogram  # noqa: E402


def test_independent_data_is_flagged_as_noise():
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal(500), rng.standard_normal(500)
    est = estimate_mi(x, y, estimator="ksg", n_shuffles=20)
    assert not est.significant


def test_correlated_data_is_significant():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    y = 0.6 * x + 0.8 * rng.standard_normal(500)
    est = estimate_mi(x, y, estimator="ksg", n_shuffles=20)
    assert est.significant


def test_histogram_mi_is_non_negative():
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal(200), rng.standard_normal(200)
    assert mi_histogram(x, y) >= 0.0
