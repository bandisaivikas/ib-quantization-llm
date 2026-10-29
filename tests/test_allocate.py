import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from allocate import allocate_bits, make_allocation  # noqa: E402


def test_allocation_hits_target_average():
    scores = list(range(32))
    bits = allocate_bits(scores, 4.0, (3, 4, 5))
    assert sum(bits) / len(bits) == 4.0


def test_higher_scores_get_more_bits():
    scores = list(range(32))
    bits = allocate_bits(scores, 4.0, (3, 4, 5))
    assert bits[-1] >= bits[0]


def test_uniform_allocation():
    bits = make_allocation("uniform", [1.0] * 8, [1.0] * 8, 4.0)
    assert bits == [4] * 8
