"""Unit tests for parsing, evaluating and encoding Jones polynomials."""
import numpy as np
import pytest

from knotml import encode_jones, evaluate_jones, jones_span, parse_jones_vector


def test_parse_figure_eight():
    lo, hi, coeffs = parse_jones_vector("[-2, 2, 1, -1, 1, -1, 1]")
    assert (lo, hi, coeffs) == (-2, 2, [1, -1, 1, -1, 1])


def test_parse_empty_returns_none():
    assert parse_jones_vector("") is None


def test_evaluate_figure_eight_at_minus_one_is_determinant():
    # V_{4_1}(t) = t^-2 - t^-1 + 1 - t + t^2, so |V(-1)| = 5 = det(4_1)
    assert abs(evaluate_jones([1, -1, 1, -1, 1], -2, -1)) == pytest.approx(5)


def test_evaluate_at_one_is_one():
    # V(1) = 1 for every knot
    assert evaluate_jones([1, -1, 1, -1, 1], -2, 1) == pytest.approx(1)


def test_span():
    assert jones_span(-2, 2) == 4


def test_encode_pads_to_fixed_length():
    X = encode_jones([[1, -1, 1, -1, 1], [1, 0, 1, -1]], [-2, 1], [2, 4])
    assert X.shape == (2, 15)
    assert list(X[0][:7]) == [-2, 2, 1, -1, 1, -1, 1]
    assert np.all(X[0][7:] == 0)


def test_encode_rejects_too_long():
    with pytest.raises(ValueError):
        encode_jones([[1] * 20], [0], [19])
