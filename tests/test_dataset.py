"""Checks on the full KnotInfo dataset: known theorems must hold for every knot."""
import numpy as np
import pytest

from knotml import evaluate_jones, load_knots


@pytest.fixture(scope="module")
def knots():
    return load_knots()


def test_dataset_size(knots):
    assert len(knots) == 2977
    assert knots["hyperbolic"].sum() == 2970


def test_jones_at_one_is_one(knots):
    vals = [evaluate_jones(c, lo, 1) for c, lo in zip(knots["coeffs"], knots["jones_min_degree"])]
    assert np.allclose(vals, 1)


def test_determinant_equals_abs_jones_at_minus_one(knots):
    vals = [abs(evaluate_jones(c, lo, -1)) for c, lo in zip(knots["coeffs"], knots["jones_min_degree"])]
    assert np.allclose(vals, knots["determinant"])


def test_kauffman_murasugi_thistlethwaite(knots):
    span = knots["jones_max_degree"] - knots["jones_min_degree"]
    alt = knots["alternating"]
    assert (span[alt] == knots["crossing_number"][alt]).all()
    assert (span[~alt] < knots["crossing_number"][~alt]).all()


def test_ckp_bound_holds_for_alternating_hyperbolic(knots):
    a = knots[knots["alternating"] & knots["hyperbolic"]]
    assert (a["volume"] < 2 * np.pi * np.log(a["determinant"])).all()
