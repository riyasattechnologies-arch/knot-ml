"""Operations on Jones polynomials stored as (min_degree, coefficient array)."""
from __future__ import annotations

import numpy as np


def evaluate_jones(coeffs, min_degree: int, t):
    """Evaluate V(t) = sum_i c_i t^(min_degree + i). ``t`` may be real or complex."""
    powers = min_degree + np.arange(len(coeffs))
    return np.sum(np.asarray(coeffs) * np.power(complex(t), powers))


def jones_span(min_degree: int, max_degree: int) -> int:
    """Span of the Jones polynomial: highest power minus lowest power."""
    return max_degree - min_degree
