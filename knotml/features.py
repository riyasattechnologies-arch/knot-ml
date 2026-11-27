"""Turn Jones polynomials into fixed-length vectors for machine learning."""
from __future__ import annotations

import numpy as np

N_COEFFS = 13   # longest Jones polynomial among knots with up to 12 crossings


def encode_jones(coeff_arrays, min_degrees, max_degrees, n_coeffs: int = N_COEFFS) -> np.ndarray:
    """Encode each polynomial as ``[min_degree, max_degree, c0, ..., c_{n-1}]``, zero-padded.

    Returns an array of shape (n_knots, n_coeffs + 2).
    """
    rows = []
    for c, lo, hi in zip(coeff_arrays, min_degrees, max_degrees):
        c = np.asarray(c, dtype=float)
        if len(c) > n_coeffs:
            raise ValueError(f"polynomial has {len(c)} coefficients, more than n_coeffs={n_coeffs}")
        rows.append(np.concatenate([[lo, hi], np.pad(c, (0, n_coeffs - len(c)))]))
    return np.array(rows)
