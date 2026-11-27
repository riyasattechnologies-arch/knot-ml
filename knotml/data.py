"""Build and load a clean table of knot invariants from the KnotInfo database.

KnotInfo (https://knotinfo.math.indiana.edu) is the standard table of knot invariants,
maintained by C. Livingston and A. H. Moore. The ``database_knotinfo`` package ships a copy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "knots.csv"


def parse_jones_vector(text: str):
    """Parse KnotInfo's Jones polynomial vector ``[min_degree, max_degree, c0, c1, ...]``.

    Example: the figure-eight knot 4_1 is ``[-2, 2, 1, -1, 1, -1, 1]``, meaning
    V(t) = t^-2 - t^-1 + 1 - t + t^2.

    Returns ``(min_degree, max_degree, coefficients)`` or ``None`` if the field is empty.
    """
    if not text:
        return None
    nums = [int(x) for x in text.strip("[]").split(",")]
    return nums[0], nums[1], nums[2:]


def _to_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def build_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Extract every prime knot with Jones polynomial data (3 to 12 crossings) and save it as CSV."""
    from database_knotinfo import link_list

    rows = []
    for k in link_list()[1:]:            # row 0 of the package is a header row
        if not k["crossing_number"] or int(k["crossing_number"]) < 3:
            continue                     # skip the unknot
        jones = parse_jones_vector(k["jones_polynomial_vector"])
        if jones is None:                # Jones data covers knots up to 12 crossings
            continue
        lo, hi, coeffs = jones
        rows.append({
            "name": k["name"],
            "crossing_number": int(k["crossing_number"]),
            "alternating": k["alternating"] == "Y",
            "determinant": int(k["determinant"]),
            "signature": int(k["signature"]),
            "three_genus": _to_float(k["three_genus"]),
            "volume": _to_float(k["volume"]),          # 0 means not hyperbolic
            "jones_min_degree": lo,
            "jones_max_degree": hi,
            "jones_coefficients": " ".join(map(str, coeffs)),
            "jones_polynomial": k["jones_polynomial"],
        })
    df = pd.DataFrame(rows)
    df["hyperbolic"] = df["volume"].fillna(0) > 0
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_knots(path: Path = DATA_PATH, hyperbolic_only: bool = False) -> pd.DataFrame:
    """Load the knot table (building it first if needed) with coefficients as integer arrays."""
    if not Path(path).exists():
        build_dataset(path)
    df = pd.read_csv(path)
    df["coeffs"] = df["jones_coefficients"].str.split().apply(lambda v: np.array(v, dtype=int))
    if hyperbolic_only:
        df = df[df["hyperbolic"]].reset_index(drop=True)
    return df
