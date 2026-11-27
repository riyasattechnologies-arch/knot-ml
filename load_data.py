"""Build a clean table of knot invariants from the KnotInfo database.

KnotInfo (https://knotinfo.math.indiana.edu) is the standard database of knot
invariants, maintained by Chuck Livingston and Allison Moore. The Python package
`database_knotinfo` ships a copy of it.

Run:  python load_data.py
Output: data/knots.csv  (one row per prime knot with up to 12 crossings)
"""
import os

import pandas as pd
from database_knotinfo import link_list


def parse_jones_vector(text):
    """KnotInfo stores the Jones polynomial as [min_degree, max_degree, c0, c1, ...].

    Example: the figure-eight knot 4_1 is [-2, 2, 1, -1, 1, -1, 1], meaning
    V(t) = t^-2 - t^-1 + 1 - t + t^2.
    Returns (min_degree, max_degree, list_of_coefficients), or None if missing.
    """
    if not text:
        return None
    nums = [int(x) for x in text.strip("[]").split(",")]
    return nums[0], nums[1], nums[2:]


def to_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


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
        "three_genus": to_float(k["three_genus"]),
        "volume": to_float(k["volume"]),         # 0 means not hyperbolic
        "jones_min_degree": lo,
        "jones_max_degree": hi,
        "jones_coefficients": " ".join(map(str, coeffs)),
        "jones_polynomial": k["jones_polynomial"],
    })

df = pd.DataFrame(rows)
df["hyperbolic"] = df["volume"].fillna(0) > 0
os.makedirs("data", exist_ok=True)
df.to_csv("data/knots.csv", index=False)
print(f"Saved {len(df)} knots to data/knots.csv")
print(df["crossing_number"].value_counts().sort_index().to_string())
