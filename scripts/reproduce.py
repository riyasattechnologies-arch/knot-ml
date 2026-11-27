"""Recompute every number reported in the README and save them to results/metrics.json.

Run from the repository root:  python scripts/reproduce.py
Takes about a minute on a laptop. All random seeds are fixed.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from knotml import (cross_validated_error, encode_jones, evaluate_jones, load_knots,  # noqa: E402
                    make_volume_network, mean_relative_error, scan_evaluation_angles)

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "results" / "metrics.json"


def main():
    df = load_knots()
    alt = df["alternating"]
    span = df["jones_max_degree"] - df["jones_min_degree"]
    det_check = [abs(evaluate_jones(c, lo, -1)) for c, lo in zip(df["coeffs"], df["jones_min_degree"])]

    h = load_knots(hyperbolic_only=True)
    y = h["volume"].values
    log_det = np.log(h["determinant"].clip(lower=1)).values
    ha = h[h["alternating"]]
    ratio = ha["volume"] / (2 * np.pi * np.log(ha["determinant"]))

    print("Project 1: theorems and conjecture ...")
    project1 = {
        "n_knots": int(len(df)),
        "n_alternating": int(alt.sum()),
        "n_hyperbolic": int(df["hyperbolic"].sum()),
        "det_equals_abs_V_minus_1": int(np.isclose(det_check, df["determinant"]).sum()),
        "span_equals_crossing_alternating": int((span[alt] == df["crossing_number"][alt]).sum()),
        "span_less_than_crossing_nonalternating": int((span[~alt] < df["crossing_number"][~alt]).sum()),
        "corr_volume_crossing_number": round(float(np.corrcoef(y, h["crossing_number"])[0, 1]), 3),
        "corr_volume_log_det_all": round(float(np.corrcoef(y, log_det)[0, 1]), 3),
        "corr_volume_log_det_alternating": round(float(np.corrcoef(ha["volume"], np.log(ha["determinant"]))[0, 1]), 3),
        "ckp_bound_holds": int((ratio < 1).sum()),
        "ckp_n_tested": int(len(ha)),
        "ckp_max_ratio": round(float(ratio.max()), 3),
        "ckp_closest_knot": ha.loc[ratio.idxmax(), "name"],
    }

    print("Project 2: neural network and baselines ...")
    X = encode_jones(h["coeffs"], h["jones_min_degree"], h["jones_max_degree"])
    pred, err_nn = cross_validated_error(make_volume_network(), X, y)
    is_alt = h["alternating"].values
    _, err_det = cross_validated_error(LinearRegression(), log_det.reshape(-1, 1), y)
    _, err_lin = cross_validated_error(LinearRegression(), X, y)

    print("Project 2: angle scan ...")
    angles = np.linspace(0.05, 1.0, 96)
    scan = scan_evaluation_angles(h["coeffs"], h["jones_min_degree"], y, angles)
    i34 = int(np.argmin(np.abs(angles - 0.75)))
    project2 = {
        "n_knots": int(len(h)),
        "neural_network_error": round(err_nn, 4),
        "neural_network_error_alternating": round(mean_relative_error(pred[is_alt], y[is_alt]), 4),
        "neural_network_error_nonalternating": round(mean_relative_error(pred[~is_alt], y[~is_alt]), 4),
        "linear_log_det_error": round(err_det, 4),
        "linear_coefficients_error": round(err_lin, 4),
        "best_angle_over_pi": round(float(angles[scan.argmin()]), 2),
        "best_angle_error": round(float(scan.min()), 4),
        "angle_3pi_over_4_error": round(float(scan[i34]), 4),
        "angle_pi_error": round(float(scan[-1]), 4),
    }

    results = {"project1_invariant_analysis": project1, "project2_volume_prediction": project2,
               "evaluation": "5-fold cross-validation, shuffled, random_state=0"}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
