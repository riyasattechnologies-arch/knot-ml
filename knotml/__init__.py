"""knotml: data analysis and machine learning on knot invariants from KnotInfo."""

from .data import DATA_PATH, build_dataset, load_knots, parse_jones_vector
from .jones import evaluate_jones, jones_span
from .features import encode_jones
from .models import (cross_validated_error, make_kfold, make_volume_network,
                     mean_relative_error, scan_evaluation_angles)

__version__ = "1.0.0"

__all__ = [
    "DATA_PATH", "build_dataset", "load_knots", "parse_jones_vector",
    "evaluate_jones", "jones_span", "encode_jones",
    "make_kfold", "make_volume_network", "mean_relative_error",
    "cross_validated_error", "scan_evaluation_angles",
]
