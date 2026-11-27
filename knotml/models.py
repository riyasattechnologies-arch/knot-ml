"""Models and evaluation for predicting hyperbolic volume."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .jones import evaluate_jones


def make_kfold(n_splits: int = 5, seed: int = 0) -> KFold:
    """Shuffled k-fold split used for every reported number."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def make_volume_network(seed: int = 0):
    """Two hidden layers of 100 ReLU units (the size used by Jejjala, Kar and Parrikar, 2019)."""
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=2000, early_stopping=True, random_state=seed),
    )


def mean_relative_error(pred, true) -> float:
    """Average of |predicted - true| / true."""
    pred, true = np.asarray(pred), np.asarray(true)
    return float(np.mean(np.abs(pred - true) / true))


def cross_validated_error(model, X, y, cv=None):
    """Cross-validated predictions (each from a model that never saw that knot) and their error."""
    cv = cv or make_kfold()
    pred = cross_val_predict(model, X, y, cv=cv)
    return pred, mean_relative_error(pred, y)


def scan_evaluation_angles(coeff_arrays, min_degrees, y, angles_over_pi, cv=None):
    """For each angle theta, fit volume ~ a * log(|V(e^{i theta})| + 1) + b and return its CV error."""
    cv = cv or make_kfold()
    errors = []
    for k in angles_over_pi:
        t = np.exp(1j * k * np.pi)
        feature = np.log(np.array([abs(evaluate_jones(c, lo, t)) for c, lo in zip(coeff_arrays, min_degrees)]) + 1)
        errors.append(cross_validated_error(LinearRegression(), feature.reshape(-1, 1), y, cv)[1])
    return np.array(errors)
