"""Validated, portable loading of the preprocessing notebook outputs."""

from pathlib import Path
import numpy as np
import pandas as pd


def load_splits(directory, format="csv"):
    directory = Path(directory)
    names = ["X_train_scaled_balanced", "X_test_scaled", "y_train_balanced", "y_test"]
    if format == "csv":
        arrays = [pd.read_csv(directory / f"{name}.csv") for name in names]
        if list(arrays[0].columns) != list(arrays[1].columns):
            raise ValueError("Training/test feature columns must match in order")
    elif format == "npy":
        arrays = [np.load(directory / f"{name}.npy", allow_pickle=False) for name in names]
    else:
        raise ValueError("Expected csv or npy")
    x_train, x_test, y_train, y_test = arrays
    y_train, y_test = np.asarray(y_train).reshape(-1), np.asarray(y_test).reshape(-1)
    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("Feature/label sample counts do not match")
    if x_train.ndim != 2 or x_test.ndim != 2 or x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Expected matching 2D feature matrices")
    for array in (x_train, x_test, y_train, y_test):
        if not np.isfinite(np.asarray(array, dtype=float)).all():
            raise ValueError("Model input must contain finite numbers")
    if len(np.unique(y_train)) != 2:
        raise ValueError("These ROC workflows require a binary target")
    return x_train, x_test, y_train, y_test


def logistic_grid():
    """SAGA L1, L2, elastic-net, and unregularized candidates without invalid pairs."""
    strengths = [0.001, 0.01, 0.1, 1, 10, 100]
    return [
        {
            "logreg__solver": ["saga"],
            "logreg__C": strengths,
            "logreg__l1_ratio": [0, 0.25, 0.5, 0.75, 1],
        },
        {"logreg__solver": ["saga"], "logreg__C": [np.inf], "logreg__l1_ratio": [0]},
    ]
