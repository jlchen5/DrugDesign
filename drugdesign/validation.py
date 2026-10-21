"""Small, shared validators for public APIs and configuration files."""

import math
from numbers import Integral, Real


def positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def finite_number(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def evaluation_options(test_size, cv_folds, split_strategy):
    finite_number(test_size, "test_size", minimum=0)
    if test_size >= 1:
        raise ValueError("test_size must be in [0, 1); 0 disables hold-out evaluation")
    positive_int(cv_folds, "cv_folds", minimum=0)
    if cv_folds == 1:
        raise ValueError("cv_folds must be 0 (disabled) or >= 2")
    if split_strategy not in {"random", "scaffold"}:
        raise ValueError("split_strategy must be 'random' or 'scaffold'")
