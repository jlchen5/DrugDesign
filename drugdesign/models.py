"""QSAR / activity prediction models built on scikit-learn estimators."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from . import __version__
from .data import mols_from_smiles
from .features import Featurizer, FeaturizerSpec, MorganFingerprint, build_featurizer, \
    tanimoto_max_similarity
from .registry import Registry

logger = logging.getLogger(__name__)

MODELS: Registry[BaseEstimator] = Registry("model")


# --- Built-in estimators -------------------------------------------------------
# Factories accept ``random_state`` (ignored when meaningless) plus estimator params.

@MODELS.register("random_forest")
def _random_forest(random_state: Optional[int] = None, **params) -> BaseEstimator:
    params.setdefault("n_estimators", 500)
    params.setdefault("n_jobs", -1)
    return RandomForestRegressor(random_state=random_state, **params)


@MODELS.register("extra_trees")
def _extra_trees(random_state: Optional[int] = None, **params) -> BaseEstimator:
    params.setdefault("n_estimators", 500)
    params.setdefault("n_jobs", -1)
    return ExtraTreesRegressor(random_state=random_state, **params)


@MODELS.register("gradient_boosting")
def _gradient_boosting(random_state: Optional[int] = None, **params) -> BaseEstimator:
    return GradientBoostingRegressor(random_state=random_state, **params)


@MODELS.register("svr")
def _svr(random_state: Optional[int] = None, **params) -> BaseEstimator:
    return make_pipeline(StandardScaler(), SVR(**params))


@MODELS.register("ridge")
def _ridge(random_state: Optional[int] = None, **params) -> BaseEstimator:
    return make_pipeline(StandardScaler(), Ridge(**params))


@MODELS.register("knn")
def _knn(random_state: Optional[int] = None, **params) -> BaseEstimator:
    params.setdefault("n_neighbors", 5)
    return make_pipeline(StandardScaler(), KNeighborsRegressor(**params))


@MODELS.register("mlp")
def _mlp(random_state: Optional[int] = None, **params) -> BaseEstimator:
    params.setdefault("hidden_layer_sizes", (256, 64))
    params.setdefault("max_iter", 1000)
    params.setdefault("early_stopping", True)
    if isinstance(params["hidden_layer_sizes"], list):
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    return make_pipeline(StandardScaler(), MLPRegressor(random_state=random_state, **params))


@MODELS.register("random_forest_classifier")
def _random_forest_classifier(random_state: Optional[int] = None, **params) -> BaseEstimator:
    params.setdefault("n_estimators", 500)
    params.setdefault("n_jobs", -1)
    params.setdefault("class_weight", "balanced")
    return RandomForestClassifier(random_state=random_state, **params)


@MODELS.register("logistic_regression")
def _logistic_regression(random_state: Optional[int] = None, **params) -> BaseEstimator:
    params.setdefault("max_iter", 2000)
    return make_pipeline(StandardScaler(), LogisticRegression(random_state=random_state, **params))


def build_estimator(name: str, params: Optional[Dict[str, Any]] = None,
                    random_state: Optional[int] = 42) -> BaseEstimator:
    return MODELS.create(name, random_state=random_state, **(params or {}))


# --- Metrics -------------------------------------------------------------------

def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 and np.std(y_pred) > 0 \
        else float("nan")
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "pearson_r": pearson,
    }


def classification_metrics(y_true, y_pred, y_score=None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary" if len(set(y_true)) <= 2
                             else "macro")),
    }
    if y_score is not None and len(set(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


# --- The model wrapper -----------------------------------------------------------

@dataclass
class TrainingReport:
    n_train: int
    n_test: int
    test_metrics: Dict[str, float] = field(default_factory=dict)
    cv_metrics: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        def fmt(d):
            return ", ".join(f"{k}={v:.3f}" for k, v in d.items())
        lines = [f"train={self.n_train}, test={self.n_test}"]
        if self.test_metrics:
            lines.append(f"hold-out: {fmt(self.test_metrics)}")
        if self.cv_metrics:
            lines.append(f"cross-val: {fmt(self.cv_metrics)}")
        return " | ".join(lines)


class QSARModel:
    """A featurizer + scikit-learn estimator that works directly on SMILES.

    >>> model = QSARModel("morgan", "random_forest").fit(smiles, y)
    >>> model.predict(["CCO"])
    """

    def __init__(self, featurizer: FeaturizerSpec = "morgan",
                 estimator: Union[str, BaseEstimator] = "random_forest",
                 estimator_params: Optional[Dict[str, Any]] = None,
                 random_state: Optional[int] = 42,
                 target_name: str = "pIC50") -> None:
        self.featurizer: Featurizer = build_featurizer(featurizer)
        if isinstance(estimator, str):
            self.estimator_name = estimator
            self.estimator = build_estimator(estimator, estimator_params, random_state)
        else:
            self.estimator_name = type(estimator).__name__
            self.estimator = estimator
        self.random_state = random_state
        self.target_name = target_name
        self.report: Optional[TrainingReport] = None
        self.version = __version__
        self._train_fps: List = []
        self._ad_featurizer = MorganFingerprint(radius=2, n_bits=2048)

    @property
    def is_classifier(self) -> bool:
        return is_classifier(self.estimator)

    # -- featurisation --
    def featurize(self, smiles: Sequence[str]):
        """Return (X, valid_mask) for a list of SMILES."""
        mols = mols_from_smiles(smiles)
        X = self.featurizer(mols)
        valid = np.array([m is not None for m in mols]) & np.isfinite(X).all(axis=1)
        return X, valid, mols

    # -- training --
    def fit(self, smiles: Sequence[str], y: Sequence[float]) -> "QSARModel":
        X, valid, mols = self.featurize(smiles)
        y = np.asarray(y)
        if not valid.all():
            logger.warning("Skipping %d invalid molecules during fit", int((~valid).sum()))
        if valid.sum() < 2:
            raise ValueError("Need at least two valid molecules to train a model")
        self.estimator.fit(X[valid], y[valid])
        self._train_fps = [self._ad_featurizer.bit_vector(m)
                           for m, ok in zip(mols, valid) if ok]
        return self

    def evaluate(self, smiles: Sequence[str], y: Sequence[float]) -> Dict[str, float]:
        X, valid, _ = self.featurize(smiles)
        y = np.asarray(y)[valid]
        pred = self.estimator.predict(X[valid])
        if self.is_classifier:
            score = self._positive_score(X[valid])
            return classification_metrics(y, pred, score)
        return regression_metrics(y, pred)

    def train_and_evaluate(self, smiles: Sequence[str], y: Sequence[float],
                           test_size: float = 0.2, cv_folds: int = 5) -> TrainingReport:
        """Hold-out evaluation + k-fold CV, then refit on *all* data."""
        smiles = list(smiles)
        y = np.asarray(y)
        stratify = y if self.is_classifier else None
        cv_metrics: Dict[str, float] = {}

        if cv_folds and cv_folds > 1 and len(smiles) >= cv_folds * 2:
            X, valid, _ = self.featurize(smiles)
            Xv, yv = X[valid], y[valid]
            splitter = (StratifiedKFold if self.is_classifier else KFold)(
                n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            pred = cross_val_predict(clone(self.estimator), Xv, yv, cv=splitter)
            cv_metrics = (classification_metrics(yv, pred) if self.is_classifier
                          else regression_metrics(yv, pred))

        test_metrics: Dict[str, float] = {}
        n_test = 0
        if test_size and 0 < test_size < 1 and len(smiles) >= 5:
            s_tr, s_te, y_tr, y_te = train_test_split(
                smiles, y, test_size=test_size, random_state=self.random_state, stratify=stratify)
            self.fit(s_tr, y_tr)
            test_metrics = self.evaluate(s_te, y_te)
            n_test = len(s_te)

        self.fit(smiles, y)  # final model uses every compound
        self.report = TrainingReport(n_train=len(smiles) - n_test, n_test=n_test,
                                     test_metrics=test_metrics, cv_metrics=cv_metrics)
        logger.info("Training report: %s", self.report.summary())
        return self.report

    # -- inference --
    def _positive_score(self, X: np.ndarray) -> Optional[np.ndarray]:
        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(X)
            return proba[:, -1] if proba.shape[1] == 2 else None
        return None

    def predict(self, smiles: Sequence[str]) -> np.ndarray:
        """Predict for each SMILES; invalid structures give NaN."""
        if isinstance(smiles, str):
            smiles = [smiles]
        X, valid, _ = self.featurize(smiles)
        out = np.full(len(smiles), np.nan)
        if valid.any():
            out[valid] = self.estimator.predict(X[valid])
        return out

    def predict_frame(self, smiles: Sequence[str]) -> pd.DataFrame:
        """Predictions plus validity flag and applicability-domain similarity."""
        if isinstance(smiles, str):
            smiles = [smiles]
        smiles = list(smiles)
        X, valid, mols = self.featurize(smiles)
        pred = np.full(len(smiles), np.nan)
        if valid.any():
            pred[valid] = self.estimator.predict(X[valid])
        frame = pd.DataFrame({"smiles": smiles, "valid": valid,
                              f"predicted_{self.target_name}": pred})
        if self.is_classifier and valid.any():
            score = self._positive_score(X[valid])
            if score is not None:
                col = np.full(len(smiles), np.nan)
                col[valid] = score
                frame["probability_active"] = col
        fps = [self._ad_featurizer.bit_vector(m) if ok else None for m, ok in zip(mols, valid)]
        frame["max_train_similarity"] = tanimoto_max_similarity(fps, self._train_fps)
        return frame

    # -- persistence --
    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.__dict__.copy()
        # ExplicitBitVects pickle fine but are bulky; store as bit strings instead.
        state["_train_fps"] = [fp.ToBitString() for fp in self._train_fps]
        joblib.dump(state, path)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "QSARModel":
        from rdkit import DataStructs

        state = joblib.load(path)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        obj._train_fps = [DataStructs.CreateFromBitString(s) for s in state["_train_fps"]]
        return obj

    def __repr__(self) -> str:
        return (f"QSARModel(featurizer={self.featurizer!r}, "
                f"estimator={self.estimator_name!r}, target={self.target_name!r})")
