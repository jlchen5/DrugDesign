"""QSAR / activity prediction models built on scikit-learn estimators."""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
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
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted

from . import __version__
from .data import mols_from_smiles, standardize_smiles
from .features import (
    Featurizer,
    FeaturizerSpec,
    MorganFingerprint,
    build_featurizer,
    tanimoto_max_similarity,
)
from .registry import Registry
from .splitting import molecule_groups
from .validation import evaluation_options

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


def build_estimator(
    name: str, params: Optional[Dict[str, Any]] = None, random_state: Optional[int] = 42
) -> BaseEstimator:
    if params is not None and not isinstance(params, dict):
        raise ValueError("Estimator params must be a dictionary")
    if params and "random_state" in params:
        raise ValueError("Set random_state on QSARModel (CLI: --seed), not in estimator_params")
    return MODELS.create(name, random_state=random_state, **(params or {}))


# --- Metrics -------------------------------------------------------------------


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pearson = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if len(y_true) > 1 and np.std(y_pred) > 0 and np.std(y_true) > 0
        else float("nan")
    )
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "pearson_r": pearson,
    }


def classification_metrics(y_true, y_pred, y_score=None) -> Dict[str, float]:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1_options = (
        {"average": "binary", "pos_label": labels[-1]} if len(labels) == 2 else {"average": "macro"}
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
                **f1_options,
            )
        ),
    }
    if y_score is not None and len(set(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def cross_val_outputs(estimator, X, y, cv):
    """Compute labels and scores in one fit per fold, without nested parallelism."""
    predictions = np.empty(len(y), dtype=float)
    binary = is_classifier(estimator) and len(np.unique(y)) == 2
    scores = np.full(len(y), np.nan) if binary else None
    for train, test in cv:
        fitted = clone(estimator).fit(X[train], y[train])
        predictions[test] = fitted.predict(X[test])
        if scores is not None:
            values = QSARModel._estimator_score(fitted, X[test])
            if values is None:
                scores = None
            else:
                scores[test] = values
    return predictions, scores


# --- The model wrapper -----------------------------------------------------------


@dataclass
class TrainingReport:
    n_train: int
    n_test: int
    test_metrics: Dict[str, float] = field(default_factory=dict)
    cv_metrics: Dict[str, float] = field(default_factory=dict)
    n_input: int = 0
    n_valid: int = 0
    n_rejected: int = 0
    n_final_fit: int = 0
    split_strategy: str = "random"
    cv_folds: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        def safe(value):
            if isinstance(value, dict):
                return {k: safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [safe(v) for v in value]
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                return None
            return value

        return safe(asdict(self))

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return path

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

    def __init__(
        self,
        featurizer: FeaturizerSpec = "morgan",
        estimator: Union[str, BaseEstimator] = "random_forest",
        estimator_params: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = 42,
        target_name: str = "pIC50",
        standardize: bool = True,
    ) -> None:
        if not isinstance(standardize, bool):
            raise ValueError("standardize must be a boolean")
        self.featurizer: Featurizer = build_featurizer(featurizer)
        if isinstance(estimator, str):
            self.estimator_name = estimator
            self.estimator = build_estimator(estimator, estimator_params, random_state)
        else:
            self.estimator_name = type(estimator).__name__
            self.estimator = estimator
        self.random_state = random_state
        self.target_name = target_name
        self.standardize = standardize
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
        if isinstance(smiles, str):
            smiles = [smiles]
        mols = mols_from_smiles(
            [standardize_smiles(s) for s in smiles] if self.standardize else smiles
        )
        X = self.featurizer(mols)
        if X.shape != (len(mols), self.featurizer.n_features):
            raise ValueError("Featurizer returned an invalid feature matrix shape")
        valid = np.array([m is not None for m in mols], dtype=bool) & np.isfinite(X).all(axis=1)
        return X, valid, mols

    def _training_data(self, smiles, y):
        smiles = [smiles] if isinstance(smiles, str) else list(smiles)
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or len(smiles) != len(y):
            raise ValueError("smiles and y must be one-dimensional with equal lengths")
        X, valid, mols = self.featurize(smiles)
        valid &= np.isfinite(y)
        if not valid.all():
            logger.warning(
                "Skipping %d rows with invalid molecules/features/targets", (~valid).sum()
            )
        if valid.sum() < 2:
            raise ValueError("Need at least two valid molecules with finite targets")
        yv = y[valid]
        if self.is_classifier:
            if not np.equal(yv, np.floor(yv)).all() or len(np.unique(yv)) < 2:
                raise ValueError("Classification requires at least two integer class labels")
        valid_mols = [m for m, ok in zip(mols, valid) if ok]
        if self.is_classifier:
            groups = molecule_groups(valid_mols)
            if (
                pd.DataFrame({"group": groups, "target": yv}).groupby("group").target.nunique() > 1
            ).any():
                raise ValueError("Duplicate structures have conflicting class labels")
        return X[valid], yv, valid_mols, len(smiles)

    def _fit_arrays(self, X, y, mols):
        estimator = clone(self.estimator)
        estimator.fit(X, y)
        fps = [self._ad_featurizer.bit_vector(m) for m in mols]
        self.estimator = estimator
        self._train_fps = fps
        self.report = None

    # -- training --
    def fit(self, smiles: Sequence[str], y: Sequence[float]) -> "QSARModel":
        X, y, mols, _ = self._training_data(smiles, y)
        self._fit_arrays(X, y, mols)
        return self

    def evaluate(self, smiles: Sequence[str], y: Sequence[float]) -> Dict[str, float]:
        check_is_fitted(self.estimator)
        smiles = [smiles] if isinstance(smiles, str) else list(smiles)
        X, valid, _ = self.featurize(smiles)
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or len(y) != len(smiles):
            raise ValueError("smiles and y must be one-dimensional with equal lengths")
        valid &= np.isfinite(y)
        if not valid.any():
            raise ValueError("No valid evaluation rows")
        y = y[valid]
        pred = self.estimator.predict(X[valid])
        if self.is_classifier:
            score = self._positive_score(X[valid])
            return classification_metrics(y, pred, score)
        return regression_metrics(y, pred)

    def train_and_evaluate(
        self,
        smiles: Sequence[str],
        y: Sequence[float],
        test_size: float = 0.2,
        cv_folds: int = 5,
        split_strategy: str = "random",
    ) -> TrainingReport:
        """Clean first, evaluate without structure overlap, then refit all valid rows.

        CV uses only the hold-out training partition. Feature extraction is stateless
        and performed once; learned scaling stays inside each estimator pipeline.
        Requested evaluations fail clearly if the dataset cannot support them.
        """
        evaluation_options(test_size, cv_folds, split_strategy)
        X, y, mols, n_input = self._training_data(smiles, y)
        groups = molecule_groups(mols, split_strategy)
        indices = np.arange(len(y))
        train, test = indices, np.array([], dtype=int)
        if test_size:
            if split_strategy == "scaffold":
                train, test = next(
                    GroupShuffleSplit(
                        n_splits=1, test_size=test_size, random_state=self.random_state
                    ).split(X, y, groups)
                )
            else:
                unique, first = np.unique(groups, return_index=True)
                tr_groups, te_groups = train_test_split(
                    unique,
                    test_size=test_size,
                    random_state=self.random_state,
                    stratify=y[first] if self.is_classifier else None,
                )
                train = indices[np.isin(groups, tr_groups)]
                test = indices[np.isin(groups, te_groups)]
        if len(train) < 2:
            raise ValueError("Split leaves fewer than two training molecules")
        self._validate_classes(y, y[train])
        Xtr, ytr, gtr = X[train], y[train], groups[train]
        cv_metrics = {}
        if cv_folds:
            if len(np.unique(gtr)) < cv_folds:
                raise ValueError(
                    "Not enough training structure groups for cv_folds; reduce CV or test_size"
                )
            grouped = split_strategy == "scaffold" or len(np.unique(gtr)) < len(gtr)
            if grouped:
                splitter = (
                    StratifiedGroupKFold(
                        n_splits=cv_folds, shuffle=True, random_state=self.random_state
                    )
                    if self.is_classifier
                    else GroupKFold(cv_folds)
                )
                splits = list(splitter.split(Xtr, ytr, gtr))
            else:
                if self.is_classifier and np.unique(ytr, return_counts=True)[1].min() < cv_folds:
                    raise ValueError("Each training class needs at least cv_folds samples")
                splitter = (StratifiedKFold if self.is_classifier else KFold)(
                    n_splits=cv_folds, shuffle=True, random_state=self.random_state
                )
                splits = list(splitter.split(Xtr, ytr))
            for fold_train, _ in splits:
                self._validate_classes(ytr, ytr[fold_train])
            pred, score = cross_val_outputs(self.estimator, Xtr, ytr, cv=splits)
            cv_metrics = (
                classification_metrics(ytr, pred, score)
                if self.is_classifier
                else regression_metrics(ytr, pred)
            )
        test_metrics = {}
        if len(test):
            heldout = clone(self.estimator).fit(Xtr, ytr)
            pred = heldout.predict(X[test])
            score = self._estimator_score(heldout, X[test]) if self.is_classifier else None
            test_metrics = (
                classification_metrics(y[test], pred, score)
                if self.is_classifier
                else regression_metrics(y[test], pred)
            )
        self._fit_arrays(X, y, mols)
        self.report = TrainingReport(
            n_train=len(train),
            n_test=len(test),
            test_metrics=test_metrics,
            cv_metrics=cv_metrics,
            n_input=n_input,
            n_valid=len(y),
            n_rejected=n_input - len(y),
            n_final_fit=len(y),
            split_strategy=split_strategy,
            cv_folds=cv_folds,
            metadata={
                "framework_version": __version__,
                "python": platform.python_version(),
                "dependencies": {
                    n: version(n) for n in ("rdkit", "scikit-learn", "numpy", "pandas")
                },
                "estimator": self.estimator_name,
                "estimator_repr": repr(self.estimator),
                "featurizer": repr(self.featurizer),
                "target": self.target_name,
                "standardize": self.standardize,
                "random_state": self.random_state,
                "test_size": test_size,
                "cv_scope": "holdout_training_partition",
                "n_groups": len(np.unique(groups)),
            },
        )
        logger.info("Training report: %s", self.report.summary())
        return self.report

    def _validate_classes(self, all_y, train_y):
        if self.is_classifier and set(train_y) != set(all_y):
            raise ValueError(
                "A training split is missing classes; change seed/split/CV or add data"
            )

    # -- inference --
    def _positive_score(self, X: np.ndarray) -> Optional[np.ndarray]:
        return self._estimator_score(self.estimator, X)

    @staticmethod
    def _estimator_score(estimator, X):
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X)
            return proba[:, -1] if proba.shape[1] == 2 else None
        if hasattr(estimator, "decision_function") and len(estimator.classes_) == 2:
            return estimator.decision_function(X)
        return None

    def predict(self, smiles: Sequence[str]) -> np.ndarray:
        """Predict for each SMILES; invalid structures give NaN."""
        check_is_fitted(self.estimator)
        if isinstance(smiles, str):
            smiles = [smiles]
        smiles = list(smiles)
        X, valid, _ = self.featurize(smiles)
        out = np.full(len(smiles), np.nan)
        if valid.any():
            out[valid] = self.estimator.predict(X[valid])
        return out

    def predict_frame(self, smiles: Sequence[str]) -> pd.DataFrame:
        """Predictions plus validity flag and applicability-domain similarity."""
        check_is_fitted(self.estimator)
        if isinstance(smiles, str):
            smiles = [smiles]
        smiles = list(smiles)
        X, valid, mols = self.featurize(smiles)
        pred = np.full(len(smiles), np.nan)
        if valid.any():
            pred[valid] = self.estimator.predict(X[valid])
        normalized = [Chem.MolToSmiles(m) if m is not None else None for m in mols]
        frame = pd.DataFrame(
            {
                "smiles": pd.Series(smiles, dtype=object),
                "model_smiles": pd.Series(normalized, dtype="string"),
                "valid": valid,
                f"predicted_{self.target_name}": pred,
            }
        )
        if self.is_classifier and hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(X[valid]) if valid.any() else None
            for j, label in enumerate(self.estimator.classes_):
                col = np.full(len(smiles), np.nan)
                if proba is not None:
                    col[valid] = proba[:, j]
                frame[f"probability_class_{label:g}"] = col
            if np.array_equal(self.estimator.classes_, [0, 1]):
                frame["probability_active"] = frame["probability_class_1"]
        fps = [self._ad_featurizer.bit_vector(m) if ok else None for m, ok in zip(mols, valid)]
        frame["max_train_similarity"] = tanimoto_max_similarity(fps, self._train_fps)
        return frame

    # -- persistence --
    def save(self, path: Union[str, Path]) -> Path:
        check_is_fitted(self.estimator)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.__dict__.copy()
        # ExplicitBitVects pickle fine but are bulky; store as bit strings instead.
        state["_train_fps"] = [fp.ToBitString() for fp in self._train_fps]
        joblib.dump(state, path)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "QSARModel":
        """Load a trusted local joblib artifact; pickle can execute Python code."""
        from rdkit import DataStructs

        state = joblib.load(path)
        if (
            not isinstance(state, dict)
            or not {"estimator", "featurizer", "_train_fps"} <= state.keys()
        ):
            raise ValueError("Not a DrugDesign model artifact")
        state.setdefault("standardize", False)  # v0.2 models used raw inference SMILES
        if state.get("version") != __version__:
            logger.warning(
                "Loading model version %s in %s; verify dependency compatibility",
                state.get("version"),
                __version__,
            )
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        obj._train_fps = [DataStructs.CreateFromBitString(s) for s in state["_train_fps"]]
        return obj

    def __repr__(self) -> str:
        return (
            f"QSARModel(featurizer={self.featurizer!r}, "
            f"estimator={self.estimator_name!r}, target={self.target_name!r})"
        )
