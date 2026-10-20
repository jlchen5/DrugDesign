"""End-to-end workflow: data -> QSAR model -> candidate prediction -> (optional) docking."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import PipelineConfig
from .data import clean_dataset, load_activity_data, read_smiles_file
from .docking import DockingBox, DockingError, dock_smiles, prepare_receptor
from .filters import druglikeness_table
from .models import QSARModel, TrainingReport

logger = logging.getLogger(__name__)


def load_plugins(plugins: List[str]) -> None:
    """Import plugin modules (dotted names or ``.py`` paths) so their registrations run."""
    for plugin in plugins:
        if plugin.endswith(".py"):
            path = Path(plugin)
            if not path.exists():
                raise FileNotFoundError(f"Plugin file not found: {path}")
            # A stable module name lets pickled models that use plugin classes be reloaded.
            name = f"drugdesign_plugin_{path.stem}"
            if name in sys.modules:
                continue
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        else:
            importlib.import_module(plugin)
        logger.info("Loaded plugin %s", plugin)


def train_model(cfg: PipelineConfig) -> QSARModel:
    raw = load_activity_data(cfg.data.path, smiles_col=cfg.data.smiles_col,
                             target_col=cfg.data.target_col, value_col=cfg.data.value_col,
                             unit=cfg.data.unit)
    data = clean_dataset(raw, target_col=cfg.data.target_col,
                         standardize=cfg.data.standardize, aggregate=cfg.data.aggregate)
    model = QSARModel(featurizer=cfg.model.featurizer, estimator=cfg.model.estimator,
                      estimator_params=cfg.model.params, random_state=cfg.model.random_state,
                      target_name=cfg.data.target_col)
    model.train_and_evaluate(data["smiles"], data[cfg.data.target_col],
                             test_size=cfg.model.test_size, cv_folds=cfg.model.cv_folds)
    if cfg.model.save_path:
        model.save(cfg.model.save_path)
        logger.info("Model saved to %s", cfg.model.save_path)
    return model


def collect_candidates(cfg: PipelineConfig) -> pd.DataFrame:
    frames = []
    if cfg.predict.input_path:
        frames.append(read_smiles_file(cfg.predict.input_path, cfg.predict.smiles_col))
    if cfg.predict.smiles:
        frames.append(pd.DataFrame({"smiles": cfg.predict.smiles}))
    if not frames:
        return pd.DataFrame(columns=["smiles"])
    return pd.concat(frames, ignore_index=True)


def predict_candidates(model: QSARModel, candidates: pd.DataFrame,
                       druglikeness: bool = True) -> pd.DataFrame:
    smiles = candidates["smiles"].astype(str).tolist()
    result = model.predict_frame(smiles)
    extra = candidates.drop(columns=["smiles"]).reset_index(drop=True)
    result = pd.concat([extra, result], axis=1)
    if druglikeness:
        dl = druglikeness_table(smiles).drop(columns=["smiles"])
        result = pd.concat([result, dl], axis=1)
    return result


def _safe_name(text: str, index: int) -> str:
    return f"cand{index:04d}_" + re.sub(r"[^A-Za-z0-9_-]+", "", str(text))[:24]


def dock_candidates(results: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    dcfg = cfg.docking
    if not dcfg.receptor:
        raise DockingError("docking.receptor must be set when docking is enabled")
    if dcfg.reference_ligand:
        box = DockingBox.from_ligand(dcfg.reference_ligand)
    elif dcfg.center:
        box = DockingBox(tuple(dcfg.center), tuple(dcfg.size))
    else:
        raise DockingError("Set docking.center or docking.reference_ligand to define the box")

    receptor = Path(dcfg.receptor)
    if receptor.suffix.lower() != ".pdbqt":
        receptor = prepare_receptor(receptor, Path(dcfg.output_dir) / f"{receptor.stem}.pdbqt")

    pred_col = next(c for c in results.columns if c.startswith("predicted_"))
    order = results[results["valid"]].sort_values(
        pred_col, ascending=_lower_is_better(pred_col))
    if dcfg.top_n:
        order = order.head(dcfg.top_n)

    results = results.copy()
    results["docking_affinity"] = float("nan")
    for idx, row in order.iterrows():
        label = row["name"] if "name" in results.columns and pd.notna(row.get("name")) \
            else row["smiles"]
        try:
            res = dock_smiles(row["smiles"], receptor, box, dcfg.output_dir,
                              name=_safe_name(label, idx), exhaustiveness=dcfg.exhaustiveness,
                              num_modes=dcfg.num_modes, seed=dcfg.seed)
            results.loc[idx, "docking_affinity"] = res.best_affinity
            logger.info("Docked %s: %.2f kcal/mol", row["smiles"], res.best_affinity)
        except DockingError as exc:
            logger.error("Docking failed for %s: %s", row["smiles"], exc)
    return results


def _lower_is_better(pred_col: str) -> bool:
    """p-activities are 'higher is better'; raw IC50-like targets are 'lower is better'."""
    name = pred_col.removeprefix("predicted_").lower()
    return not name.startswith("p") and any(k in name for k in ("ic50", "ki", "kd", "ec50"))


@dataclass
class PipelineResult:
    model: QSARModel
    report: Optional[TrainingReport]
    predictions: pd.DataFrame


def run_pipeline(cfg: PipelineConfig) -> PipelineResult:
    load_plugins(cfg.plugins)
    model = train_model(cfg)

    candidates = collect_candidates(cfg)
    predictions = pd.DataFrame()
    if not candidates.empty:
        predictions = predict_candidates(model, candidates, cfg.predict.druglikeness)
        if cfg.docking.enabled:
            predictions = dock_candidates(predictions, cfg)
        if cfg.output:
            out = Path(cfg.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(out, index=False)
            logger.info("Predictions written to %s", out)
    else:
        logger.info("No candidate molecules given; skipping prediction")
    return PipelineResult(model=model, report=model.report, predictions=predictions)
