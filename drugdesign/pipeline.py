"""End-to-end workflow: data -> QSAR model -> candidate prediction -> (optional) docking."""

from __future__ import annotations

import hashlib
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
from .features import FEATURIZERS
from .filters import druglikeness_table
from .models import MODELS, QSARModel, TrainingReport
from .validation import positive_int

logger = logging.getLogger(__name__)


def load_plugins(plugins: List[str]) -> None:
    """Import plugin modules (dotted names or ``.py`` paths) so their registrations run."""
    for plugin in plugins:
        if plugin.endswith(".py"):
            path = Path(plugin).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Plugin file not found: {path}")
            # A stable module name lets pickled models that use plugin classes be reloaded.
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            name = f"drugdesign_plugin_{path.stem}_{digest}"
            if name in sys.modules:
                loaded_path = Path(sys.modules[name].__file__).resolve()
                if loaded_path != path:
                    raise ValueError(
                        f"Identical plugin loaded from two paths: {loaded_path}, {path}"
                    )
                continue
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ValueError(f"Cannot load plugin: {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            snapshots = [dict(reg._items) for reg in (FEATURIZERS, MODELS)]
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(name, None)
                for reg, snapshot in zip((FEATURIZERS, MODELS), snapshots):
                    reg._items = snapshot
                raise
        else:
            importlib.import_module(plugin)
        logger.info("Loaded plugin %s", plugin)


def train_model(cfg: PipelineConfig) -> QSARModel:
    cfg.validate()
    load_plugins(cfg.plugins)
    raw = load_activity_data(
        cfg.data.path,
        smiles_col=cfg.data.smiles_col,
        target_col=cfg.data.target_col,
        value_col=cfg.data.value_col,
        unit=cfg.data.unit,
        unit_col=cfg.data.unit_col,
        relation_col=cfg.data.relation_col,
    )
    model = QSARModel(
        featurizer=cfg.model.featurizer,
        estimator=cfg.model.estimator,
        estimator_params=cfg.model.params,
        random_state=cfg.model.random_state,
        target_name=cfg.data.target_col,
        standardize=cfg.data.standardize,
    )
    data = clean_dataset(
        raw,
        target_col=cfg.data.target_col,
        standardize=cfg.data.standardize,
        aggregate=cfg.data.aggregate,
        classification=model.is_classifier,
    )
    model.train_and_evaluate(
        data["smiles"],
        data[cfg.data.target_col],
        test_size=cfg.model.test_size,
        cv_folds=cfg.model.cv_folds,
        split_strategy=cfg.model.split_strategy,
    )
    model.report.metadata["config"] = cfg.to_dict()
    model.report.metadata["data_rows"] = {"loaded": len(raw), "cleaned": len(data)}
    model.report.metadata["data_sha256"] = hashlib.sha256(
        Path(cfg.data.path).read_bytes()
    ).hexdigest()
    if cfg.model.save_path:
        model.save(cfg.model.save_path)
        logger.info("Model saved to %s", cfg.model.save_path)
    if cfg.model.report_path:
        model.report.save(cfg.model.report_path)
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


def predict_candidates(
    model: QSARModel, candidates: pd.DataFrame, druglikeness: bool = True, batch_size: int = 1024
) -> pd.DataFrame:
    positive_int(batch_size, "batch_size")
    if not candidates.columns.is_unique:
        raise ValueError("Candidate table has duplicate column names")
    smiles = candidates["smiles"].tolist()
    frames = [
        model.predict_frame(smiles[i : i + batch_size]) for i in range(0, len(smiles), batch_size)
    ]
    result = pd.concat(frames, ignore_index=True) if frames else model.predict_frame([])
    extra = candidates.drop(columns=["smiles"]).reset_index(drop=True)
    if druglikeness:
        dl = druglikeness_table(result["model_smiles"].tolist()).drop(columns=["smiles"])
        result = pd.concat([result, dl], axis=1)
    # Generated columns keep their meaning; preserve colliding input metadata explicitly.
    used = set(result.columns) | set(extra.columns)
    renames = {}
    for col in extra.columns:
        if col in result.columns or str(col).startswith(("predicted_", "probability_", "docking_")):
            new = f"input_{col}"
            while new in used:
                new = f"input_{new}"
            renames[col] = new
            used.add(new)
    result = pd.concat([extra.rename(columns=renames), result], axis=1)
    return result


def _safe_name(text: str, index: int) -> str:
    return f"cand{index:04d}_" + re.sub(r"[^A-Za-z0-9_-]+", "", str(text))[:24]


def dock_candidates(results: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    dcfg = cfg.docking
    if not dcfg.receptor:
        raise DockingError("docking.receptor must be set when docking is enabled")
    if dcfg.reference_ligand:
        box = DockingBox.from_ligand(dcfg.reference_ligand)
    elif dcfg.center is not None:
        box = DockingBox(tuple(dcfg.center), tuple(dcfg.size))
    else:
        raise DockingError("Set docking.center or docking.reference_ligand to define the box")

    receptor = Path(dcfg.receptor)
    if receptor.suffix.lower() != ".pdbqt":
        receptor = prepare_receptor(receptor, Path(dcfg.output_dir) / f"{receptor.stem}.pdbqt")

    pred_col = next(c for c in results.columns if c.startswith("predicted_"))
    rank_col = "probability_active" if "probability_active" in results else pred_col
    ascending = (
        dcfg.ranking == "lower"
        if dcfg.ranking != "auto"
        else (_lower_is_better(rank_col) if rank_col == pred_col else False)
    )
    order = results[results["valid"] & results[rank_col].notna()].sort_values(
        rank_col, ascending=ascending, kind="stable"
    )
    if dcfg.top_n is not None:
        order = order.head(dcfg.top_n)

    results = results.copy()
    results["docking_affinity"] = float("nan")
    results["docking_status"] = "not_selected"
    results["docking_error"] = ""
    for idx, row in order.iterrows():
        label = (
            row["name"]
            if "name" in results.columns and pd.notna(row.get("name"))
            else row["smiles"]
        )
        try:
            res = dock_smiles(
                row.get("model_smiles", row["smiles"]),
                receptor,
                box,
                dcfg.output_dir,
                name=_safe_name(label, idx),
                exhaustiveness=dcfg.exhaustiveness,
                num_modes=dcfg.num_modes,
                seed=dcfg.seed,
                cpu=dcfg.cpu,
                timeout=dcfg.timeout,
            )
            results.loc[idx, "docking_affinity"] = res.best_affinity
            results.loc[idx, "docking_status"] = "success"
            logger.info("Docked %s: %.2f kcal/mol", row["smiles"], res.best_affinity)
        except DockingError as exc:
            results.loc[idx, "docking_status"] = "failed"
            results.loc[idx, "docking_error"] = str(exc)
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
    cfg.validate()
    load_plugins(cfg.plugins)
    model = train_model(cfg)

    candidates = collect_candidates(cfg)
    predictions = pd.DataFrame()
    if not candidates.empty:
        predictions = predict_candidates(
            model, candidates, cfg.predict.druglikeness, cfg.predict.batch_size
        )
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
