"""Command-line interface: ``drugdesign <command> ...`` (or ``python -m drugdesign``)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from . import __version__
from .config import PipelineConfig, load_config
from .data import quiet_rdkit, read_smiles_file
from .docking import DockingBox, DockingError, dock_smiles, prepare_receptor
from .features import FEATURIZERS
from .models import MODELS, QSARModel
from .pipeline import load_plugins, predict_candidates, run_pipeline, train_model

logger = logging.getLogger("drugdesign")


def _featurizer_spec(values: List[str]):
    """``--featurizer morgan`` / ``--featurizer morgan maccs`` / ``--featurizer '{"name":...}'``."""
    specs = [json.loads(v) if v.strip().startswith("{") else v for v in values]
    return specs[0] if len(specs) == 1 else specs


def cmd_train(args: argparse.Namespace) -> int:
    params = json.loads(args.params) if args.params else {}
    cfg = PipelineConfig.from_dict(
        {
            "data": {
                "path": args.data,
                "smiles_col": args.smiles_col,
                "target_col": args.target_col,
                "value_col": args.value_col,
                "unit": args.unit,
                "unit_col": args.unit_col,
                "relation_col": args.relation_col,
                "standardize": not args.no_standardize,
                "aggregate": args.aggregate,
            },
            "model": {
                "featurizer": _featurizer_spec(args.featurizer),
                "estimator": args.model,
                "params": params,
                "random_state": args.seed,
                "test_size": args.test_size,
                "cv_folds": args.cv,
                "split_strategy": args.split,
                "save_path": args.out,
                "report_path": args.report or str(Path(args.out).with_suffix(".report.json")),
            },
        }
    )
    model = train_model(cfg)
    print(model.report.summary())
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    model = QSARModel.load(args.model)
    if args.input:
        candidates = read_smiles_file(args.input, args.smiles_col)
    else:
        candidates = pd.DataFrame({"smiles": args.smiles})
    results = predict_candidates(
        model, candidates, druglikeness=not args.no_druglikeness, batch_size=args.batch_size
    )
    if args.out:
        output_path = Path(args.out).resolve()
        inputs = [args.model, args.input]
        if output_path in {Path(p).resolve() for p in inputs if p}:
            raise ValueError("Prediction output must not overwrite the model or candidate input")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.out, index=False)
        print(f"Predictions written to {args.out}")
    else:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(results.to_string(index=False))
    return 0


def cmd_dock(args: argparse.Namespace) -> int:
    if args.reference_ligand:
        box = DockingBox.from_ligand(args.reference_ligand, padding=args.padding)
    elif args.center:
        box = DockingBox(tuple(args.center), tuple(args.size))
    else:
        raise SystemExit("Provide --center X Y Z or --reference-ligand to define the box")
    receptor = Path(args.receptor)
    if receptor.suffix.lower() != ".pdbqt":
        receptor = prepare_receptor(receptor, Path(args.out_dir) / f"{receptor.stem}.pdbqt")
    status = 0
    for i, smi in enumerate(args.smiles):
        try:
            res = dock_smiles(
                smi,
                receptor,
                box,
                args.out_dir,
                name=f"ligand{i:03d}",
                exhaustiveness=args.exhaustiveness,
                num_modes=args.num_modes,
                seed=args.seed,
                cpu=args.cpu,
                timeout=args.timeout,
            )
            print(f"{smi}\t{res.best_affinity:.2f} kcal/mol\t{res.output_path}")
        except DockingError as exc:
            print(f"{smi}\tFAILED: {exc}", file=sys.stderr)
            status = 1
    return status


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    result = run_pipeline(cfg)
    if not result.predictions.empty:
        cols = [
            c
            for c in result.predictions.columns
            if c
            in {
                "name",
                "smiles",
                "docking_affinity",
                "qed",
                "lipinski_pass",
                "max_train_similarity",
            }
            or c.startswith("predicted_")
        ]
        print(result.predictions[cols].to_string(index=False))
    return 0


def cmd_init_config(args: argparse.Namespace) -> int:
    path = Path(args.out)
    if path.exists() and not args.force:
        raise SystemExit(f"{path} exists (use --force to overwrite)")
    if path.suffix.lower() != ".json":
        raise ValueError("init-config writes JSON; use an output filename ending in .json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(PipelineConfig().to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote default config to {path}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    print("Featurizers:", ", ".join(FEATURIZERS.names()))
    print("Models:     ", ", ".join(MODELS.names()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drugdesign", description="QSAR prediction & docking toolkit"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Debug logging and RDKit warnings"
    )
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        help="Module or .py file registering extra featurizers/models",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="Train a QSAR model from an activity table")
    p.add_argument("--data", required=True, help="CSV/TSV with SMILES and activity")
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--target-col", default="pIC50")
    p.add_argument("--value-col", help="Raw concentration column used to derive the target")
    p.add_argument("--unit", default="nM")
    p.add_argument("--unit-col", help="Per-row concentration unit column")
    p.add_argument("--relation-col", help="Keep only exact '=' activity measurements")
    p.add_argument(
        "--aggregate", choices=["mean", "median", "min", "max", "first"], default="median"
    )
    p.add_argument(
        "--no-standardize",
        action="store_true",
        help="Keep salts and charges at training and inference",
    )
    p.add_argument("--featurizer", nargs="+", default=["morgan"])
    p.add_argument("--model", default="random_forest")
    p.add_argument("--params", help="Estimator params as JSON, e.g. '{\"n_estimators\": 300}'")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--split", choices=["random", "scaffold"], default="random")
    p.add_argument("--report", help="Training report JSON (default: beside model)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs/model.joblib")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="Predict activity for new molecules")
    p.add_argument("--model", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--smiles", nargs="+")
    src.add_argument("--input", help=".csv/.tsv/.smi file of candidates")
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--no-druglikeness", action="store_true")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--out")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("dock", help="Dock molecules with AutoDock Vina")
    p.add_argument("--receptor", required=True, help="Protein .pdb or .pdbqt")
    p.add_argument("--smiles", nargs="+", required=True)
    p.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"))
    p.add_argument("--size", nargs=3, type=float, default=[20.0, 20.0, 20.0])
    p.add_argument("--reference-ligand", help="Bound ligand file to derive the box from")
    p.add_argument("--padding", type=float, default=8.0)
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--num-modes", type=int, default=9)
    p.add_argument("--cpu", type=int, default=0)
    p.add_argument("--timeout", type=float, help="Vina CLI time limit in seconds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="outputs/docking")
    p.set_defaults(func=cmd_dock)

    p = sub.add_parser("run", help="Run the full workflow from a TOML/JSON config")
    p.add_argument("config")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("init-config", help="Write a default JSON config")
    p.add_argument("--out", default="drugdesign_config.json")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init_config)

    p = sub.add_parser("list", help="List registered featurizers and models")
    p.set_defaults(func=cmd_list)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    quiet_rdkit(not args.verbose)
    try:
        load_plugins(args.plugin)
        return args.func(args)
    except (OSError, KeyError, ValueError, TypeError, ImportError, DockingError) as exc:
        logger.error("%s", exc)
        if args.verbose:
            logger.debug("Command failed", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
