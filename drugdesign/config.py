"""Typed configuration for the end-to-end workflow, loadable from TOML or JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

from .validation import evaluation_options, finite_number, positive_int

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None


@dataclass
class DataConfig:
    path: str = "chembl_data.csv"
    smiles_col: str = "smiles"
    target_col: str = "pIC50"
    value_col: Optional[str] = None  # e.g. "standard_value" (nM) to derive pIC50
    unit: str = "nM"
    unit_col: Optional[str] = None
    relation_col: Optional[str] = None
    standardize: bool = True
    aggregate: str = "median"


@dataclass
class ModelConfig:
    featurizer: Any = "morgan"  # str, {"name": ..., **params} or a list of those
    estimator: str = "random_forest"
    params: Dict[str, Any] = field(default_factory=dict)
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    save_path: Optional[str] = "outputs/model.joblib"
    split_strategy: str = "random"
    report_path: Optional[str] = "outputs/training_report.json"


@dataclass
class PredictConfig:
    smiles: List[str] = field(default_factory=list)
    input_path: Optional[str] = None  # .csv/.tsv/.smi with candidates
    smiles_col: str = "smiles"
    druglikeness: bool = True
    batch_size: int = 1024


@dataclass
class DockingConfig:
    enabled: bool = False
    receptor: Optional[str] = None  # .pdb (prepared automatically) or .pdbqt
    center: Optional[List[float]] = None
    size: List[float] = field(default_factory=lambda: [20.0, 20.0, 20.0])
    reference_ligand: Optional[str] = None  # derive the box from a bound ligand instead
    exhaustiveness: int = 8
    num_modes: int = 9
    seed: int = 42
    output_dir: str = "outputs/docking"
    top_n: Optional[int] = None  # dock only the N best-predicted candidates
    cpu: int = 0
    timeout: Optional[float] = None  # CLI Vina timeout in seconds
    ranking: str = "auto"  # auto, higher, lower


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    docking: DockingConfig = field(default_factory=DockingConfig)
    output: str = "outputs/predictions.csv"
    plugins: List[str] = field(default_factory=list)  # extra modules/files to import

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PipelineConfig":
        cfg = _from_dict(cls, raw)
        cfg.validate()
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        # Also validate instances constructed directly or mutated after loading.
        _from_dict(type(self), self.to_dict())
        evaluation_options(self.model.test_size, self.model.cv_folds, self.model.split_strategy)
        if self.data.aggregate not in {"mean", "median", "max", "min", "first"}:
            raise ValueError("data.aggregate must be mean, median, max, min or first")
        if self.data.target_col == "smiles":
            raise ValueError("data.target_col cannot be 'smiles'")
        positive_int(self.model.random_state, "model.random_state", minimum=0)
        positive_int(self.predict.batch_size, "predict.batch_size")
        if self.model.random_state > 2**32 - 1:
            raise ValueError("model.random_state must fit an unsigned 32-bit integer")
        d = self.docking
        positive_int(d.exhaustiveness, "docking.exhaustiveness")
        positive_int(d.num_modes, "docking.num_modes")
        positive_int(d.cpu, "docking.cpu", minimum=0)
        positive_int(d.seed, "docking.seed", minimum=0)
        if d.seed > 2**31 - 1:
            raise ValueError("docking.seed must fit a signed 32-bit integer")
        if d.top_n is not None:
            positive_int(d.top_n, "docking.top_n")
        if d.timeout is not None:
            finite_number(d.timeout, "docking.timeout", minimum=0.001)
        if d.ranking not in {"auto", "higher", "lower"}:
            raise ValueError("docking.ranking must be auto, higher or lower")
        if len(d.size) != 3 or any(v <= 0 for v in d.size):
            raise ValueError("docking.size must contain three positive numbers")
        for v in d.size:
            finite_number(v, "docking.size")
        if d.center is not None:
            if len(d.center) != 3:
                raise ValueError("docking.center must contain three numbers")
            for v in d.center:
                finite_number(v, "docking.center")
        if d.enabled and (not d.receptor or (d.center is None and not d.reference_ligand)):
            raise ValueError("Enabled docking requires receptor and center or reference_ligand")
        paths = [self.model.save_path, self.model.report_path, self.output]
        resolved = [str(Path(p).resolve()).casefold() for p in paths if p]
        if len(set(resolved)) != len(resolved):
            raise ValueError("Model, report and prediction output paths must be different")
        inputs = [self.data.path, self.predict.input_path, d.receptor, d.reference_ligand]
        inputs += [p for p in self.plugins if p.endswith(".py")]
        if set(resolved) & {str(Path(p).resolve()).casefold() for p in inputs if p}:
            raise ValueError("Output paths must not overwrite input files")


def _from_dict(cls, raw: Dict[str, Any]):
    if not isinstance(raw, dict):
        raise ValueError(f"{cls.__name__} must be a table/object")
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    hints = get_type_hints(cls)
    for name, value in raw.items():
        default = known[name].default_factory() if callable(known[name].default_factory) else None
        if is_dataclass(default) and isinstance(value, dict):
            value = _from_dict(type(default), value)
        if not _matches_type(value, hints[name]):
            raise ValueError(f"Invalid type for {cls.__name__}.{name}: expected {hints[name]}")
        kwargs[name] = value
    return cls(**kwargs)


def _matches_type(value, annotation):
    if annotation is Any:
        return True
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Union:
        return any(_matches_type(value, a) for a in args)
    if origin is list:
        return isinstance(value, list) and all(_matches_type(v, args[0]) for v in value)
    if origin is dict:
        return isinstance(value, dict) and all(
            _matches_type(k, args[0]) and _matches_type(v, args[1]) for k, v in value.items()
        )
    if annotation is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if annotation is int:
        return type(value) is int
    return isinstance(value, annotation)


def load_config(path: Union[str, Path]) -> PipelineConfig:
    """Load a :class:`PipelineConfig` from a ``.toml`` or ``.json`` file.

    Relative paths inside the file are resolved against the file's directory.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".toml":
        if tomllib is None:
            raise RuntimeError("Reading TOML on Python < 3.11 requires 'pip install tomli'")
        raw = tomllib.loads(text)
    elif path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix} (use .toml or .json)")
    cfg = PipelineConfig.from_dict(raw)
    _resolve_paths(cfg, path.parent)
    cfg.validate()
    return cfg


def _resolve_paths(cfg: PipelineConfig, base: Path) -> None:
    def resolve(p: Optional[str]) -> Optional[str]:
        if not p or Path(p).is_absolute():
            return p
        return str((base / p).resolve())

    cfg.data.path = resolve(cfg.data.path)
    cfg.model.save_path = resolve(cfg.model.save_path)
    cfg.model.report_path = resolve(cfg.model.report_path)
    cfg.predict.input_path = resolve(cfg.predict.input_path)
    cfg.docking.receptor = resolve(cfg.docking.receptor)
    cfg.docking.reference_ligand = resolve(cfg.docking.reference_ligand)
    cfg.docking.output_dir = resolve(cfg.docking.output_dir)
    cfg.output = resolve(cfg.output)
    cfg.plugins = [resolve(p) if p.endswith(".py") else p for p in cfg.plugins]
