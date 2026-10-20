"""Typed configuration for the end-to-end workflow, loadable from TOML or JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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


@dataclass
class PredictConfig:
    smiles: List[str] = field(default_factory=list)
    input_path: Optional[str] = None  # .csv/.tsv/.smi with candidates
    smiles_col: str = "smiles"
    druglikeness: bool = True


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
        return _from_dict(cls, raw)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _from_dict(cls, raw: Dict[str, Any]):
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in raw.items():
        default = known[name].default_factory() if callable(known[name].default_factory) \
            else None
        if is_dataclass(default) and isinstance(value, dict):
            value = _from_dict(type(default), value)
        kwargs[name] = value
    return cls(**kwargs)


def load_config(path: Union[str, Path]) -> PipelineConfig:
    """Load a :class:`PipelineConfig` from a ``.toml`` or ``.json`` file.

    Relative paths inside the file are resolved against the file's directory.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
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
    return cfg


def _resolve_paths(cfg: PipelineConfig, base: Path) -> None:
    def resolve(p: Optional[str]) -> Optional[str]:
        if p is None or Path(p).is_absolute():
            return p
        return str((base / p).resolve())

    cfg.data.path = resolve(cfg.data.path)
    cfg.model.save_path = resolve(cfg.model.save_path)
    cfg.predict.input_path = resolve(cfg.predict.input_path)
    cfg.docking.receptor = resolve(cfg.docking.receptor)
    cfg.docking.reference_ligand = resolve(cfg.docking.reference_ligand)
    cfg.docking.output_dir = resolve(cfg.docking.output_dir)
    cfg.output = resolve(cfg.output)
    cfg.plugins = [resolve(p) if p.endswith(".py") else p for p in cfg.plugins]
