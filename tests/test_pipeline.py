import json

import pandas as pd

from drugdesign.cli import main
from drugdesign.config import PipelineConfig, load_config
from drugdesign.pipeline import run_pipeline

from .conftest import EXAMPLES


def test_example_config_loads():
    cfg = load_config(EXAMPLES / "config.toml")
    assert cfg.model.estimator == "random_forest"
    assert cfg.data.path.endswith("demo_activity.csv")


def test_run_pipeline(demo_csv, tmp_path):
    cfg = PipelineConfig.from_dict({
        "data": {"path": str(demo_csv)},
        "model": {"featurizer": "basic_descriptors", "params": {"n_estimators": 50},
                  "cv_folds": 3, "save_path": str(tmp_path / "model.joblib")},
        "predict": {"smiles": ["CCOc1ccccc1NC(=O)C", "bad"]},
        "output": str(tmp_path / "pred.csv"),
    })
    result = run_pipeline(cfg)
    out = pd.read_csv(tmp_path / "pred.csv")
    assert len(out) == 2 and "qed" in out.columns
    assert (tmp_path / "model.joblib").exists()
    assert result.report.cv_metrics


def test_unknown_config_key():
    import pytest

    with pytest.raises(ValueError):
        PipelineConfig.from_dict({"model": {"bogus": 1}})


def test_cli_train_predict_and_plugin(demo_csv, tmp_path, capsys):
    model_path = tmp_path / "m.joblib"
    plugin = str(EXAMPLES / "plugins" / "custom_featurizer.py")
    assert main(["--plugin", plugin, "train", "--data", str(demo_csv), "--featurizer",
                 "atom_pair", "--model", "hist_gb", "--params", json.dumps({"max_iter": 50}),
                 "--cv", "3", "--out", str(model_path)]) == 0
    out_csv = tmp_path / "p.csv"
    assert main(["--plugin", plugin, "predict", "--model", str(model_path),
                 "--input", str(EXAMPLES / "data" / "candidates.smi"),
                 "--out", str(out_csv)]) == 0
    assert len(pd.read_csv(out_csv)) == 4


def test_cli_errors_return_nonzero(tmp_path):
    assert main(["train", "--data", str(tmp_path / "missing.csv")]) == 1


def test_cli_init_config(tmp_path):
    path = tmp_path / "cfg.json"
    assert main(["init-config", "--out", str(path)]) == 0
    cfg = load_config(path)
    assert cfg.model.estimator == "random_forest"
