import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from drugdesign.cli import main
from drugdesign.config import PipelineConfig
from drugdesign.data import load_activity_data
from drugdesign.docking import DockingBox, DockingError, dock_smiles, run_vina
from drugdesign.features import FEATURIZERS
from drugdesign.pipeline import dock_candidates, load_plugins


def test_chembl_mixed_units_and_censored_values(tmp_path):
    p = tmp_path / "chembl.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC", "CCCC"],
            "standard_value": [100, 0.1, 5, 10],
            "standard_units": ["nM", "uM", "nM", "nM"],
            "standard_relation": ["=", "=", ">", None],
        }
    ).to_csv(p, index=False)
    data = load_activity_data(p, value_col="standard_value")
    np.testing.assert_allclose(data.pIC50, [7, 7])
    assert data.smiles.tolist() == ["CCO", "CCN"]
    pd.DataFrame({"smiles": ["CCO"], "standard_value": [10], "standard_units": ["mg/L"]}).to_csv(
        p, index=False
    )
    with pytest.raises(ValueError, match="units"):
        load_activity_data(p, value_col="standard_value")


def test_plugin_failed_import_rolls_back_and_can_retry(tmp_path):
    p = tmp_path / "broken.py"
    source = (
        "from drugdesign.features import FEATURIZERS, MorganFingerprint\n"
        '@FEATURIZERS.register("transaction_test")\n'
        "class F(MorganFingerprint): pass\n"
    )
    p.write_text(source + 'raise RuntimeError("broken import")\n')
    with pytest.raises(RuntimeError):
        load_plugins([str(p)])
    assert "transaction_test" not in FEATURIZERS
    p.write_text(source)
    try:
        load_plugins([str(p), str(p)])
        assert "transaction_test" in FEATURIZERS
    finally:
        FEATURIZERS._items.pop("transaction_test", None)


def test_same_stem_plugins_are_distinct(tmp_path):
    paths = []
    for name in ("first", "second"):
        d = tmp_path / name
        d.mkdir()
        p = d / "custom.py"
        p.write_text(
            f"from drugdesign.features import FEATURIZERS, MorganFingerprint\n"
            f'FEATURIZERS.register("{name}_test")(MorganFingerprint)\n'
        )
        paths.append(str(p))
    try:
        load_plugins(paths)
        assert "first_test" in FEATURIZERS and "second_test" in FEATURIZERS
    finally:
        for n in ("first_test", "second_test"):
            FEATURIZERS._items.pop(n, None)


def test_plugin_artifact_loads_in_fresh_process(tmp_path, demo_csv):
    from .conftest import EXAMPLES

    plugin = EXAMPLES / "plugins" / "custom_featurizer.py"
    model = tmp_path / "model.joblib"
    common = [sys.executable, "-m", "drugdesign", "--plugin", str(plugin)]
    train = subprocess.run(
        common
        + [
            "train",
            "--data",
            str(demo_csv),
            "--featurizer",
            "atom_pair",
            "--model",
            "hist_gb",
            "--params",
            json.dumps({"max_iter": 5}),
            "--cv",
            "0",
            "--out",
            str(model),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert train.returncode == 0, train.stderr
    predict = subprocess.run(
        common + ["predict", "--model", str(model), "--smiles", "CCO"],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert predict.returncode == 0, predict.stderr
    assert "predicted_pIC50" in predict.stdout


@pytest.mark.parametrize(
    "center,size",
    [([np.nan, 0, 0], [20] * 3), ([0] * 3, [np.inf, 20, 20]), ([0] * 3, [-1, 20, 20])],
)
def test_docking_rejects_nonfinite_box(center, size):
    with pytest.raises(ValueError):
        DockingBox(center, size)


def test_docking_filename_and_timeout_validation(tmp_path, monkeypatch):
    from drugdesign import docking

    with pytest.raises(ValueError, match="Ligand name"):
        dock_smiles("CCO", "x", DockingBox((0, 0, 0)), tmp_path, name="../escape")
    receptor, ligand = tmp_path / "r.pdbqt", tmp_path / "l.pdbqt"
    receptor.write_text("receptor")
    ligand.write_text("ligand")
    monkeypatch.setattr(docking.shutil, "which", lambda _: None)
    with pytest.raises(DockingError, match="timeout requires"):
        run_vina(receptor, ligand, DockingBox((0, 0, 0)), timeout=1)
    with pytest.raises(ValueError, match="overwrite"):
        run_vina(receptor, ligand, DockingBox((0, 0, 0)), output_pdbqt=ligand)


def test_docking_ranks_by_probability_and_keeps_failure(tmp_path, monkeypatch):
    from drugdesign import pipeline

    cfg = PipelineConfig.from_dict(
        {
            "docking": {
                "enabled": True,
                "receptor": str(tmp_path / "r.pdbqt"),
                "center": [0, 0, 0],
                "top_n": 1,
            }
        }
    )
    results = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN"],
            "model_smiles": ["CCO", "CCN"],
            "valid": [True, True],
            "predicted_label": [1, 1],
            "probability_active": [0.1, 0.9],
        }
    )
    calls = []

    def fail(smiles, *args, **kwargs):
        calls.append(smiles)
        raise DockingError("test failure")

    monkeypatch.setattr(pipeline, "dock_smiles", fail)
    out = dock_candidates(results, cfg)
    assert calls == ["CCN"]
    assert out.docking_status.tolist() == ["not_selected", "failed"]
    assert out.loc[1, "docking_error"] == "test failure"


def test_cli_rejects_bad_params_and_config_extension(tmp_path, demo_csv):
    assert main(["train", "--data", str(demo_csv), "--params", "[]"]) == 1
    assert main(["train", "--data", str(demo_csv), "--params", '{"random_state": 1}']) == 1
    assert main(["init-config", "--out", str(tmp_path / "bad.toml")]) == 1
    assert main(["init-config", "--out", str(tmp_path / "nested" / "good.json")]) == 0


def test_outputs_cannot_overwrite_models_or_receptors(tmp_path):
    from drugdesign import QSARModel
    from drugdesign.docking import prepare_receptor

    p = tmp_path / "m.joblib"
    QSARModel("basic_descriptors", "ridge").fit(["CCO", "CCN"], [1, 2]).save(p)
    original = p.read_bytes()
    assert main(["predict", "--model", str(p), "--smiles", "CCO", "--out", str(p)]) == 1
    assert p.read_bytes() == original
    receptor = tmp_path / "ligand.pdbqt"
    receptor.write_text("receptor")
    with pytest.raises(ValueError, match="overwrite"):
        dock_smiles("CCO", receptor, DockingBox((0, 0, 0)), tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        prepare_receptor(receptor, receptor)
    assert receptor.read_text() == "receptor"
