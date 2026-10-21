import numpy as np
import pandas as pd
import pytest

from drugdesign.data import (
    clean_dataset,
    load_activity_data,
    read_smiles_file,
    standardize_smiles,
    to_pactivity,
)


def test_to_pactivity_units():
    np.testing.assert_allclose(to_pactivity([1.0, 1000.0], "nM"), [9.0, 6.0])
    np.testing.assert_allclose(to_pactivity(1.0, "uM"), 6.0)
    assert np.isnan(to_pactivity([0.0, -1.0])).all()
    with pytest.raises(ValueError):
        to_pactivity(1.0, "furlongs")


def test_standardize_strips_salts_and_rejects_invalid():
    assert standardize_smiles("CC(=O)Nc1ccc(O)cc1.Cl") == standardize_smiles("CC(=O)Nc1ccc(O)cc1")
    assert standardize_smiles("CC(=O)[O-].[Na+]") == "CC(=O)O"
    assert standardize_smiles("not_a_smiles") is None
    assert standardize_smiles("") is None
    assert standardize_smiles(None) is None


def test_load_and_clean(demo_csv):
    raw = load_activity_data(demo_csv)
    assert list(raw.columns) == ["smiles", "pIC50"]
    assert raw["pIC50"].notna().all()
    clean = clean_dataset(raw)
    assert clean["smiles"].is_unique
    assert len(clean) < len(raw)
    assert "not_a_smiles" not in set(clean["smiles"])


def test_load_derives_pic50_from_raw_values(tmp_path):
    path = tmp_path / "chembl.csv"
    pd.DataFrame({"canonical_smiles": ["CCO", "CCN"], "standard_value": [100.0, 10.0]}).to_csv(
        path, index=False
    )
    df = load_activity_data(path, smiles_col="canonical_smiles", value_col="standard_value")
    np.testing.assert_allclose(df["pIC50"], [7.0, 8.0])


def test_load_semicolon_chembl_export(tmp_path):
    path = tmp_path / "chembl.csv"
    path.write_text("smiles;pIC50\nCCO;5.0\n")
    assert load_activity_data(path)["pIC50"].tolist() == [5.0]


def test_load_missing_columns(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("smiles,other\nCCO,1\n")
    with pytest.raises(KeyError):
        load_activity_data(path)


def test_read_smi_file(tmp_path):
    path = tmp_path / "c.smi"
    path.write_text("# header\nCCO ethanol\nc1ccccc1\n")
    df = read_smiles_file(path)
    assert df["smiles"].tolist() == ["CCO", "c1ccccc1"]
    assert df["name"].tolist()[0] == "ethanol"
