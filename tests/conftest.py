from pathlib import Path

import pytest

from drugdesign.data import quiet_rdkit

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

quiet_rdkit()


@pytest.fixture
def demo_csv() -> Path:
    return EXAMPLES / "data" / "demo_activity.csv"


@pytest.fixture
def demo_data(demo_csv):
    from drugdesign.data import clean_dataset, load_activity_data

    return clean_dataset(load_activity_data(demo_csv))
