"""Shared pytest fixtures.

Every test runs against the committed synthetic sample data, so the suite is
fully self-contained: no Kaggle download, no database, no API key required.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Force sample mode before any project module reads configuration.
os.environ.setdefault("DATA_MODE", "sample")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    """Directory holding the committed synthetic fixtures."""
    return PROJECT_ROOT / "data" / "sample"


@pytest.fixture(scope="session")
def application_train(sample_dir: Path):
    """The synthetic ``application_train`` table."""
    import pandas as pd

    return pd.read_csv(sample_dir / "application_train.csv")


@pytest.fixture(scope="session")
def bureau(sample_dir: Path):
    """The synthetic ``bureau`` table."""
    import pandas as pd

    return pd.read_csv(sample_dir / "bureau.csv")
