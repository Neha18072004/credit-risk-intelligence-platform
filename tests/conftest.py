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


@pytest.fixture(scope="session")
def joined_dataset():
    """The application table LEFT JOINed with aggregated bureau features."""
    from src.data.loader import build_dataset

    return build_dataset("train")


@pytest.fixture(scope="session")
def fitted_preprocessor(joined_dataset):
    """A :class:`CreditPreprocessor` fitted on the sample training split."""
    from src.data.preprocessor import CreditPreprocessor

    return CreditPreprocessor().fit(joined_dataset)


@pytest.fixture(scope="session")
def feature_matrix(fitted_preprocessor, joined_dataset):
    """The model-ready feature frame produced from the sample data."""
    return fitted_preprocessor.transform(joined_dataset)


@pytest.fixture(scope="session")
def trained_artifacts(joined_dataset):
    """Train the full bake-off once and persist artifacts for the session.

    Saving to disk is required because :func:`src.ml.predict.load_bundle` is the
    real inference path and reads from ``models/``. Artifacts are gitignored and
    fully regenerable, so overwriting them is harmless.
    """
    from src.ml import train as train_module
    from src.ml.predict import load_bundle

    outcome = train_module.train(save=True)
    load_bundle.cache_clear()  # pick up the artifacts just written
    return outcome
