"""Inference: calibrated probability, risk score, risk band and explanation.

Loads the artifacts written by :mod:`src.ml.train` once, caches them, and
scores either a single applicant or a batch. The same preprocessing object used
in training is reloaded here, so an applicant is transformed by exactly the same
logic that produced the training matrix -- the usual source of silent
train/serve skew.

Typical use::

    from src.ml.predict import predict_applicant

    result = predict_applicant(applicant_row)
    print(result.risk_band, result.risk_score)
    for contribution in result.top_contributions:
        print(contribution.describe())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.preprocessor import CreditPreprocessor
from src.ml.train import (
    CALIBRATOR_FILE,
    METADATA_FILE,
    MODEL_FILE,
    PREPROCESSOR_FILE,
    THRESHOLDS_FILE,
)
from src.utils.config import settings
from src.utils.helpers import probability_to_score, read_json
from src.utils.logger import get_logger
from src.xai.shap_explainer import FeatureContribution, ShapExplainer

logger = get_logger(__name__)

RISK_BANDS: tuple[str, str, str] = ("Low", "Medium", "High")


@dataclass(frozen=True)
class PredictionResult:
    """A scored applicant, with everything needed to justify the decision."""

    probability: float
    risk_score: float
    risk_band: str
    decision: str
    threshold: float
    applicant_id: int | None = None
    top_contributions: list[FeatureContribution] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Flat, JSON-serialisable view for APIs and the UI."""
        return {
            "applicant_id": self.applicant_id,
            "probability_of_default": round(self.probability, 6),
            "risk_score": self.risk_score,
            "risk_band": self.risk_band,
            "decision": self.decision,
            "decision_threshold": self.threshold,
            "top_contributions": [
                {
                    "feature": c.feature,
                    "value": c.value if not isinstance(c.value, np.generic) else c.value.item(),
                    "contribution": round(c.contribution, 6),
                    "direction": c.direction,
                }
                for c in self.top_contributions
            ],
        }


@dataclass
class ModelBundle:
    """Every artifact inference needs, loaded once and reused."""

    model: Any
    preprocessor: CreditPreprocessor
    calibrator: Any
    metadata: dict[str, Any]
    thresholds: dict[str, Any]
    explainer: ShapExplainer

    @property
    def model_name(self) -> str:
        return str(self.metadata["model_name"])

    @property
    def feature_names(self) -> list[str]:
        return list(self.metadata["feature_names"])


def artifacts_exist(directory: Path | None = None) -> bool:
    """Whether a trained model is available to load."""
    target = directory or settings.models_dir
    return all(
        (target / name).exists()
        for name in (MODEL_FILE, PREPROCESSOR_FILE, CALIBRATOR_FILE, METADATA_FILE, THRESHOLDS_FILE)
    )


@lru_cache(maxsize=1)
def load_bundle() -> ModelBundle:
    """Load and cache the trained artifacts.

    Returns:
        The assembled :class:`ModelBundle`.

    Raises:
        FileNotFoundError: If the model has not been trained yet. The message
            names the exact command to fix it, because this is the first error
            a new user hits.
    """
    directory = settings.models_dir
    if not artifacts_exist(directory):
        raise FileNotFoundError(
            f"No trained model in {directory}. Train one first:\n"
            "    python -m src.ml.train"
        )

    metadata = read_json(directory / METADATA_FILE)
    model = joblib.load(directory / MODEL_FILE)
    bundle = ModelBundle(
        model=model,
        preprocessor=joblib.load(directory / PREPROCESSOR_FILE),
        calibrator=joblib.load(directory / CALIBRATOR_FILE),
        metadata=metadata,
        thresholds=read_json(directory / THRESHOLDS_FILE),
        explainer=ShapExplainer(
            model=model,
            model_name=metadata["model_name"],
            feature_names=metadata["feature_names"],
            categorical_features=metadata["categorical_features"],
        ),
    )
    logger.info("Loaded %s model with %d features", bundle.model_name, len(bundle.feature_names))
    return bundle


def _prepare(frame: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """Transform raw applicant rows into the exact matrix the model expects."""
    features = bundle.preprocessor.transform(frame)
    if bundle.metadata.get("requires_string_categoricals"):
        for column in bundle.metadata["categorical_features"]:
            features[column] = features[column].astype(str)
    return features[bundle.feature_names]


def assign_band(probability: float, thresholds: dict[str, Any]) -> str:
    """Map a calibrated probability onto a risk band.

    Band edges come from :func:`src.ml.train.tune_thresholds`, which derives
    them from out-of-fold predictions -- never the naive 0.5 cut-off, which at a
    ~9% base rate would place virtually every applicant in the lowest band.
    """
    if probability <= thresholds["band_low_max"]:
        return "Low"
    if probability <= thresholds["band_medium_max"]:
        return "Medium"
    return "High"


def predict_batch(
    frame: pd.DataFrame, explain: bool = False, top_n: int = 10
) -> list[PredictionResult]:
    """Score a batch of applicants.

    Args:
        frame: Raw applicant rows, in the same shape as the training data. Only
            the columns the pipeline needs are used; the rest are ignored, and
            absent ones are filled with NaN.
        explain: Compute per-applicant SHAP contributions. Off by default
            because it costs materially more than scoring alone.
        top_n: Contributions to keep per applicant when ``explain`` is set.

    Returns:
        One :class:`PredictionResult` per input row, in input order.
    """
    bundle = load_bundle()
    features = _prepare(frame, bundle)

    raw = bundle.model.predict_proba(features)[:, 1]
    calibrated = np.clip(bundle.calibrator.predict(raw), 0.0, 1.0)
    threshold = float(bundle.thresholds["decision_threshold"])

    identifiers = (
        frame["SK_ID_CURR"].tolist() if "SK_ID_CURR" in frame.columns else [None] * len(frame)
    )

    results: list[PredictionResult] = []
    for position, probability in enumerate(calibrated):
        contributions: list[FeatureContribution] = []
        if explain:
            contributions = bundle.explainer.explain_row(features.iloc[[position]], top_n=top_n)

        probability = float(probability)
        results.append(
            PredictionResult(
                probability=probability,
                risk_score=float(probability_to_score(probability)),
                risk_band=assign_band(probability, bundle.thresholds),
                decision="Refer for review" if probability >= threshold else "Approve",
                threshold=threshold,
                applicant_id=identifiers[position],
                top_contributions=contributions,
            )
        )
    return results


def predict_applicant(row: pd.DataFrame | pd.Series, top_n: int = 10) -> PredictionResult:
    """Score one applicant, always with an explanation.

    Args:
        row: A single applicant as a one-row frame or a Series.
        top_n: Number of SHAP contributions to return.

    Returns:
        The scored result including its explanation.
    """
    frame = row.to_frame().T if isinstance(row, pd.Series) else row
    if len(frame) != 1:
        raise ValueError(f"predict_applicant expects one row, got {len(frame)}")
    return predict_batch(frame, explain=True, top_n=top_n)[0]


def score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Score a batch and return a tidy dataframe, for the UI and bulk runs.

    Args:
        frame: Raw applicant rows.

    Returns:
        Columns ``SK_ID_CURR`` (when present), ``probability_of_default``,
        ``risk_score``, ``risk_band`` and ``decision``.
    """
    results = predict_batch(frame, explain=False)
    scored = pd.DataFrame(
        {
            "probability_of_default": [r.probability for r in results],
            "risk_score": [r.risk_score for r in results],
            "risk_band": pd.Categorical(
                [r.risk_band for r in results], categories=list(RISK_BANDS), ordered=True
            ),
            "decision": [r.decision for r in results],
        },
        index=frame.index,
    )
    if "SK_ID_CURR" in frame.columns:
        scored.insert(0, "SK_ID_CURR", frame["SK_ID_CURR"].to_numpy())
    return scored


def main() -> None:  # pragma: no cover - CLI entry point
    """Score a handful of sample applicants and print the explanations."""
    from src.data.loader import build_dataset

    data = build_dataset("train").head(5)
    print(f"\nScoring {len(data)} sample applicants with the trained model\n")
    print(score_frame(data).to_string(index=False))

    result = predict_applicant(data.iloc[[0]])
    print(f"\nExplanation for applicant {result.applicant_id}:")
    print(f"  P(default) {result.probability:.4f}  ->  score {result.risk_score:.0f}/"
          f"{settings.risk_score_scale}  ->  {result.risk_band} risk  ->  {result.decision}")
    print("\n  Top contributions (log-odds):")
    for contribution in result.top_contributions:
        print(f"    {contribution.describe()}")


if __name__ == "__main__":  # pragma: no cover
    main()
