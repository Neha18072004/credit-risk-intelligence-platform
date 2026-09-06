"""Tests for inference, banding and SHAP explanations."""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.predict import (
    RISK_BANDS,
    assign_band,
    load_bundle,
    predict_applicant,
    predict_batch,
    score_frame,
)
from src.utils.config import settings


# ------------------------------------------------------------------ bands --
def test_assign_band_respects_tuned_edges() -> None:
    thresholds = {"band_low_max": 0.10, "band_medium_max": 0.25}
    assert assign_band(0.02, thresholds) == "Low"
    assert assign_band(0.10, thresholds) == "Low"      # inclusive lower edge
    assert assign_band(0.18, thresholds) == "Medium"
    assert assign_band(0.25, thresholds) == "Medium"
    assert assign_band(0.40, thresholds) == "High"


def test_bands_are_not_derived_from_half(trained_artifacts) -> None:
    """Regression guard: a 0.5 cut-off would put everyone in 'Low'."""
    thresholds = load_bundle().thresholds
    assert thresholds["band_medium_max"] < 0.5
    assert thresholds["decision_threshold"] < 0.5


# ------------------------------------------------------------- prediction --
def test_score_frame_shape_and_ranges(trained_artifacts, joined_dataset) -> None:
    scored = score_frame(joined_dataset.head(50))
    assert len(scored) == 50
    assert scored["probability_of_default"].between(0, 1).all()
    assert scored["risk_score"].between(0, settings.risk_score_scale).all()
    assert set(scored["risk_band"].unique()) <= set(RISK_BANDS)
    assert scored["SK_ID_CURR"].tolist() == joined_dataset.head(50)["SK_ID_CURR"].tolist()


def test_risk_score_is_monotone_in_probability(trained_artifacts, joined_dataset) -> None:
    """Higher probability must always mean a higher (riskier) score."""
    scored = score_frame(joined_dataset.head(200)).sort_values("probability_of_default")
    assert scored["risk_score"].is_monotonic_increasing


def test_bands_order_by_mean_probability(trained_artifacts, joined_dataset) -> None:
    scored = score_frame(joined_dataset.head(600))
    means = scored.groupby("risk_band", observed=True)["probability_of_default"].mean()
    present = [band for band in RISK_BANDS if band in means.index]
    assert list(means[present].sort_values().index) == present


def test_predictions_are_deterministic(trained_artifacts, joined_dataset) -> None:
    sample = joined_dataset.head(30)
    first = score_frame(sample)["probability_of_default"].to_numpy()
    second = score_frame(sample)["probability_of_default"].to_numpy()
    np.testing.assert_allclose(first, second)


def test_predict_applicant_rejects_multiple_rows(trained_artifacts, joined_dataset) -> None:
    with pytest.raises(ValueError, match="one row"):
        predict_applicant(joined_dataset.head(3))


def test_predict_accepts_a_series(trained_artifacts, joined_dataset) -> None:
    result = predict_applicant(joined_dataset.iloc[0])
    assert 0 <= result.probability <= 1
    assert result.risk_band in RISK_BANDS


def test_prediction_survives_missing_columns(trained_artifacts, joined_dataset) -> None:
    """The UI submits a partial applicant form; scoring must still work."""
    partial = joined_dataset.iloc[[0]][
        ["SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
         "DAYS_BIRTH", "EXT_SOURCE_2", "CODE_GENDER"]
    ]
    result = predict_applicant(partial)
    assert 0 <= result.probability <= 1
    assert result.decision in {"Approve", "Refer for review"}


def test_decision_follows_threshold(trained_artifacts, joined_dataset) -> None:
    results = predict_batch(joined_dataset.head(300))
    for result in results:
        expected = "Refer for review" if result.probability >= result.threshold else "Approve"
        assert result.decision == expected


def test_result_serialises_to_json_safe_types(trained_artifacts, joined_dataset) -> None:
    import json

    payload = predict_applicant(joined_dataset.iloc[[0]]).to_dict()
    json.dumps(payload)  # must not raise on numpy scalars
    assert payload["top_contributions"]
    assert {"feature", "value", "contribution", "direction"} <= set(payload["top_contributions"][0])


# ----------------------------------------------------------------- SHAP ---
def test_explanation_returns_ranked_contributions(trained_artifacts, joined_dataset) -> None:
    result = predict_applicant(joined_dataset.iloc[[0]], top_n=8)
    contributions = result.top_contributions
    assert len(contributions) == 8
    magnitudes = [abs(c.contribution) for c in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all(c.feature for c in contributions)
    assert all(c.direction in {"increases risk", "decreases risk"} for c in contributions)


def test_explanations_differ_between_applicants(trained_artifacts, joined_dataset) -> None:
    """A local explanation must be local, not a global ranking in disguise."""
    scored = score_frame(joined_dataset.head(400)).sort_values("probability_of_default")
    safest = joined_dataset.loc[[scored.index[0]]]
    riskiest = joined_dataset.loc[[scored.index[-1]]]

    low = predict_applicant(safest, top_n=6)
    high = predict_applicant(riskiest, top_n=6)
    assert low.probability < high.probability

    low_values = [c.contribution for c in low.top_contributions]
    high_values = [c.contribution for c in high.top_contributions]
    assert low_values != high_values


def test_global_importance_ranks_features(trained_artifacts, joined_dataset) -> None:
    bundle = load_bundle()
    features = bundle.preprocessor.transform(joined_dataset.head(300))
    if bundle.metadata.get("requires_string_categoricals"):
        for column in bundle.metadata["categorical_features"]:
            features[column] = features[column].astype(str)

    importance = bundle.explainer.global_importance(features[bundle.feature_names], top_n=15)
    assert len(importance) == 15
    assert importance["mean_abs_shap"].is_monotonic_decreasing
    assert importance["importance_pct"].sum() <= 100.0
    # External scores dominate in the EDA; the model must agree.
    assert any("EXT_SOURCE" in feature for feature in importance["feature"].head(6))
