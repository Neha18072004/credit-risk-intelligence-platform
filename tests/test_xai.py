"""Tests for SHAP explanations and the surrogate policy rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.config import settings
from src.xai.rules import (
    Condition,
    PolicyRule,
    derive_rules_from_trained_model,
    fit_surrogate,
    humanise,
)


# --------------------------------------------------------------- rendering --
def test_humanise_maps_engineered_names() -> None:
    assert humanise("EXT_SOURCE_MEAN") == "average external credit score"
    assert humanise("CREDIT_TO_INCOME_RATIO") == "loan-to-income ratio"
    # One-hot indicators are already readable and pass through untouched.
    assert humanise("NAME_EDUCATION_TYPE = Higher education") == (
        "NAME_EDUCATION_TYPE = Higher education"
    )
    # Unknown names degrade to something readable rather than raising.
    assert humanise("SOME_NEW_COLUMN") == "some new column"


def test_condition_renders_indicator_as_membership() -> None:
    positive = Condition("NAME_INCOME_TYPE = Pensioner", ">", 0.5, is_indicator=True)
    negative = Condition("NAME_INCOME_TYPE = Pensioner", "<=", 0.5, is_indicator=True)
    assert positive.describe() == "NAME_INCOME_TYPE = Pensioner"
    assert negative.describe().startswith("NOT ")


def test_rule_renders_as_if_then_policy() -> None:
    rule = PolicyRule(
        rule_id=1,
        conditions=[
            Condition("EXT_SOURCE_MEAN", "<=", 0.4),
            Condition("CREDIT_TO_INCOME_RATIO", ">", 5.0),
        ],
        predicted_probability=0.31,
        n_applicants=200,
        coverage_pct=5.0,
        observed_default_rate=29.5,
        band="High",
    )
    text = rule.describe()
    assert "IF   average external credit score <= 0.400" in text
    assert "AND  loan-to-income ratio > 5.000" in text
    assert "THEN predicted default risk 31.0%" in text
    assert "High risk band" in text
    assert "observed default rate" in text


# ---------------------------------------------------------------- fitting --
def test_surrogate_recovers_a_known_rule() -> None:
    """A surrogate must reproduce structure that is genuinely in the scores."""
    rng = np.random.default_rng(0)
    n = 1500
    features = pd.DataFrame(
        {
            "EXT_SOURCE_MEAN": rng.uniform(0, 1, n),
            "CREDIT_TO_INCOME_RATIO": rng.uniform(1, 10, n),
            "NOISE": rng.normal(0, 1, n),
        }
    )
    # Risk is high only when the score is low; the surrogate should find that.
    probabilities = np.where(features["EXT_SOURCE_MEAN"] < 0.35, 0.40, 0.03)

    report = fit_surrogate(features, probabilities, max_depth=3)
    assert report.fidelity["r2"] > 0.9
    used = {condition.feature for rule in report.rules for condition in rule.conditions}
    assert "EXT_SOURCE_MEAN" in used


def test_surrogate_respects_max_depth() -> None:
    rng = np.random.default_rng(1)
    features = pd.DataFrame(rng.normal(size=(800, 6)), columns=[f"F{i}" for i in range(6)])
    probabilities = np.clip(0.1 + 0.05 * features["F0"], 0, 1).to_numpy()

    report = fit_surrogate(features, probabilities, max_depth=3)
    assert report.fidelity["depth"] <= 3
    assert all(len(rule.conditions) <= 3 for rule in report.rules)


def test_surrogate_handles_categoricals_as_indicators() -> None:
    rng = np.random.default_rng(2)
    n = 1200
    education = rng.choice(["Higher", "Secondary", "Lower"], n, p=[0.4, 0.5, 0.1])
    features = pd.DataFrame(
        {"AMT_CREDIT": rng.uniform(1e5, 1e6, n), "NAME_EDUCATION_TYPE": education}
    )
    probabilities = np.where(education == "Lower", 0.35, 0.04)

    report = fit_surrogate(
        features, probabilities, categorical_features=["NAME_EDUCATION_TYPE"], max_depth=3
    )
    used = {condition.feature for rule in report.rules for condition in rule.conditions}
    assert any("NAME_EDUCATION_TYPE = " in feature for feature in used)
    assert report.fidelity["r2"] > 0.9


def test_surrogate_tolerates_missing_values() -> None:
    """The tree cannot branch on NaN the way the boosters can, so it imputes."""
    rng = np.random.default_rng(3)
    features = pd.DataFrame(
        {"A": rng.normal(size=600), "B": rng.normal(size=600), "C": rng.normal(size=600)}
    )
    features.loc[features.sample(200, random_state=1).index, "A"] = np.nan
    probabilities = np.clip(0.1 + 0.02 * features["B"], 0, 1).to_numpy()

    report = fit_surrogate(features, probabilities, max_depth=3)
    assert np.isfinite(report.fidelity["r2"])


def test_rules_cover_the_whole_population() -> None:
    """Leaves partition the data, so coverage must sum to 100%."""
    rng = np.random.default_rng(4)
    features = pd.DataFrame(rng.normal(size=(1000, 4)), columns=list("ABCD"))
    probabilities = np.clip(0.1 + 0.04 * features["A"], 0.001, 0.999).to_numpy()

    report = fit_surrogate(features, probabilities, max_depth=3)
    assert sum(rule.coverage_pct for rule in report.rules) == pytest.approx(100.0, abs=0.5)
    assert sum(rule.n_applicants for rule in report.rules) == 1000


# ----------------------------------------------------- against real model --
def test_derived_rules_track_the_deployed_model(trained_artifacts) -> None:
    """Fidelity is the point: a surrogate that misleads is worse than none."""
    report = derive_rules_from_trained_model(sample_size=2000)

    assert report.fidelity["r2"] > 0.4, "surrogate does not resemble the model"
    assert report.fidelity["band_agreement_pct"] > 60
    assert report.fidelity["depth"] <= settings.surrogate_tree_max_depth
    assert 2 <= len(report.rules) <= 40


def test_derived_rules_are_ordered_and_banded(trained_artifacts) -> None:
    report = derive_rules_from_trained_model(sample_size=2000)
    frame = report.to_frame()

    assert frame["predicted_default_pct"].is_monotonic_increasing
    assert set(frame["risk_band"]) <= {"Low", "Medium", "High"}
    # The riskiest rule must actually be riskier than the safest.
    assert frame.iloc[-1]["predicted_default_pct"] > frame.iloc[0]["predicted_default_pct"]


def test_rule_predictions_track_observed_rates(trained_artifacts) -> None:
    """Predicted and realised default rates should correlate across leaves."""
    frame = derive_rules_from_trained_model(sample_size=3000).to_frame().dropna(
        subset=["observed_default_pct"]
    )
    assert len(frame) >= 4
    correlation = frame["predicted_default_pct"].corr(frame["observed_default_pct"])
    assert correlation > 0.8, f"rules do not track reality (r={correlation:.2f})"


def test_rules_render_and_tabulate(trained_artifacts) -> None:
    report = derive_rules_from_trained_model(sample_size=1500)
    text = report.render()
    assert "SURROGATE POLICY RULES" in text
    assert "Fidelity to the deployed model" in text
    assert "IF" in text and "THEN" in text

    frame = report.to_frame()
    assert {"rule_id", "conditions", "predicted_default_pct", "coverage_pct", "risk_band"} <= set(
        frame.columns
    )


# ----------------------------------------------------------- SHAP figures --
def test_shap_figures_are_written(trained_artifacts, joined_dataset) -> None:
    from src.ml.predict import load_bundle, predict_applicant
    from src.xai.shap_explainer import plot_global_importance, plot_local_explanation

    bundle = load_bundle()
    sample = joined_dataset.head(200)
    features = bundle.preprocessor.transform(sample)
    if bundle.metadata.get("requires_string_categoricals"):
        for column in bundle.metadata["categorical_features"]:
            features[column] = features[column].astype(str)

    importance = bundle.explainer.global_importance(features[bundle.feature_names], top_n=12)
    assert plot_global_importance(importance).exists()

    result = predict_applicant(sample.iloc[[0]], top_n=8)
    assert plot_local_explanation(
        result.top_contributions, result.probability, result.applicant_id
    ).exists()
