"""Tests for the bake-off, calibration and threshold tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss

from src.ml.train import (
    CandidateResult,
    build_logistic_pipeline,
    fit_calibrator,
    select_winner,
    tune_thresholds,
)
from src.utils.config import settings


def _candidate(name: str, pr_auc: float, folds: list[float], shap: bool, seconds: float):
    return CandidateResult(
        name=name, pr_auc=pr_auc, roc_auc=0.8, brier=0.1, log_loss=0.4,
        fit_seconds=seconds, supports_shap=shap, pr_auc_folds=folds,
    )


# ------------------------------------------------------------- selection --
def test_select_winner_prefers_clear_pr_auc_leader() -> None:
    """A lead well outside the sampling error must win outright."""
    tight = [0.60, 0.61, 0.60, 0.59, 0.60]
    winner = select_winner(
        [
            _candidate("logistic", 0.30, [0.30] * 5, False, 1.0),
            _candidate("lightgbm", 0.60, tight, True, 2.0),
        ]
    )
    assert winner.name == "lightgbm"


def test_select_winner_breaks_statistical_ties_on_explainability() -> None:
    """Inside one standard error, exact tree SHAP decides the choice.

    A marginally higher point estimate is not worth losing per-applicant
    explanations on a model that has to justify every decision.
    """
    noisy = [0.20, 0.45, 0.25, 0.40, 0.30]  # wide spread -> large standard error
    winner = select_winner(
        [
            _candidate("logistic", 0.34, [0.33, 0.35, 0.34, 0.34, 0.34], False, 0.5),
            _candidate("catboost", 0.32, noisy, True, 4.0),
        ]
    )
    assert winner.name == "catboost"


def test_select_winner_prefers_faster_model_among_equals() -> None:
    folds = [0.50] * 5
    winner = select_winner(
        [
            _candidate("catboost", 0.50, folds, True, 9.0),
            _candidate("lightgbm", 0.50, folds, True, 1.0),
        ]
    )
    assert winner.name == "lightgbm"


def test_pr_auc_stderr_shrinks_with_agreement() -> None:
    agree = _candidate("a", 0.5, [0.50, 0.50, 0.50, 0.50, 0.50], True, 1.0)
    disagree = _candidate("b", 0.5, [0.30, 0.70, 0.40, 0.60, 0.50], True, 1.0)
    assert agree.pr_auc_stderr < disagree.pr_auc_stderr
    assert _candidate("c", 0.5, [], True, 1.0).pr_auc_stderr == 0.0


# ----------------------------------------------------------- calibration --
def test_calibrator_improves_brier_and_matches_base_rate() -> None:
    """Weighted training inflates scores; calibration must undo that."""
    rng = np.random.default_rng(0)
    labels = pd.Series(rng.binomial(1, 0.09, 2000))
    # Well-ranked but badly scaled, as scale_pos_weight produces.
    inflated = np.clip(0.35 + 0.4 * (labels + rng.normal(0, 0.5, 2000)), 0.01, 0.99)

    calibrator = fit_calibrator(inflated, labels)
    calibrated = np.clip(calibrator.predict(inflated), 0, 1)

    assert brier_score_loss(labels, calibrated) < brier_score_loss(labels, inflated)
    # A calibrated model's mean prediction should match the observed base rate.
    assert calibrated.mean() == pytest.approx(labels.mean(), abs=0.02)


def test_calibration_preserves_ranking() -> None:
    """Calibration is monotone, so it must not damage ranking quality."""
    rng = np.random.default_rng(1)
    labels = pd.Series(rng.binomial(1, 0.1, 1500))
    scores = np.clip(0.3 + 0.3 * labels + rng.normal(0, 0.25, 1500), 0.01, 0.99)

    calibrated = np.clip(fit_calibrator(scores, labels).predict(scores), 0, 1)
    before = average_precision_score(labels, scores)
    after = average_precision_score(labels, calibrated)
    assert after >= before - 0.02


# ------------------------------------------------------------ thresholds --
def test_tuned_threshold_is_not_naive_half() -> None:
    """At a ~9% base rate, 0.5 would approve essentially everyone."""
    rng = np.random.default_rng(2)
    labels = pd.Series(rng.binomial(1, 0.09, 3000))
    scores = np.clip(0.06 + 0.25 * labels + rng.normal(0, 0.06, 3000), 0.001, 0.999)

    thresholds = tune_thresholds(scores, labels)
    assert thresholds["decision_threshold"] != 0.5
    assert 0.0 < thresholds["decision_threshold"] < 0.5
    assert thresholds["band_low_max"] < thresholds["band_medium_max"]


def test_band_profile_is_monotonic_in_risk() -> None:
    """Bands are only meaningful if realised default rate rises across them."""
    rng = np.random.default_rng(3)
    labels = pd.Series(rng.binomial(1, 0.09, 3000))
    scores = np.clip(0.06 + 0.3 * labels + rng.normal(0, 0.05, 3000), 0.001, 0.999)

    profile = pd.DataFrame(tune_thresholds(scores, labels)["band_profile"]).set_index("band")
    assert profile.loc["Low", "default_rate"] < profile.loc["Medium", "default_rate"]
    assert profile.loc["Medium", "default_rate"] <= profile.loc["High", "default_rate"]
    assert profile["population_share"].sum() == pytest.approx(100.0, abs=0.1)


# --------------------------------------------------------------- pipeline --
def test_logistic_pipeline_handles_nan_and_categories(feature_matrix, fitted_preprocessor) -> None:
    """The linear baseline carries its own imputation and encoding."""
    pipeline = build_logistic_pipeline(
        fitted_preprocessor.numeric_features_, fitted_preprocessor.categorical_features_
    )
    labels = np.random.default_rng(4).binomial(1, 0.1, len(feature_matrix))
    pipeline.fit(feature_matrix, labels)
    probabilities = pipeline.predict_proba(feature_matrix)[:, 1]
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


# ------------------------------------------------------- end-to-end train --
def test_training_produces_all_artifacts(trained_artifacts) -> None:
    from src.ml.train import (
        BAKEOFF_FILE, CALIBRATOR_FILE, METADATA_FILE,
        METRICS_FILE, MODEL_FILE, PREPROCESSOR_FILE, THRESHOLDS_FILE,
    )

    for name in (MODEL_FILE, PREPROCESSOR_FILE, CALIBRATOR_FILE,
                 METADATA_FILE, THRESHOLDS_FILE, METRICS_FILE, BAKEOFF_FILE):
        assert (settings.models_dir / name).exists(), f"missing artifact {name}"


def test_bakeoff_compares_all_three_candidates(trained_artifacts) -> None:
    comparison = trained_artifacts["comparison"]
    assert set(comparison["name"]) == {"logistic", "lightgbm", "catboost"}
    assert (comparison["pr_auc"] > 0).all()


def test_model_beats_random_baseline(trained_artifacts) -> None:
    """PR-AUC must clearly exceed the no-skill baseline, which is the base rate."""
    metrics = trained_artifacts["metrics"]
    base_rate = metrics["default_rate"]
    assert metrics["oof_calibrated"]["pr_auc"] > 2 * base_rate


def test_final_model_is_calibrated(trained_artifacts) -> None:
    metrics = trained_artifacts["metrics"]["oof_calibrated"]
    assert metrics["mean_predicted"] == pytest.approx(metrics["observed_rate"], abs=0.01)
    assert metrics["brier"] < trained_artifacts["metrics"]["oof_uncalibrated"]["brier"]


# ------------------------------------------------------ hyperparameters ----
def test_search_spaces_cover_every_candidate() -> None:
    from src.ml.train import SEARCH_SPACES

    assert set(SEARCH_SPACES) == {"logistic", "lightgbm", "catboost"}
    for name, space in SEARCH_SPACES.items():
        assert space, f"{name} has an empty search space"
        for parameter, values in space.items():
            assert len(values) >= 2, f"{name}.{parameter} has nothing to search"


def test_tuning_scores_on_the_selection_metric(feature_matrix, fitted_preprocessor) -> None:
    """Tuning for one objective and selecting on another produces a model that
    looks better and performs worse, so both use average precision."""
    import numpy as np
    import pandas as pd

    from src.ml.train import tune_candidate

    rng = np.random.default_rng(0)
    labels = pd.Series(rng.binomial(1, 0.1, len(feature_matrix)), index=feature_matrix.index)
    result = tune_candidate(
        "logistic", feature_matrix, labels, fitted_preprocessor, 10.0, n_iter=2
    )
    assert "model__C" in result["best_params"]
    assert 0.0 <= result["best_score"] <= 1.0
    assert result["n_candidates"] == 2


def test_tuned_parameters_reach_the_model(fitted_preprocessor) -> None:
    """A search that does not change the fitted estimator is decoration."""
    from src.ml.train import _instantiate

    default = _instantiate("lightgbm", fitted_preprocessor, 10.0)
    tuned = _instantiate(
        "lightgbm", fitted_preprocessor, 10.0, params={"num_leaves": 8, "learning_rate": 0.02}
    )
    assert tuned.get_params()["num_leaves"] == 8
    assert tuned.get_params()["learning_rate"] == 0.02
    assert default.get_params()["num_leaves"] != 8


def test_tuning_is_off_by_default(joined_dataset) -> None:
    """The documented quick-start must stay quick."""
    from src.utils.config import Settings

    assert Settings(_env_file=None).tune_hyperparameters is False


def test_metrics_record_the_search(trained_artifacts) -> None:
    """The search space and outcome must be recoverable from the artifacts."""
    tuning = trained_artifacts["metrics"]["tuning"]
    assert "enabled" in tuning
    assert tuning["scoring"] == "average_precision"


def test_guard_blocks_overwriting_a_larger_model(tmp_path) -> None:
    """A sample-mode run must not silently replace a model trained on real data.

    This happened twice during development -- once from the test suite, once
    from a manual command -- and both times the only symptom was the reported
    metrics quietly changing.
    """
    from src.ml.train import METRICS_FILE, ArtifactOverwriteError, _guard_existing_artifacts
    from src.utils.helpers import write_json

    write_json({"n_rows": 307_511}, tmp_path / METRICS_FILE)

    with pytest.raises(ArtifactOverwriteError, match="307,511"):
        _guard_existing_artifacts(tmp_path, n_rows=4_000)

    # Explicit override, a comparable retrain, and a larger one all proceed.
    _guard_existing_artifacts(tmp_path, n_rows=4_000, force=True)
    _guard_existing_artifacts(tmp_path, n_rows=307_511)
    _guard_existing_artifacts(tmp_path, n_rows=400_000)


def test_guard_is_silent_on_a_first_run(tmp_path) -> None:
    from src.ml.train import _guard_existing_artifacts

    _guard_existing_artifacts(tmp_path, n_rows=1_000)
