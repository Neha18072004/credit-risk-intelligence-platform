"""Tests for the evaluation report and its metric policy."""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.evaluate import classification_report_at, evaluate, load_oof


def test_classification_report_confusion_reconciles() -> None:
    probabilities = np.array([0.9, 0.8, 0.2, 0.1, 0.6])
    labels = np.array([1, 0, 0, 0, 1])
    report = classification_report_at(probabilities, labels, 0.5)

    confusion = report["confusion"]
    assert sum(confusion.values()) == len(labels)
    assert confusion["true_positive_default_caught"] == 2
    assert confusion["false_positive_good_declined"] == 1
    assert confusion["true_negative_good_approved"] == 2
    assert confusion["false_negative_default_missed"] == 0
    assert report["recall"] == pytest.approx(1.0)


def test_classification_report_handles_no_flags() -> None:
    """A threshold above every score must not divide by zero."""
    report = classification_report_at(np.array([0.1, 0.2]), np.array([0, 1]), 0.99)
    assert report["precision"] == 0.0
    assert report["flagged_share"] == 0.0


def test_evaluation_reports_pr_auc_against_baseline(trained_artifacts) -> None:
    report = evaluate(save=False)
    ranking = report["ranking"]
    # The no-skill PR-AUC is the base rate; reporting it stops the headline
    # number being read without context.
    assert ranking["pr_auc_baseline"] == pytest.approx(report["default_rate"], abs=1e-6)
    assert ranking["pr_auc"] > ranking["pr_auc_baseline"]
    assert ranking["pr_auc_lift_over_random"] > 2.0


def test_tuned_threshold_beats_naive_on_recall(trained_artifacts) -> None:
    """The whole point of tuning: catch materially more defaults than 0.5 does."""
    report = evaluate(save=False)
    tuned = report["at_tuned_threshold"]
    naive = report["at_default_threshold"]

    assert tuned["threshold"] < 0.5
    assert tuned["recall"] > naive["recall"]
    assert (
        tuned["confusion"]["true_positive_default_caught"]
        > naive["confusion"]["true_positive_default_caught"]
    )


def test_oof_predictions_align_with_labels(trained_artifacts) -> None:
    probabilities, labels = load_oof()
    assert len(probabilities) == len(labels)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert set(np.unique(labels)) == {0, 1}


def test_band_profile_present_and_ordered(trained_artifacts) -> None:
    report = evaluate(save=False)
    bands = {row["band"]: row for row in report["band_profile"]}
    assert set(bands) == {"Low", "Medium", "High"}
    assert bands["Low"]["default_rate"] < bands["High"]["default_rate"]
    assert sum(row["population_share"] for row in report["band_profile"]) == pytest.approx(
        100.0, abs=0.1
    )


def test_evaluation_writes_figures(trained_artifacts) -> None:
    from src.utils.config import settings

    evaluate(save=True)
    for name in ("10_pr_roc_curves.png", "11_calibration.png", "12_risk_bands.png"):
        assert (settings.figures_dir / name).exists(), f"missing figure {name}"
