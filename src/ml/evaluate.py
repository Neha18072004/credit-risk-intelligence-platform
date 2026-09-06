"""Model evaluation: metrics, curves and the bake-off report.

Everything here reads the out-of-fold predictions saved by
:mod:`src.ml.train`, so evaluation never requires a retrain and never scores a
model on data it was fitted to.

Metric policy: **PR-AUC leads**. At a ~9% default rate ROC-AUC is inflated by
the large negative class and accuracy is actively misleading -- predicting
"everyone repays" scores over 91%. Precision, recall and the confusion matrix
are always reported at the *tuned* threshold, never at 0.5.

Run it with::

    python -m src.ml.evaluate
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.ml.train import METRICS_FILE, THRESHOLDS_FILE
from src.utils.config import settings
from src.utils.helpers import read_json, write_json
from src.utils.logger import get_logger
from src.utils.viz import (
    BASELINE,
    INK_MUTED,
    INK_SECONDARY,
    RISK_BAND_COLORS,
    SERIES,
    apply_theme,
    label_bars,
    new_figure,
    save_figure,
    style_axes,
)

logger = get_logger(__name__)
apply_theme()


def load_oof() -> tuple[np.ndarray, np.ndarray]:
    """Load the saved calibrated out-of-fold predictions and their labels.

    Returns:
        ``(calibrated_probabilities, labels)``.

    Raises:
        FileNotFoundError: If the model has not been trained yet.
    """
    predictions = settings.models_dir / "oof_calibrated.npy"
    labels = settings.models_dir / "oof_labels.npy"
    if not predictions.exists() or not labels.exists():
        raise FileNotFoundError(
            "No out-of-fold predictions found. Train the model first: python -m src.ml.train"
        )
    return np.load(predictions), np.load(labels)


def classification_report_at(
    probabilities: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Precision, recall, F1 and the confusion matrix at one decision threshold.

    Args:
        probabilities: Calibrated default probabilities.
        labels: True binary outcomes.
        threshold: The probability above which an applicant is actioned.

    Returns:
        A dictionary of metrics plus the four confusion-matrix cells, named in
        business terms so the trade-off is legible without a legend.
    """
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 6),
        "precision": round(float(precision_score(labels, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, predicted, zero_division=0)), 4),
        "flagged_share": round(float(predicted.mean()), 4),
        "confusion": {
            "true_negative_good_approved": int(tn),
            "false_positive_good_declined": int(fp),
            "false_negative_default_missed": int(fn),
            "true_positive_default_caught": int(tp),
        },
    }


def evaluate(save: bool = True) -> dict[str, Any]:
    """Produce the full evaluation report from saved out-of-fold predictions.

    Args:
        save: Write the JSON report and figures to ``reports/``.

    Returns:
        The evaluation payload: headline metrics, threshold sweep, band profile
        and the bake-off comparison.
    """
    probabilities, labels = load_oof()
    thresholds = read_json(settings.models_dir / THRESHOLDS_FILE)
    metrics = read_json(settings.models_dir / METRICS_FILE)

    tuned = float(thresholds["decision_threshold"])
    report: dict[str, Any] = {
        "model": metrics["selected_model"],
        "n_rows": int(len(labels)),
        "n_defaults": int(labels.sum()),
        "default_rate": round(float(labels.mean()), 4),
        "ranking": {
            "pr_auc": round(float(average_precision_score(labels, probabilities)), 4),
            "roc_auc": round(float(roc_auc_score(labels, probabilities)), 4),
            # The PR-AUC a random model would achieve: the base rate. Reporting
            # it stops 0.31 from being read as "bad" without context.
            "pr_auc_baseline": round(float(labels.mean()), 4),
        },
        "calibration": {
            "brier": round(float(brier_score_loss(labels, probabilities)), 4),
            "mean_predicted": round(float(probabilities.mean()), 4),
            "observed_rate": round(float(labels.mean()), 4),
            # Stated plainly because the reliability diagram looks close to
            # perfect and would otherwise be over-read.
            "caveat": (
                "The isotonic calibrator is fitted on these same out-of-fold "
                "predictions, so this reliability curve is in-sample and "
                "optimistic. The ranking metrics above are genuinely "
                "out-of-fold; the calibration fit is not. A held-out set "
                "would be required to measure calibration honestly."
            ),
        },
        "at_tuned_threshold": classification_report_at(probabilities, labels, tuned),
        "at_default_threshold": classification_report_at(probabilities, labels, 0.5),
        "band_profile": thresholds["band_profile"],
        "bakeoff": metrics["bakeoff"],
    }
    report["ranking"]["pr_auc_lift_over_random"] = round(
        report["ranking"]["pr_auc"] / max(report["ranking"]["pr_auc_baseline"], 1e-9), 2
    )

    if save:
        settings.ensure_directories()
        write_json(report, settings.reports_dir / "evaluation.json")
        plot_pr_and_roc(probabilities, labels, report)
        plot_calibration(probabilities, labels)
        plot_band_profile(thresholds["band_profile"])
        logger.info("Evaluation artifacts written to %s", settings.reports_dir)

    return report


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_pr_and_roc(
    probabilities: np.ndarray, labels: np.ndarray, report: dict[str, Any]
) -> Path:
    """Precision-recall and ROC curves, side by side.

    PR is placed first and given the wider panel: it is the metric that governs
    the decision here, and its no-skill baseline (the default rate) makes the
    difficulty of the problem visible in a way the ROC diagonal does not.
    """
    import matplotlib.pyplot as plt

    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for ax in (ax_pr, ax_roc):
        style_axes(ax, xgrid=True, ygrid=True)

    precision, recall, _ = precision_recall_curve(labels, probabilities)
    ax_pr.plot(recall, precision, color=SERIES[0], linewidth=2.0)
    baseline = float(labels.mean())
    ax_pr.axhline(baseline, color=INK_MUTED, linewidth=1.0, linestyle="--")
    ax_pr.text(
        0.98, baseline, f" no-skill {baseline:.3f}",
        va="bottom", ha="right", fontsize=8, color=INK_MUTED,
    )
    ax_pr.set_xlabel("Recall (share of defaults caught)")
    ax_pr.set_ylabel("Precision (share of flags that default)")
    ax_pr.set_title(f"Precision-recall  --  PR-AUC {report['ranking']['pr_auc']:.3f}")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, min(1.0, float(precision.max()) * 1.1))

    false_positive, true_positive, _ = roc_curve(labels, probabilities)
    ax_roc.plot(false_positive, true_positive, color=SERIES[0], linewidth=2.0)
    ax_roc.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1.0, linestyle="--")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title(f"ROC  --  AUC {report['ranking']['roc_auc']:.3f}")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)

    fig.suptitle(
        f"Out-of-fold discrimination ({report['model']}, {settings.cv_folds}-fold)",
        x=0.02, ha="left", fontsize=12, fontweight="600",
    )
    fig.tight_layout()
    return save_figure(fig, "10_pr_roc_curves")


def plot_calibration(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> Path:
    """Reliability diagram: predicted probability against observed frequency.

    The point of calibration is that a score of 0.20 means one-in-five actually
    default. This chart is the evidence for that claim, so it is worth as much
    as the ranking metrics.

    Caveat, annotated on the figure itself: the calibrator was fitted on these
    same out-of-fold predictions, so the curve is in-sample with respect to the
    calibration step and will look better than it would on fresh data.
    """
    observed, predicted = calibration_curve(labels, probabilities, n_bins=bins, strategy="quantile")

    fig, ax = new_figure(figsize=(6.4, 5.4))
    limit = max(float(predicted.max()), float(observed.max())) * 1.1
    ax.plot([0, limit], [0, limit], color=INK_MUTED, linewidth=1.0, linestyle="--")
    ax.text(limit, limit, " perfect", fontsize=8, color=INK_MUTED, va="center")
    ax.plot(predicted, observed, color=SERIES[0], linewidth=2.0, marker="o", markersize=6)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_title(
        f"Calibration  --  Brier {brier_score_loss(labels, probabilities):.4f}\n"
        f"mean predicted {probabilities.mean():.3f} vs observed {labels.mean():.3f}"
    )
    ax.text(
        0.0, -0.16,
        "In-sample with respect to calibration: the isotonic fit used these same\n"
        "out-of-fold predictions, so the true curve on fresh data would be looser.",
        transform=ax.transAxes, fontsize=8, color=INK_SECONDARY, va="top",
    )
    fig.tight_layout()
    return save_figure(fig, "11_calibration")


def plot_band_profile(band_profile: list[dict[str, Any]]) -> Path:
    """Realised default rate and population share for each risk band.

    Two panels sharing a category axis rather than one dual-axis plot. Band
    colours are the reserved status palette and every bar is labelled, so the
    band is never communicated by colour alone.
    """
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(band_profile)
    colors = [RISK_BAND_COLORS[band] for band in frame["band"]]

    fig, (ax_rate, ax_share) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax in (ax_rate, ax_share):
        style_axes(ax)

    bars = ax_rate.bar(frame["band"], frame["default_rate"], color=colors, width=0.55)
    label_bars(ax_rate, bars, frame["default_rate"].tolist(), fmt="{:.1f}%", pad=0.03)
    ax_rate.set_ylabel("Observed default rate (%)")
    ax_rate.set_title("Realised risk by band")
    ax_rate.set_ylim(0, frame["default_rate"].max() * 1.22)

    bars = ax_share.bar(frame["band"], frame["population_share"], color=colors, width=0.55)
    label_bars(ax_share, bars, frame["population_share"].tolist(), fmt="{:.1f}%", pad=0.03)
    ax_share.set_ylabel("Share of applicants (%)")
    ax_share.set_title("Population in each band")
    ax_share.set_ylim(0, frame["population_share"].max() * 1.22)

    fig.suptitle(
        "Risk bands separate the book by realised default rate",
        x=0.02, ha="left", fontsize=12, fontweight="600",
    )
    fig.tight_layout()
    return save_figure(fig, "12_risk_bands")


def main() -> dict[str, Any]:  # pragma: no cover - CLI entry point
    """Command-line entry point: print the evaluation report."""
    report = evaluate()

    print("\n" + "=" * 80)
    print(f"EVALUATION  --  {report['model']}  (out-of-fold, {settings.cv_folds}-fold)")
    print("=" * 80)
    ranking = report["ranking"]
    print(f"  PR-AUC          {ranking['pr_auc']:.4f}   "
          f"({ranking['pr_auc_lift_over_random']:.1f}x the no-skill baseline of "
          f"{ranking['pr_auc_baseline']:.4f})")
    print(f"  ROC-AUC         {ranking['roc_auc']:.4f}")
    calibration = report["calibration"]
    print(f"  Brier           {calibration['brier']:.4f}")
    print(f"  mean predicted  {calibration['mean_predicted']:.4f} vs observed "
          f"{calibration['observed_rate']:.4f}")

    print("\n  At the TUNED threshold vs the naive 0.5 cut-off:")
    header = f"    {'':22s} {'tuned':>10s} {'0.5':>10s}"
    print(header)
    tuned, naive = report["at_tuned_threshold"], report["at_default_threshold"]
    for key in ("threshold", "precision", "recall", "f1", "flagged_share"):
        print(f"    {key:22s} {tuned[key]:>10.4f} {naive[key]:>10.4f}")
    print(f"    {'defaults caught':22s} "
          f"{tuned['confusion']['true_positive_default_caught']:>10d} "
          f"{naive['confusion']['true_positive_default_caught']:>10d}")
    print(f"    {'defaults missed':22s} "
          f"{tuned['confusion']['false_negative_default_missed']:>10d} "
          f"{naive['confusion']['false_negative_default_missed']:>10d}")

    print("\n  Risk bands:")
    print(pd.DataFrame(report["band_profile"]).to_string(index=False))
    print("\n  Bake-off:")
    print(pd.DataFrame(report["bakeoff"])[
        ["name", "pr_auc", "pr_auc_stderr", "roc_auc", "brier", "fit_seconds", "supports_shap"]
    ].to_string(index=False))
    return report


if __name__ == "__main__":  # pragma: no cover
    main()
