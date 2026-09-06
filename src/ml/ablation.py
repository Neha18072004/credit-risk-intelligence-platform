"""Measure what each auxiliary table actually contributes.

Adding a table is a cost as well as a benefit: more runtime, more memory, more
columns to explain, more ways for the pipeline to fail. So each block is
switched on in turn and the gain is measured, rather than assumed from the fact
that Kaggle solutions use it.

The comparison is deliberately like-for-like: one model family (LightGBM, the
fastest of the three), the same folds, the same early-stopping protocol, and the
same metric the bake-off selects on. The absolute numbers are lower than the
final tuned model's -- that is not the point. The differences between rows are.

Run it with::

    python -m src.ml.ablation
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data.loader import build_dataset, split_features_target
from src.data.preprocessor import CreditPreprocessor
from src.ml.train import EARLY_STOPPING_PATIENCE, build_lightgbm
from src.utils.config import settings
from src.utils.helpers import write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Each configuration adds one block to the one above it, so the difference
# between consecutive rows is that block's marginal contribution.
CONFIGURATIONS: list[tuple[str, dict[str, bool]]] = [
    ("application only", {"include_bureau": False, "include_previous": False,
                          "include_installments": False}),
    ("+ bureau", {"include_bureau": True, "include_previous": False,
                  "include_installments": False}),
    ("+ previous_application", {"include_bureau": True, "include_previous": True,
                                "include_installments": False}),
    ("+ installments_payments", {"include_bureau": True, "include_previous": True,
                                 "include_installments": True}),
]

# Three folds rather than five: this is a comparison between configurations, and
# halving the fit count matters on 307k rows. The final model still uses five.
ABLATION_FOLDS: int = 3


def evaluate_configuration(name: str, flags: dict[str, bool]) -> dict[str, Any]:
    """Cross-validate one table configuration.

    Args:
        name: Human-readable label for the configuration.
        flags: Keyword arguments for :func:`build_dataset`.

    Returns:
        Metrics plus the shape and cost of the configuration.
    """
    started = time.perf_counter()
    raw = build_dataset("train", **flags)
    _, labels = split_features_target(raw)
    if labels is None:
        raise ValueError("Training data has no TARGET column.")

    preprocessor = CreditPreprocessor().fit(raw)
    features = preprocessor.transform(raw)
    positives = int(labels.sum())
    scale_pos_weight = float((len(labels) - positives) / max(positives, 1))

    splitter = StratifiedKFold(
        n_splits=ABLATION_FOLDS, shuffle=True, random_state=settings.random_seed
    )
    oof = np.zeros(len(features), dtype=float)

    for train_idx, valid_idx in splitter.split(features, labels):
        X_train, y_train = features.iloc[train_idx], labels.iloc[train_idx]
        X_fit, X_stop, y_fit, y_stop = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train,
            random_state=settings.random_seed,
        )
        from lightgbm import early_stopping, log_evaluation

        model = build_lightgbm(scale_pos_weight)
        model.fit(
            X_fit, y_fit,
            eval_set=[(X_stop, y_stop)],
            eval_metric="average_precision",
            callbacks=[
                early_stopping(EARLY_STOPPING_PATIENCE, verbose=False, first_metric_only=True),
                log_evaluation(0),
            ],
        )
        oof[valid_idx] = model.predict_proba(features.iloc[valid_idx])[:, 1]

    elapsed = time.perf_counter() - started
    result = {
        "configuration": name,
        "n_features": int(features.shape[1]),
        "pr_auc": round(float(average_precision_score(labels, oof)), 4),
        "roc_auc": round(float(roc_auc_score(labels, oof)), 4),
        "seconds": round(elapsed, 1),
    }
    logger.info(
        "%-24s %d features | PR-AUC %.4f | ROC-AUC %.4f | %.0fs",
        name, result["n_features"], result["pr_auc"], result["roc_auc"], elapsed,
    )
    return result


def run_ablation(save: bool = True) -> pd.DataFrame:
    """Evaluate every configuration and report the marginal gain of each block.

    Args:
        save: Write the table to ``reports/``.

    Returns:
        The comparison, with per-block deltas.
    """
    rows = [evaluate_configuration(name, flags) for name, flags in CONFIGURATIONS]
    frame = pd.DataFrame(rows)

    # The marginal contribution: what this block added over the row above it.
    frame["pr_auc_gain"] = frame["pr_auc"].diff().round(4)
    frame["roc_auc_gain"] = frame["roc_auc"].diff().round(4)
    frame["features_added"] = frame["n_features"].diff().fillna(0).astype(int)

    if save:
        settings.ensure_directories()
        frame.to_csv(settings.reports_dir / "table_ablation.csv", index=False)
        write_json(rows, settings.reports_dir / "table_ablation.json")
        logger.info("Ablation written to %s", settings.reports_dir / "table_ablation.csv")
    return frame


def main() -> pd.DataFrame:  # pragma: no cover - CLI entry point
    """Command-line entry point."""
    frame = run_ablation()
    print("\n" + "=" * 92)
    print("TABLE ABLATION  (LightGBM, 3-fold, out-of-fold; differences are what matter)")
    print("=" * 92)
    print(frame.to_string(index=False))
    print(
        "\n  Each row adds one block to the row above it, so pr_auc_gain is that "
        "block's marginal contribution."
    )
    return frame


if __name__ == "__main__":  # pragma: no cover
    main()
