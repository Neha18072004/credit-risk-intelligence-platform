"""Model bake-off, calibration, threshold tuning and artifact persistence.

Three candidates are trained on identical features under identical cross
validation, and the winner is selected on PR-AUC first:

* **Logistic regression** -- the interpretable baseline every credit model is
  measured against, and the sanity check that the tree models are earning their
  complexity.
* **LightGBM** -- fast on CPU, handles NaN and categoricals natively, and pairs
  with SHAP's exact TreeExplainer.
* **CatBoost** -- ordered boosting with strong native categorical handling.

"Identical features" means the same feature *set*, encoded appropriately per
model family: the trees consume NaN and ``category`` dtype directly, while the
linear baseline gets median imputation, one-hot encoding and standardisation
inside its own pipeline, fitted separately on every fold so nothing leaks
across the split.

Run it with::

    python -m src.ml.train
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.loader import build_dataset, split_features_target
from src.data.preprocessor import CreditPreprocessor
from src.utils.config import settings
from src.utils.helpers import write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Artifact filenames. Collected here so predict.py and the UI never guess.
MODEL_FILE: Final[str] = "model.joblib"
PREPROCESSOR_FILE: Final[str] = "preprocessor.joblib"
CALIBRATOR_FILE: Final[str] = "calibrator.joblib"
METADATA_FILE: Final[str] = "feature_metadata.json"
THRESHOLDS_FILE: Final[str] = "thresholds.json"
METRICS_FILE: Final[str] = "metrics.json"
BAKEOFF_FILE: Final[str] = "bakeoff.csv"

# Rounds without improvement before a booster stops. Applied identically to
# both boosted candidates so the comparison stays like-for-like.
EARLY_STOPPING_PATIENCE: Final[int] = 80


@dataclass
class CandidateResult:
    """Cross-validated performance of one bake-off candidate."""

    name: str
    pr_auc: float
    roc_auc: float
    brier: float
    log_loss: float
    fit_seconds: float
    supports_shap: bool
    notes: str = ""
    best_iteration: int | None = None
    pr_auc_std: float = 0.0
    pr_auc_folds: list[float] = field(default_factory=list)
    oof_predictions: np.ndarray | None = field(default=None, repr=False)

    @property
    def pr_auc_stderr(self) -> float:
        """Standard error of the fold-level PR-AUC mean.

        With only a few hundred defaults the fold-to-fold spread is wide, and a
        point-estimate gap smaller than this is not evidence that one model
        beats another.
        """
        if len(self.pr_auc_folds) < 2:
            return 0.0
        return float(np.std(self.pr_auc_folds, ddof=1) / np.sqrt(len(self.pr_auc_folds)))

    def to_row(self) -> dict[str, Any]:
        """Row for the comparison table (drops the bulky OOF array)."""
        row = {
            k: v for k, v in asdict(self).items()
            if k not in {"oof_predictions", "pr_auc_folds"}
        }
        for key in ("pr_auc", "roc_auc", "brier", "log_loss", "pr_auc_std"):
            row[key] = round(float(row[key]), 4)
        row["pr_auc_stderr"] = round(self.pr_auc_stderr, 4)
        row["fit_seconds"] = round(float(row["fit_seconds"]), 1)
        return row


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_logistic_pipeline(
    numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    """Assemble the interpretable linear baseline.

    Unlike the tree models this one cannot consume NaN or raw categories, so it
    carries its own preprocessing. Keeping that inside the pipeline means it is
    re-fitted on every CV fold, so imputation statistics never leak from the
    validation fold into training.

    Args:
        numeric_features: Numeric column names.
        categorical_features: Categorical column names.

    Returns:
        An unfitted scikit-learn pipeline.
    """
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False)

    return Pipeline(
        [
            (
                "encode",
                ColumnTransformer(
                    [
                        ("numeric", numeric, numeric_features),
                        ("categorical", categorical, categorical_features),
                    ],
                    remainder="drop",
                ),
            ),
            (
                "model",
                LogisticRegression(
                    # 'balanced' is the linear analogue of scale_pos_weight: it
                    # reweights the loss rather than resampling the data.
                    class_weight="balanced",
                    max_iter=2000,
                    C=0.1,  # mild regularisation; the one-hot matrix is wide
                    solver="lbfgs",
                    random_state=settings.random_seed,
                ),
            ),
        ]
    )


def build_lightgbm(scale_pos_weight: float) -> Any:
    """Construct the LightGBM candidate.

    Args:
        scale_pos_weight: Negative-to-positive class ratio, applied as a loss
            weight so the rare class is not ignored.

    Returns:
        An unfitted ``LGBMClassifier``.
    """
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=24,
        max_depth=6,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        # Replace LightGBM's default metric list. Left at the default, it also
        # tracks binary_logloss, and early stopping halts when *any* tracked
        # metric stalls -- under scale_pos_weight the unweighted logloss on a
        # held-out slice degrades from the very first rounds, so training
        # stopped at iteration 5 and the model was left badly underfitted.
        metric="average_precision",
        random_state=settings.random_seed,
        n_jobs=-1,
        verbose=-1,
    )


def build_catboost(scale_pos_weight: float, categorical_features: list[str]) -> Any:
    """Construct the CatBoost candidate.

    Args:
        scale_pos_weight: Negative-to-positive class ratio.
        categorical_features: Column names CatBoost should treat as categorical.

    Returns:
        An unfitted ``CatBoostClassifier``.
    """
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=5,
        l2_leaf_reg=3.0,
        scale_pos_weight=scale_pos_weight,
        cat_features=categorical_features,
        eval_metric="PRAUC",
        random_seed=settings.random_seed,
        verbose=0,
        allow_writing_files=False,
    )


def _prepare_for_model(frame: pd.DataFrame, model_name: str, categorical: list[str]) -> pd.DataFrame:
    """Apply the per-family encoding a model expects.

    CatBoost requires categorical columns as strings with no NaN; LightGBM and
    the linear pipeline take the frame as produced by the preprocessor.
    """
    if model_name != "catboost":
        return frame
    adjusted = frame.copy()
    for column in categorical:
        adjusted[column] = adjusted[column].astype(str)
    return adjusted


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def cross_validate_candidate(
    name: str,
    features: pd.DataFrame,
    labels: pd.Series,
    preprocessor: CreditPreprocessor,
    scale_pos_weight: float,
) -> CandidateResult:
    """Run stratified k-fold CV for one candidate and collect OOF predictions.

    Out-of-fold predictions are the backbone of everything downstream: the
    bake-off comparison, the probability calibrator and the threshold tuning all
    read from them, so no model is ever evaluated or tuned on data it saw during
    training.

    Args:
        name: ``"logistic"``, ``"lightgbm"`` or ``"catboost"``.
        features: Model-ready feature frame.
        labels: Binary target.
        preprocessor: The fitted preprocessor, for its column metadata.
        scale_pos_weight: Class-imbalance weight.

    Returns:
        The candidate's cross-validated result, including OOF probabilities.
    """
    splitter = StratifiedKFold(
        n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_seed
    )
    oof = np.zeros(len(features), dtype=float)
    best_iterations: list[int] = []
    fold_scores: list[float] = []
    started = time.perf_counter()

    prepared = _prepare_for_model(features, name, preprocessor.categorical_features_)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(prepared, labels), start=1):
        X_train = prepared.iloc[train_idx]
        X_valid = prepared.iloc[valid_idx]
        y_train = labels.iloc[train_idx]

        model = _instantiate(name, preprocessor, scale_pos_weight)
        rounds = _fit_with_early_stopping(model, name, X_train, y_train)
        if rounds:
            best_iterations.append(rounds)
        fold_predictions = model.predict_proba(X_valid)[:, 1]
        oof[valid_idx] = fold_predictions
        fold_scores.append(float(average_precision_score(labels.iloc[valid_idx], fold_predictions)))
        logger.debug(
            "%s fold %d/%d PR-AUC %.4f (best_iter=%s)",
            name, fold, settings.cv_folds, fold_scores[-1], rounds,
        )

    elapsed = time.perf_counter() - started
    result = CandidateResult(
        name=name,
        pr_auc=float(average_precision_score(labels, oof)),
        roc_auc=float(roc_auc_score(labels, oof)),
        brier=float(brier_score_loss(labels, oof)),
        log_loss=float(log_loss(labels, oof)),
        fit_seconds=elapsed,
        supports_shap=name in {"lightgbm", "catboost"},
        notes=_CANDIDATE_NOTES[name],
        best_iteration=int(np.mean(best_iterations)) if best_iterations else None,
        pr_auc_std=float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
        pr_auc_folds=fold_scores,
        oof_predictions=oof,
    )
    logger.info(
        "%-9s PR-AUC %.4f (+/-%.4f across folds) | ROC-AUC %.4f | Brier %.4f | "
        "best_iter %s | %.1fs",
        name, result.pr_auc, result.pr_auc_std, result.roc_auc, result.brier,
        result.best_iteration, elapsed,
    )
    return result


def _fit_with_early_stopping(
    model: Any, name: str, X_train: pd.DataFrame, y_train: pd.Series
) -> int | None:
    """Fit a candidate, using early stopping for the boosted models.

    Without it, both boosters run a fixed 400 rounds on roughly 3,200 rows and
    overfit badly, which would make the bake-off a comparison of overfitting
    rather than of model families. A stratified 20% slice of the fold's own
    training data serves as the stopping set, so the outer validation fold stays
    untouched and out-of-fold predictions remain honest.

    Both boosters stop on **average precision** -- the same metric the bake-off
    selects on. Stopping on AUC instead was measurably worse (it cost CatBoost
    ~0.04 PR-AUC), which is the expected result of optimising one objective and
    halting on another. The stopping slice is 20% with a patience of 80 rounds,
    both raised from smaller values because a slice holding only ~50 defaults
    made the stopping point erratic across folds.

    ``first_metric_only`` matters here: LightGBM tracks its objective's default
    metric in addition to the requested one, and stops when *any* of them
    stalls. That silently capped LightGBM at 5 boosting rounds and cost it
    0.07 PR-AUC -- a bake-off result that would have been an artefact of the
    stopping protocol rather than a property of the model.

    Args:
        model: The unfitted candidate.
        name: Candidate name.
        X_train: Fold training features.
        y_train: Fold training labels.

    Returns:
        The chosen number of boosting rounds, or None for the linear model.
    """
    if name == "logistic":
        model.fit(X_train, y_train)
        return None

    from sklearn.model_selection import train_test_split

    X_fit, X_stop, y_fit, y_stop = train_test_split(
        X_train, y_train,
        test_size=0.20, stratify=y_train, random_state=settings.random_seed,
    )

    if name == "lightgbm":
        from lightgbm import early_stopping, log_evaluation

        model.fit(
            X_fit, y_fit,
            eval_set=[(X_stop, y_stop)],
            eval_metric="average_precision",
            callbacks=[
                early_stopping(EARLY_STOPPING_PATIENCE, verbose=False, first_metric_only=True),
                log_evaluation(0),
            ],
        )
        return int(getattr(model, "best_iteration_", 0) or model.n_estimators)

    model.fit(
        X_fit, y_fit,
        eval_set=(X_stop, y_stop),
        early_stopping_rounds=EARLY_STOPPING_PATIENCE,
        verbose=False,
    )
    return int(model.get_best_iteration() or model.tree_count_)


_CANDIDATE_NOTES: Final[dict[str, str]] = {
    "logistic": "Interpretable baseline; needs imputation, one-hot and scaling.",
    "lightgbm": "Native NaN and categorical handling; exact SHAP TreeExplainer; fastest.",
    "catboost": "Ordered boosting, strong categorical handling; slower to fit.",
}


def _instantiate(name: str, preprocessor: CreditPreprocessor, scale_pos_weight: float) -> Any:
    """Build a fresh, unfitted candidate by name."""
    if name == "logistic":
        return build_logistic_pipeline(
            preprocessor.numeric_features_, preprocessor.categorical_features_
        )
    if name == "lightgbm":
        return build_lightgbm(scale_pos_weight)
    if name == "catboost":
        return build_catboost(scale_pos_weight, preprocessor.categorical_features_)
    raise ValueError(f"Unknown candidate {name!r}")


# --------------------------------------------------------------------------- #
# Selection, calibration and thresholds
# --------------------------------------------------------------------------- #
def select_winner(results: list[CandidateResult]) -> CandidateResult:
    """Pick the model to ship.

    PR-AUC is the primary criterion -- under a ~9% positive rate it is the
    honest measure of ranking quality, where ROC-AUC is flattered by the large
    negative class.

    But a point estimate alone is not a decision. With a few hundred defaults
    the fold-to-fold spread in PR-AUC is wide, so any candidate whose score is
    within one standard error of the leader is treated as **statistically
    tied**, and the tie is broken on the brief's stated secondary criteria:
    exact tree-SHAP support first, then fit cost. Shipping a marginally higher
    point estimate that cannot be explained per-applicant would be the wrong
    trade for a credit model that has to justify every decision.

    Args:
        results: All cross-validated candidates.

    Returns:
        The selected candidate.
    """
    ranked = sorted(results, key=lambda r: r.pr_auc, reverse=True)
    leader = ranked[0]

    # A candidate is a genuine contender if the leader's advantage over it is no
    # larger than the sampling error on that difference.
    contenders = [
        candidate for candidate in ranked
        if leader.pr_auc - candidate.pr_auc
        <= max(candidate.pr_auc_stderr, leader.pr_auc_stderr)
    ]

    best = sorted(contenders, key=lambda r: (not r.supports_shap, r.fit_seconds))[0]

    if best is not leader:
        logger.info(
            "%s leads on PR-AUC (%.4f) but %s is within one standard error "
            "(%.4f, +/-%.4f); selecting %s for exact tree-SHAP support",
            leader.name, leader.pr_auc, best.name, best.pr_auc,
            best.pr_auc_stderr, best.name,
        )
    logger.info(
        "Selected model: %s (PR-AUC %.4f +/- %.4f across %d folds)",
        best.name, best.pr_auc, best.pr_auc_std, len(best.pr_auc_folds),
    )
    return best


def fit_calibrator(oof_predictions: np.ndarray, labels: pd.Series) -> Any:
    """Fit a probability calibrator on out-of-fold predictions.

    A gradient-boosted model trained with ``scale_pos_weight`` produces well
    *ranked* but badly *scaled* scores -- the weighting deliberately inflates
    them away from the true class prior. Since the product here is a risk
    probability, not just a ranking, that has to be corrected.

    Calibrating on out-of-fold predictions uses every row exactly once without
    a nested CV loop, and the calibrator never sees a prediction the model made
    on its own training data.

    Args:
        oof_predictions: Out-of-fold probabilities from the winning model.
        labels: True binary outcomes.

    Returns:
        A fitted calibrator exposing ``predict``.
    """
    if settings.calibration_method == "isotonic":
        # out_of_bounds="clip" keeps inference safe when a raw score falls
        # outside the range seen during calibration.
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(oof_predictions, labels.to_numpy())
    else:
        platt = LogisticRegression(solver="lbfgs")
        platt.fit(oof_predictions.reshape(-1, 1), labels.to_numpy())

        class _PlattCalibrator:
            """Thin adapter giving Platt scaling the same ``predict`` surface."""

            def __init__(self, model: LogisticRegression) -> None:
                self.model = model

            def predict(self, scores: np.ndarray) -> np.ndarray:
                return self.model.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]

        calibrator = _PlattCalibrator(platt)

    logger.info("Fitted %s calibrator on out-of-fold predictions", settings.calibration_method)
    return calibrator


def tune_thresholds(
    calibrated: np.ndarray, labels: pd.Series
) -> dict[str, Any]:
    """Derive decision and risk-band thresholds from out-of-fold predictions.

    Nothing here uses 0.5. At a 9% base rate a 0.5 cut-off approves essentially
    every applicant, so both the decision point and the band edges are tuned:

    * **Decision threshold** -- the probability maximising F1 on out-of-fold
      predictions. F2 is reported alongside it, because in lending a missed
      default usually costs more than a declined good applicant, and a reviewer
      may prefer the recall-weighted point.
    * **Low band edge** -- the portfolio base rate. An applicant is "Low risk"
      when their predicted probability is no worse than the average applicant
      in the book, which is a statement a credit committee can actually reason
      about.

      An earlier version set this edge wherever the *average* default rate of
      everyone below it met a 5% target. That was wrong in a way worth
      recording: averaging over a wide band hid its own upper end, and put
      applicants with a 13-16% individual default probability in the band
      labelled "Low". Band edges have to bound the marginal applicant, not the
      mean of the group.
    * **High band edge** -- the tuned decision threshold, so the High band is
      exactly the population the model would action.

    Args:
        calibrated: Calibrated out-of-fold probabilities.
        labels: True binary outcomes.

    Returns:
        Threshold values plus the realised default rate and population share of
        each band, so the bands can be sanity-checked before they are trusted.
    """
    y = labels.to_numpy()
    precision, recall, cuts = precision_recall_curve(y, calibrated)
    # precision_recall_curve returns one more point than thresholds.
    precision, recall = precision[:-1], recall[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
        f2 = np.nan_to_num(5 * precision * recall / (4 * precision + recall))

    best_f1_index = int(np.argmax(f1))
    decision_threshold = float(cuts[best_f1_index])
    best_f2_index = int(np.argmax(f2))

    # Low band edge: the portfolio base rate, bounded by the decision threshold
    # so the bands can never invert on an unusual sample.
    base_rate = float(y.mean())
    low_edge = float(min(base_rate, decision_threshold * 0.9))
    high_edge = max(decision_threshold, low_edge + 1e-6)

    bands = _band_profile(calibrated, y, low_edge, high_edge)
    thresholds = {
        "decision_threshold": round(decision_threshold, 6),
        "decision_precision": round(float(precision[best_f1_index]), 4),
        "decision_recall": round(float(recall[best_f1_index]), 4),
        "decision_f1": round(float(f1[best_f1_index]), 4),
        "f2_optimal_threshold": round(float(cuts[best_f2_index]), 6),
        "f2_optimal_recall": round(float(recall[best_f2_index]), 4),
        "band_low_max": round(low_edge, 6),
        "band_medium_max": round(high_edge, 6),
        "band_definition": (
            "Low: predicted default probability at or below the portfolio base "
            "rate. Medium: above the base rate but below the tuned decision "
            "threshold. High: at or above the tuned decision threshold, i.e. "
            "the population the model refers for review."
        ),
        "band_profile": bands,
        "calibration_method": settings.calibration_method,
        "risk_score_scale": settings.risk_score_scale,
    }
    logger.info(
        "Tuned thresholds: decision=%.4f (P=%.3f R=%.3f F1=%.3f) | bands low<=%.4f high>%.4f",
        decision_threshold, precision[best_f1_index], recall[best_f1_index],
        f1[best_f1_index], low_edge, high_edge,
    )
    return thresholds


def _band_profile(
    calibrated: np.ndarray, y: np.ndarray, low_edge: float, high_edge: float
) -> list[dict[str, Any]]:
    """Realised default rate and population share within each risk band."""
    assignments = np.where(
        calibrated <= low_edge, "Low", np.where(calibrated <= high_edge, "Medium", "High")
    )
    profile: list[dict[str, Any]] = []
    for band in ("Low", "Medium", "High"):
        mask = assignments == band
        count = int(mask.sum())
        profile.append(
            {
                "band": band,
                "n": count,
                "population_share": round(100 * count / max(len(y), 1), 2),
                "default_rate": round(100 * float(y[mask].mean()), 2) if count else 0.0,
                "share_of_all_defaults": (
                    round(100 * float(y[mask].sum()) / max(int(y.sum()), 1), 2) if count else 0.0
                ),
            }
        )
    return profile


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _set_n_estimators(model: Any, name: str, rounds: int) -> None:
    """Pin the boosting-round count on the final, full-data refit."""
    rounds = max(int(rounds), 10)
    if name == "lightgbm":
        model.set_params(n_estimators=rounds)
    elif name == "catboost":
        model.set_params(iterations=rounds)


def train(save: bool = True, candidates: list[str] | None = None) -> dict[str, Any]:
    """Run the full training pipeline and persist every artifact.

    Steps: load and join, fit the preprocessor, cross-validate all candidates,
    select on PR-AUC, refit the winner on the full training split, calibrate on
    out-of-fold predictions, tune thresholds, and save.

    Args:
        save: Write artifacts to ``models/``. Set False in tests.
        candidates: Restrict the bake-off; defaults to all three.

    Returns:
        A summary dictionary containing the comparison table, the selected
        model name, out-of-fold metrics and the tuned thresholds.
    """
    settings.ensure_directories()
    names = candidates or ["logistic", "lightgbm", "catboost"]

    # --- data -------------------------------------------------------------
    raw = build_dataset("train")
    _, labels = split_features_target(raw)
    if labels is None:
        raise ValueError("Training data has no TARGET column.")

    preprocessor = CreditPreprocessor().fit(raw)
    features = preprocessor.transform(raw)

    positives = int(labels.sum())
    scale_pos_weight = float((len(labels) - positives) / max(positives, 1))
    logger.info(
        "Training on %s rows x %s features | %s defaults (%.2f%%) | scale_pos_weight %.2f",
        f"{len(features):,}", features.shape[1], f"{positives:,}",
        100 * labels.mean(), scale_pos_weight,
    )

    # --- bake-off ---------------------------------------------------------
    results = [
        cross_validate_candidate(name, features, labels, preprocessor, scale_pos_weight)
        for name in names
    ]
    comparison = pd.DataFrame([r.to_row() for r in results]).sort_values(
        "pr_auc", ascending=False
    )
    winner = select_winner(results)

    # --- final fit, calibration, thresholds -------------------------------
    final_model = _instantiate(winner.name, preprocessor, scale_pos_weight)
    if winner.best_iteration:
        # Refit on all the data for the number of rounds cross-validation showed
        # to be optimal, rather than holding data back for a stopping set.
        _set_n_estimators(final_model, winner.name, winner.best_iteration)
    prepared = _prepare_for_model(features, winner.name, preprocessor.categorical_features_)
    final_model.fit(prepared, labels)

    calibrator = fit_calibrator(winner.oof_predictions, labels)
    calibrated_oof = np.clip(calibrator.predict(winner.oof_predictions), 0.0, 1.0)
    thresholds = tune_thresholds(calibrated_oof, labels)

    metrics = {
        "selected_model": winner.name,
        "selection_criterion": (
            "PR-AUC first; candidates within one standard error of the leader are "
            "treated as tied and decided on exact tree-SHAP support, then fit cost"
        ),
        "n_rows": int(len(features)),
        "n_features": int(features.shape[1]),
        "n_defaults": positives,
        "default_rate": round(float(labels.mean()), 4),
        "scale_pos_weight": round(scale_pos_weight, 3),
        "cv_folds": settings.cv_folds,
        "oof_uncalibrated": {
            "pr_auc": round(winner.pr_auc, 4),
            "roc_auc": round(winner.roc_auc, 4),
            "brier": round(winner.brier, 4),
        },
        "oof_calibrated": {
            "pr_auc": round(float(average_precision_score(labels, calibrated_oof)), 4),
            "roc_auc": round(float(roc_auc_score(labels, calibrated_oof)), 4),
            "brier": round(float(brier_score_loss(labels, calibrated_oof)), 4),
            "mean_predicted": round(float(calibrated_oof.mean()), 4),
            "observed_rate": round(float(labels.mean()), 4),
        },
        "bakeoff": comparison.to_dict(orient="records"),
        "thresholds": thresholds,
    }

    metadata = {
        "feature_names": preprocessor.feature_names_,
        "numeric_features": preprocessor.numeric_features_,
        "categorical_features": preprocessor.categorical_features_,
        "categorical_indices": preprocessor.categorical_indices_,
        "model_name": winner.name,
        "requires_string_categoricals": winner.name == "catboost",
    }

    if save:
        directory = settings.models_dir
        joblib.dump(final_model, directory / MODEL_FILE)
        joblib.dump(preprocessor, directory / PREPROCESSOR_FILE)
        joblib.dump(calibrator, directory / CALIBRATOR_FILE)
        write_json(metadata, directory / METADATA_FILE)
        write_json(thresholds, directory / THRESHOLDS_FILE)
        write_json(metrics, directory / METRICS_FILE)
        comparison.to_csv(directory / BAKEOFF_FILE, index=False)
        # OOF predictions power the evaluation report without a retrain.
        np.save(directory / "oof_calibrated.npy", calibrated_oof)
        np.save(directory / "oof_labels.npy", labels.to_numpy())
        logger.info("Artifacts written to %s", directory)

    return {
        "comparison": comparison,
        "winner": winner.name,
        "metrics": metrics,
        "thresholds": thresholds,
        "model": final_model,
        "preprocessor": preprocessor,
        "calibrator": calibrator,
        "oof_calibrated": calibrated_oof,
        "labels": labels,
    }


def main() -> dict[str, Any]:  # pragma: no cover - CLI entry point
    """Command-line entry point: run the bake-off and print the results."""
    outcome = train()
    comparison = outcome["comparison"]
    metrics = outcome["metrics"]

    print("\n" + "=" * 84)
    print("MODEL BAKE-OFF  (stratified 5-fold, out-of-fold predictions)")
    print("=" * 84)
    print(
        comparison[
            ["name", "pr_auc", "pr_auc_stderr", "roc_auc", "brier",
             "log_loss", "fit_seconds", "supports_shap"]
        ].to_string(index=False)
    )
    print(f"\n  Selected: {outcome['winner']}  ({metrics['selection_criterion']})")
    for result in comparison.itertuples():
        print(f"    - {result.name:9s} {result.notes}")

    print("\n" + "=" * 84)
    print("CALIBRATION")
    print("=" * 84)
    before, after = metrics["oof_uncalibrated"], metrics["oof_calibrated"]
    print(f"  Brier   {before['brier']:.4f} -> {after['brier']:.4f}")
    print(f"  PR-AUC  {before['pr_auc']:.4f} -> {after['pr_auc']:.4f}")
    print(f"  mean predicted {after['mean_predicted']:.4f} vs observed {after['observed_rate']:.4f}")

    print("\n" + "=" * 84)
    print("TUNED THRESHOLDS AND RISK BANDS")
    print("=" * 84)
    thresholds = outcome["thresholds"]
    print(
        f"  decision threshold {thresholds['decision_threshold']:.4f}  "
        f"(precision {thresholds['decision_precision']:.3f}, "
        f"recall {thresholds['decision_recall']:.3f}, F1 {thresholds['decision_f1']:.3f})"
    )
    print(f"  band edges: Low <= {thresholds['band_low_max']:.4f} "
          f"< Medium <= {thresholds['band_medium_max']:.4f} < High\n")
    print(pd.DataFrame(thresholds["band_profile"]).to_string(index=False))
    return outcome


if __name__ == "__main__":  # pragma: no cover
    main()
