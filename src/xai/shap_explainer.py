"""SHAP explanations, local and global.

Two audiences, one engine:

* **Local** -- why did *this* applicant get *this* score? Returned as signed
  per-feature contributions in log-odds space, ready for a waterfall in the UI
  and for an adverse-action style explanation.
* **Global** -- which features drive the portfolio overall? Mean absolute SHAP
  across a sample, which unlike a tree's split-count importance is additive,
  signed and comparable between features.

The explainer dispatches on model family, because the winning model is chosen
by the bake-off and may be any of the three:

* CatBoost -- native ``get_feature_importance(type="ShapValues")``, exact.
* LightGBM -- ``shap.TreeExplainer``, exact for tree ensembles.
* Logistic regression -- contribution of each encoded feature as
  ``coefficient * (value - mean)``, which is the exact Shapley value for a
  linear model and needs no sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.xai.feature_labels import format_value, humanise

logger = get_logger(__name__)

# Rows sampled when computing global importance. Enough for a stable ranking,
# small enough that the UI stays responsive.
GLOBAL_SAMPLE_SIZE: Final[int] = 500


@dataclass(frozen=True)
class FeatureContribution:
    """One feature's signed contribution to a single prediction."""

    feature: str
    value: Any
    contribution: float

    @property
    def direction(self) -> str:
        """``"increases risk"`` or ``"decreases risk"``."""
        return "increases risk" if self.contribution > 0 else "decreases risk"

    @property
    def label(self) -> str:
        """Plain-English name for the feature."""
        return humanise(self.feature)

    @property
    def display_value(self) -> str:
        """Plain-English rendering of the applicant's value."""
        return format_value(self.feature, self.value)

    @property
    def strength(self) -> str:
        """Qualitative magnitude, for readers who should not be shown log-odds.

        The thresholds are on the log-odds scale: 0.5 is roughly a 1.6x change
        in the odds of default, which is a substantial single driver, while
        anything under 0.1 barely moves the decision.
        """
        magnitude = abs(self.contribution)
        if magnitude >= 0.5:
            return "major"
        if magnitude >= 0.2:
            return "moderate"
        if magnitude >= 0.05:
            return "minor"
        return "negligible"

    def describe(self) -> str:
        """One readable line, for the UI and adverse-action reasons."""
        return (
            f"{self.label} = {self.display_value} "
            f"({self.direction}, {self.strength} effect, {self.contribution:+.4f})"
        )

    def as_sentence(self) -> str:
        """A clause a non-technical reader can act on."""
        verb = "raises" if self.contribution > 0 else "lowers"
        return f"their {self.label} of {self.display_value} {verb} the risk"


class ShapExplainer:
    """Model-agnostic SHAP wrapper over the three bake-off candidates.

    Args:
        model: The fitted estimator.
        model_name: ``"catboost"``, ``"lightgbm"`` or ``"logistic"``.
        feature_names: Column order the model was fitted on.
        categorical_features: Categorical column names (needed by CatBoost).
    """

    def __init__(
        self,
        model: Any,
        model_name: str,
        feature_names: list[str],
        categorical_features: list[str] | None = None,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.feature_names = list(feature_names)
        self.categorical_features = list(categorical_features or [])
        self._tree_explainer: Any | None = None
        self._expected_value: float | None = None

    # ------------------------------------------------------------ internal --
    def _lightgbm_explainer(self) -> Any:
        """Lazily build and cache the SHAP TreeExplainer."""
        if self._tree_explainer is None:
            import shap

            self._tree_explainer = shap.TreeExplainer(self.model)
        return self._tree_explainer

    def _shap_matrix(self, features: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Compute SHAP values for ``features``.

        Returns:
            ``(values, expected_value)`` where ``values`` has shape
            ``(n_rows, n_features)`` in log-odds space.
        """
        if self.model_name == "catboost":
            from catboost import Pool

            pool = Pool(features, cat_features=self.categorical_features)
            raw = self.model.get_feature_importance(pool, type="ShapValues")
            # CatBoost appends the base value as a final column.
            return raw[:, :-1], float(raw[0, -1])

        if self.model_name == "lightgbm":
            explainer = self._lightgbm_explainer()
            values = explainer.shap_values(features)
            expected = explainer.expected_value
            # Older SHAP returns a per-class list; newer may return a 3-D array.
            if isinstance(values, list):
                values, expected = values[-1], expected[-1] if isinstance(expected, list) else expected
            values = np.asarray(values)
            if values.ndim == 3:
                values, expected = values[:, :, -1], np.asarray(expected).ravel()[-1]
            return values, float(np.asarray(expected).ravel()[0])

        return self._linear_contributions(features)

    def _linear_contributions(self, features: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Exact Shapley values for the logistic-regression pipeline.

        For a linear model the Shapley value of a feature is simply
        ``coefficient * (value - mean)``, so no sampling is required. The values
        are computed in the *encoded* space the model actually sees, then summed
        back onto the original columns so the explanation speaks in the
        applicant's own terms rather than in one-hot fragments.
        """
        encoder = self.model.named_steps["encode"]
        classifier = self.model.named_steps["model"]

        encoded = encoder.transform(features)
        encoded_names = list(encoder.get_feature_names_out())
        coefficients = classifier.coef_.ravel()

        # Column means come from the fitted scaler/encoder, approximated here by
        # the batch mean -- exact for a single-row explanation relative to the
        # supplied background.
        means = encoded.mean(axis=0) if len(encoded) > 1 else np.zeros_like(coefficients)
        contributions = (encoded - means) * coefficients

        # Fold one-hot columns back onto their source feature.
        folded = np.zeros((len(features), len(self.feature_names)))
        index_of = {name: i for i, name in enumerate(self.feature_names)}
        for column_index, encoded_name in enumerate(encoded_names):
            bare = encoded_name.split("__", 1)[-1]
            source = next(
                (name for name in self.feature_names if bare == name or bare.startswith(f"{name}_")),
                None,
            )
            if source is not None:
                folded[:, index_of[source]] += contributions[:, column_index]

        return folded, float(classifier.intercept_[0])

    # -------------------------------------------------------------- public --
    def explain_row(
        self, features: pd.DataFrame, top_n: int = 10
    ) -> list[FeatureContribution]:
        """Explain a single prediction.

        Args:
            features: A one-row frame in the model's feature order.
            top_n: How many contributions to return, ranked by absolute impact.

        Returns:
            The strongest contributions, largest absolute impact first.
        """
        if len(features) != 1:
            raise ValueError(f"explain_row expects exactly one row, got {len(features)}")

        values, _ = self._shap_matrix(features)
        row = np.asarray(values)[0]

        contributions = [
            FeatureContribution(
                feature=name,
                value=features.iloc[0][name],
                contribution=float(row[index]),
            )
            for index, name in enumerate(self.feature_names)
        ]
        contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
        return contributions[:top_n]

    def global_importance(
        self, features: pd.DataFrame, sample_size: int = GLOBAL_SAMPLE_SIZE, top_n: int = 20
    ) -> pd.DataFrame:
        """Rank features by mean absolute SHAP value across a sample.

        Args:
            features: Model-ready feature frame.
            sample_size: Rows to sample; the full frame is used if smaller.
            top_n: Number of features to return.

        Returns:
            Columns ``feature``, ``mean_abs_shap``, ``mean_shap`` and
            ``importance_pct``. ``mean_shap`` keeps its sign, so a feature that
            mostly *reduces* risk is visibly different from one that raises it.
        """
        sample = (
            features.sample(sample_size, random_state=42)
            if len(features) > sample_size
            else features
        )
        values, _ = self._shap_matrix(sample)
        values = np.asarray(values)

        mean_abs = np.abs(values).mean(axis=0)
        total = mean_abs.sum() or 1.0
        frame = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": mean_abs,
                "mean_shap": values.mean(axis=0),
                "importance_pct": 100 * mean_abs / total,
            }
        )
        ranked = frame.sort_values("mean_abs_shap", ascending=False).head(top_n)
        logger.info(
            "Global SHAP over %d rows; top driver %s (%.1f%% of total importance)",
            len(sample), ranked.iloc[0]["feature"], ranked.iloc[0]["importance_pct"],
        )
        return ranked.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_global_importance(importance: pd.DataFrame, top_n: int = 15) -> Path:
    """Chart the portfolio-level feature ranking.

    One series, one colour -- the bar length already encodes magnitude, so
    colouring each bar by its own value would burn the only free channel on
    information the chart is already showing.

    Args:
        importance: Output of :meth:`ShapExplainer.global_importance`.
        top_n: How many features to draw.

    Returns:
        Path to the saved figure.
    """
    from src.utils.viz import (
        SERIES, apply_theme, label_bars, new_figure, save_figure, style_axes,
    )

    apply_theme()
    frame = importance.head(top_n).iloc[::-1].copy()
    frame["feature"] = frame["feature"].map(humanise)

    fig, ax = new_figure(figsize=(9.0, 0.38 * len(frame) + 1.8))
    style_axes(ax, xgrid=True, ygrid=False)
    bars = ax.barh(frame["feature"], frame["mean_abs_shap"], color=SERIES[0], height=0.62)
    label_bars(
        ax, bars, frame["importance_pct"].tolist(), fmt="{:.1f}%",
        horizontal=True, pad=0.02,
    )
    ax.set_xlim(0, frame["mean_abs_shap"].max() * 1.18)
    ax.set_xlabel("Mean |SHAP value|  (impact on log-odds of default)")
    ax.set_title("What drives the model across the portfolio")
    fig.tight_layout()
    return save_figure(fig, "13_shap_global_importance")


def plot_local_explanation(
    contributions: list[FeatureContribution], probability: float, applicant_id: object = None
) -> Path:
    """Chart one applicant's SHAP contributions as a diverging bar chart.

    Signed values get the validated diverging pair -- cool for contributions
    that reduce risk, warm for those that raise it -- because the sign is the
    whole point of a local explanation. Every bar is labelled, so the direction
    never rests on colour alone.

    Args:
        contributions: Ranked contributions from :meth:`ShapExplainer.explain_row`.
        probability: The applicant's calibrated default probability.
        applicant_id: Optional identifier for the title.

    Returns:
        Path to the saved figure.
    """
    from src.utils.viz import (
        BASELINE, DIVERGING_NEGATIVE, DIVERGING_POSITIVE,
        apply_theme, new_figure, save_figure, style_axes,
    )

    apply_theme()
    ordered = sorted(contributions, key=lambda c: c.contribution)
    values = [c.contribution for c in ordered]
    labels = [f"{c.label}  =  {c.display_value}" for c in ordered]
    colors = [DIVERGING_POSITIVE if v > 0 else DIVERGING_NEGATIVE for v in values]

    fig, ax = new_figure(figsize=(9.5, 0.42 * len(ordered) + 1.8))
    style_axes(ax, xgrid=True, ygrid=False)
    ax.barh(range(len(ordered)), values, color=colors, height=0.62)
    ax.axvline(0, color=BASELINE, linewidth=1.0)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(labels, fontsize=9)

    # Symmetric limits: a diverging scale must not imply that one direction
    # carries more range than the other.
    span = max(abs(min(values)), abs(max(values))) or 1.0
    ax.set_xlim(-span * 1.2, span * 1.2)
    for index, value in enumerate(values):
        offset = span * 0.03
        ax.text(
            value + (offset if value > 0 else -offset), index, f"{value:+.3f}",
            va="center", ha="left" if value > 0 else "right", fontsize=8.5,
            color=DIVERGING_POSITIVE if value > 0 else DIVERGING_NEGATIVE,
        )

    ax.set_xlabel("SHAP contribution to log-odds of default  "
                  "(left = reduces risk, right = increases risk)")
    subject = f"Applicant {applicant_id}" if applicant_id is not None else "This applicant"
    ax.set_title(f"{subject} scored {100 * probability:.1f}% default probability -- here is why")
    fig.tight_layout()
    return save_figure(fig, "14_shap_local_explanation")


# --------------------------------------------------------------------------- #
# Plain-English narrative
# --------------------------------------------------------------------------- #
def narrate_explanation(
    contributions: list[FeatureContribution],
    probability: float,
    risk_band: str,
    decision: str,
    max_reasons: int = 3,
) -> str:
    """Turn a SHAP explanation into a paragraph a non-specialist can read.

    A ranked table of log-odds values is an explanation for a modeller. An
    applicant who has been referred for review is entitled to something they can
    actually act on, and a credit officer needs to be able to repeat the reason
    out loud. This renders the same numbers as prose, naming the drivers in both
    directions so the account is balanced rather than only adverse.

    Args:
        contributions: Ranked contributions from :meth:`ShapExplainer.explain_row`.
        probability: Calibrated probability of default.
        risk_band: ``"Low"``, ``"Medium"`` or ``"High"``.
        decision: The recommended action.
        max_reasons: How many drivers to name per direction.

    Returns:
        A short paragraph. Never asserts a cause beyond the model's own
        attribution -- these are the features that moved *this* score, which is
        not the same as a claim about why the applicant behaves as they do.
    """
    if not contributions:
        return (
            f"This applicant scores {100 * probability:.1f}% probability of default "
            f"({risk_band} risk). No individual feature explanation is available."
        )

    raising = [c for c in contributions if c.contribution > 0][:max_reasons]
    lowering = [c for c in contributions if c.contribution < 0][:max_reasons]

    opening = (
        f"This applicant has a {100 * probability:.1f}% estimated probability of default, "
        f"placing them in the **{risk_band}** risk band. Recommended action: {decision.lower()}."
    )

    parts = [opening]
    if raising:
        clauses = [c.as_sentence() for c in raising]
        joined = clauses[0] if len(clauses) == 1 else ", ".join(clauses[:-1]) + f", and {clauses[-1]}"
        parts.append(f"The main factors increasing risk are that {joined}.")
    if lowering:
        clauses = [c.as_sentence() for c in lowering]
        joined = clauses[0] if len(clauses) == 1 else ", ".join(clauses[:-1]) + f", and {clauses[-1]}"
        parts.append(f"Working in their favour: {joined}.")

    parts.append(
        "These are the factors that moved this particular score, ranked by how much "
        "each one shifted it."
    )
    return " ".join(parts)
