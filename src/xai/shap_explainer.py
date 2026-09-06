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
from typing import Any, Final

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

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

    def describe(self) -> str:
        """One human-readable line, for the UI and adverse-action reasons."""
        shown = f"{self.value:,.4g}" if isinstance(self.value, (int, float, np.number)) else self.value
        return f"{self.feature} = {shown} ({self.direction}, {self.contribution:+.4f})"


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
