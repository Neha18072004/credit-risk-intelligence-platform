"""Surrogate decision tree and derived credit-policy rules.

SHAP explains *one applicant*. A credit committee also needs to see the policy
the model is implicitly applying across the whole book, in language that can be
written into a manual, argued with, and audited.

This module fits a deliberately shallow decision tree that mimics the trained
model's calibrated probabilities, then exports its leaves as IF/THEN rules. The
tree is not a second model competing with the first -- it is a readable
*approximation* of the first, which is why **fidelity is reported alongside
every rule set**. A surrogate that does not track the model would be worse than
no surrogate at all, because it would look authoritative while being wrong.

Together with :mod:`src.xai.shap_explainer` this gives one coherent
explainability story: SHAP for the individual decision, the surrogate tree for
the portfolio-level policy.

Run it with::

    python -m src.xai.rules
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, _tree

from src.utils.config import settings
from src.utils.helpers import bound_probability
from src.utils.logger import get_logger
from src.xai.feature_labels import humanise

logger = get_logger(__name__)

# Leaves covering fewer applicants than this are not worth writing into policy:
# the rule would be fitted to noise.
MIN_LEAF_FRACTION: Final[float] = 0.02

@dataclass(frozen=True)
class Condition:
    """One branch test along the path to a leaf."""

    feature: str
    operator: str
    threshold: float
    is_indicator: bool = False

    def describe(self) -> str:
        """Render the condition as a policy clause."""
        label = humanise(self.feature)
        if self.is_indicator:
            # A one-hot column split at 0.5 is really a yes/no membership test.
            return f"{label}" if self.operator == ">" else f"NOT {label}"
        if self.threshold == int(self.threshold) and abs(self.threshold) >= 1000:
            return f"{label} {self.operator} {self.threshold:,.0f}"
        return f"{label} {self.operator} {self.threshold:,.3f}"


@dataclass
class PolicyRule:
    """One leaf of the surrogate tree, expressed as a credit-policy rule."""

    rule_id: int
    conditions: list[Condition]
    predicted_probability: float
    n_applicants: int
    coverage_pct: float
    observed_default_rate: float | None = None
    band: str = ""

    def describe(self) -> str:
        """Multi-line IF/THEN rendering for the UI, README and policy docs."""
        header = (
            f"RULE {self.rule_id}  --  covers {self.n_applicants:,} applicants "
            f"({self.coverage_pct:.1f}% of the book)"
        )
        if not self.conditions:
            body = "  IF   (no conditions -- this is the whole population)"
        else:
            clauses = [f"  IF   {self.conditions[0].describe()}"]
            clauses += [f"  AND  {c.describe()}" for c in self.conditions[1:]]
            body = "\n".join(clauses)
        outcome = (
            f"  THEN predicted default risk {100 * self.predicted_probability:.1f}%"
            f"  ->  {self.band} risk band"
        )
        if self.observed_default_rate is not None:
            outcome += f"\n       observed default rate in this group: {self.observed_default_rate:.1f}%"
        return f"{header}\n{body}\n{outcome}"

    def to_row(self) -> dict[str, Any]:
        """Flat row for the rules table."""
        return {
            "rule_id": self.rule_id,
            "conditions": " AND ".join(c.describe() for c in self.conditions) or "(all applicants)",
            "n_conditions": len(self.conditions),
            "predicted_default_pct": round(100 * self.predicted_probability, 2),
            "observed_default_pct": (
                round(self.observed_default_rate, 2)
                if self.observed_default_rate is not None else None
            ),
            "n_applicants": self.n_applicants,
            "coverage_pct": round(self.coverage_pct, 2),
            "risk_band": self.band,
        }


@dataclass
class SurrogateReport:
    """A fitted surrogate plus its rules and its fidelity to the real model."""

    rules: list[PolicyRule]
    fidelity: dict[str, float]
    tree: DecisionTreeRegressor = field(repr=False)
    feature_names: list[str] = field(default_factory=list, repr=False)

    def to_frame(self) -> pd.DataFrame:
        """The rules table, ordered from lowest to highest predicted risk."""
        frame = pd.DataFrame([rule.to_row() for rule in self.rules])
        return frame.sort_values("predicted_default_pct").reset_index(drop=True)

    def render(self) -> str:
        """Full text rendering, fidelity first."""
        lines = [
            "SURROGATE POLICY RULES",
            "=" * 78,
            f"Fidelity to the deployed model: R^2 {self.fidelity['r2']:.3f}, "
            f"band agreement {self.fidelity['band_agreement_pct']:.1f}%",
            f"Tree depth {self.fidelity['depth']:.0f}, {len(self.rules)} rules",
            "",
        ]
        for rule in sorted(self.rules, key=lambda r: -r.predicted_probability):
            lines.append(rule.describe())
            lines.append("")
        return "\n".join(lines)


def _encode_for_tree(
    features: pd.DataFrame, categorical_features: list[str]
) -> pd.DataFrame:
    """Prepare the feature matrix for a decision tree.

    Categoricals become one-hot indicators named ``"COLUMN = level"``, so that a
    split on one renders as a plain membership test rather than an opaque
    threshold on an integer code. Numeric NaNs are filled with the median,
    because scikit-learn's tree cannot branch on missing values the way the
    boosted models can -- the surrogate is an approximation, and this is one of
    the places it approximates.

    Args:
        features: Model-ready feature frame.
        categorical_features: Names of the categorical columns.

    Returns:
        A fully numeric, NaN-free frame.
    """
    numeric = features.drop(columns=categorical_features, errors="ignore")
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    numeric = numeric.fillna(0.0)

    indicators: dict[str, np.ndarray] = {}
    for column in categorical_features:
        if column not in features.columns:
            continue
        series = features[column].astype(str)
        # Only levels with real support are worth a policy rule.
        for level in series.value_counts().head(8).index:
            indicators[f"{column} = {level}"] = (series == level).to_numpy(dtype=float)

    if indicators:
        numeric = pd.concat(
            [numeric, pd.DataFrame(indicators, index=features.index)], axis=1
        )
    return numeric


def _walk(
    tree: _tree.Tree, feature_names: list[str], total: int
) -> list[tuple[list[Condition], float, int]]:
    """Depth-first walk collecting the path and value of every leaf."""
    leaves: list[tuple[list[Condition], float, int]] = []

    def recurse(node: int, path: list[Condition]) -> None:
        if tree.feature[node] == _tree.TREE_UNDEFINED:
            leaves.append((list(path), float(tree.value[node][0][0]), int(tree.n_node_samples[node])))
            return
        name = feature_names[tree.feature[node]]
        threshold = float(tree.threshold[node])
        is_indicator = " = " in name

        recurse(tree.children_left[node], path + [Condition(name, "<=", threshold, is_indicator)])
        recurse(tree.children_right[node], path + [Condition(name, ">", threshold, is_indicator)])

    recurse(0, [])
    return leaves


def fit_surrogate(
    features: pd.DataFrame,
    model_probabilities: np.ndarray,
    categorical_features: list[str] | None = None,
    max_depth: int | None = None,
    actual_labels: np.ndarray | None = None,
    band_thresholds: dict[str, float] | None = None,
) -> SurrogateReport:
    """Fit a shallow surrogate tree and export it as policy rules.

    The tree is fitted against the deployed model's **calibrated probabilities**,
    not against the true labels. That distinction matters: the goal is to
    describe what the model does, including where it is wrong, rather than to
    train a second competing classifier.

    Args:
        features: Model-ready feature frame.
        model_probabilities: The deployed model's calibrated predictions.
        categorical_features: Categorical column names to one-hot for the tree.
        max_depth: Tree depth; defaults to the configured value (3-4 keeps the
            rules short enough to read).
        actual_labels: True outcomes, used to report each rule's realised
            default rate beside its predicted one.
        band_thresholds: Band edges, so each rule carries a risk band.

    Returns:
        The fitted surrogate, its rules and its fidelity metrics.
    """
    depth = max_depth or settings.surrogate_tree_max_depth
    encoded = _encode_for_tree(features, categorical_features or [])

    tree = DecisionTreeRegressor(
        max_depth=depth,
        min_samples_leaf=max(int(MIN_LEAF_FRACTION * len(encoded)), 20),
        random_state=settings.random_seed,
    )
    tree.fit(encoded, model_probabilities)

    surrogate_predictions = tree.predict(encoded)
    fidelity = {
        "r2": float(r2_score(model_probabilities, surrogate_predictions)),
        "depth": float(tree.get_depth()),
        "n_leaves": float(tree.get_n_leaves()),
        "mean_absolute_error": float(np.mean(np.abs(surrogate_predictions - model_probabilities))),
    }

    thresholds = band_thresholds or {
        "band_low_max": settings.risk_band_low_max,
        "band_medium_max": settings.risk_band_medium_max,
    }

    def band_of(probability: float) -> str:
        if probability <= thresholds["band_low_max"]:
            return "Low"
        return "Medium" if probability <= thresholds["band_medium_max"] else "High"

    # Band agreement is the metric that actually matters operationally: the
    # rules are trusted to route applicants, so they must route them the same
    # way the model does.
    model_bands = np.array([band_of(p) for p in model_probabilities])
    surrogate_bands = np.array([band_of(p) for p in surrogate_predictions])
    fidelity["band_agreement_pct"] = float(100 * (model_bands == surrogate_bands).mean())

    feature_names = list(encoded.columns)
    leaves = _walk(tree.tree_, feature_names, len(encoded))

    # Map every row to its leaf so realised default rates can be attached.
    leaf_ids = tree.apply(encoded)
    leaf_order = {}
    for index, node in enumerate(np.unique(leaf_ids)):
        leaf_order[node] = index

    rules: list[PolicyRule] = []
    for rule_id, (conditions, predicted, count) in enumerate(leaves, start=1):
        observed: float | None = None
        if actual_labels is not None:
            # Match the leaf by its predicted value, which is unique per leaf
            # up to floating point.
            mask = np.isclose(surrogate_predictions, predicted)
            if mask.sum() > 0:
                observed = float(100 * np.asarray(actual_labels)[mask].mean())

        rules.append(
            PolicyRule(
                rule_id=rule_id,
                conditions=conditions,
                predicted_probability=predicted,
                n_applicants=count,
                coverage_pct=100 * count / max(len(encoded), 1),
                observed_default_rate=observed,
                band=band_of(predicted),
            )
        )

    logger.info(
        "Surrogate fitted: depth %d, %d rules, R^2 %.3f, band agreement %.1f%%",
        tree.get_depth(), len(rules), fidelity["r2"], fidelity["band_agreement_pct"],
    )
    return SurrogateReport(rules=rules, fidelity=fidelity, tree=tree, feature_names=feature_names)


def derive_rules_from_trained_model(sample_size: int = 50_000) -> SurrogateReport:
    """Build the policy rules for the currently deployed model.

    Loads the trained artifacts, scores the training population, fits the
    surrogate against those scores and returns the rules.

    Args:
        sample_size: Maximum rows used to fit the surrogate. Large enough that
            each leaf still covers thousands of applicants on the full dataset,
            so a rule's realised default rate is a stable estimate rather than
            an artefact of a small slice.

    Returns:
        The surrogate report.
    """
    from src.data.loader import build_dataset
    from src.ml.predict import load_bundle

    bundle = load_bundle()
    raw = build_dataset("train")
    if len(raw) > sample_size:
        raw = raw.sample(sample_size, random_state=settings.random_seed)

    features = bundle.preprocessor.transform(raw)
    if bundle.metadata.get("requires_string_categoricals"):
        for column in bundle.metadata["categorical_features"]:
            features[column] = features[column].astype(str)
    features = features[bundle.feature_names]

    probabilities = bound_probability(
        bundle.calibrator.predict(bundle.model.predict_proba(features)[:, 1])
    )
    labels = raw["TARGET"].to_numpy() if "TARGET" in raw.columns else None

    return fit_surrogate(
        features=features,
        model_probabilities=probabilities,
        categorical_features=bundle.metadata["categorical_features"],
        actual_labels=labels,
        band_thresholds=bundle.thresholds,
    )


def main() -> SurrogateReport:  # pragma: no cover - CLI entry point
    """Command-line entry point: derive and print the credit-policy rules."""
    from src.utils.helpers import write_json

    report = derive_rules_from_trained_model()
    print("\n" + report.render())

    print("RULES TABLE")
    print("=" * 78)
    frame = report.to_frame()
    print(frame.to_string(index=False, max_colwidth=68))

    settings.ensure_directories()
    frame.to_csv(settings.reports_dir / "policy_rules.csv", index=False)
    (settings.reports_dir / "policy_rules.txt").write_text(report.render(), encoding="utf-8")
    write_json(report.fidelity, settings.reports_dir / "surrogate_fidelity.json")
    print(f"\nWritten to {settings.reports_dir}/policy_rules.{{csv,txt}}")
    return report


if __name__ == "__main__":  # pragma: no cover
    main()
