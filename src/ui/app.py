"""Streamlit application: the whole platform in one page.

Five sections, matching the five things the platform does:

* **Overview** -- portfolio summary, data quality, and the EDA insights.
* **Predict** -- score an applicant and see the band and decision.
* **Explain** -- SHAP, globally and for the selected applicant.
* **Rules** -- the surrogate tree's credit-policy rules.
* **Chat** -- ask the database a question in plain English.

Every section degrades independently. A missing model does not break the EDA
tab; an unavailable LLM does not break anything but Chat. Each shows what is
missing and the exact command that fixes it, because the first thing a new user
hits is a half-configured system.

Run it with::

    streamlit run src/ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Allow `streamlit run src/ui/app.py` from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.viz import RISK_BAND_COLORS

logger = get_logger(__name__)

st.set_page_config(
    page_title=settings.app_name,
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Band colours are the reserved status palette. They always appear beside the
# band's own name -- colour never carries the meaning by itself.
_BAND_CSS = "".join(
    f".band-{band.lower()} {{ background: {color}22; border-left: 4px solid {color}; }}"
    for band, color in RISK_BAND_COLORS.items()
)
st.markdown(
    f"""<style>
    .risk-card {{ padding: 1rem 1.25rem; border-radius: 8px; margin: 0.5rem 0; }}
    .risk-value {{ font-size: 2.4rem; font-weight: 600; line-height: 1.1; }}
    .risk-label {{ font-size: 0.85rem; opacity: 0.75; text-transform: uppercase;
                   letter-spacing: 0.04em; }}
    {_BAND_CSS}
    </style>""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading applicant data...")
def load_applicants() -> pd.DataFrame:
    """Load and cache the joined applicant dataset."""
    from src.data.loader import build_dataset

    return build_dataset("train")


@st.cache_data(show_spinner=False)
def load_summary(_frame: pd.DataFrame) -> dict[str, Any]:
    """Portfolio summary statistics."""
    from src.data.loader import dataset_summary

    return dataset_summary(_frame)


@st.cache_data(show_spinner=False)
def load_report_json(name: str) -> Any | None:
    """Read a generated report file, returning None when it does not exist."""
    from src.utils.helpers import read_json

    path = settings.reports_dir / name
    return read_json(path) if path.exists() else None


@st.cache_resource(show_spinner="Loading model...")
def load_model_bundle() -> Any | None:
    """Load the trained model bundle, or None when nothing is trained yet."""
    from src.ml.predict import artifacts_exist, load_bundle

    if not artifacts_exist():
        return None
    try:
        return load_bundle()
    except Exception as error:  # noqa: BLE001 - surfaced in the UI
        logger.exception("Failed to load model bundle")
        st.session_state["model_error"] = str(error)
        return None


@st.cache_resource(show_spinner="Deriving policy rules...")
def load_policy_rules() -> Any | None:
    """Derive the surrogate policy rules, or None if no model is available."""
    from src.xai.rules import derive_rules_from_trained_model

    try:
        return derive_rules_from_trained_model(sample_size=3000)
    except Exception as error:  # noqa: BLE001
        logger.warning("Could not derive rules: %s", error)
        return None


def get_chat_interface() -> Any:
    """Build the talk-to-data interface once per session.

    Held in session state rather than a cache_resource so that conversation
    memory belongs to the user's session and is not shared between visitors.
    """
    if "chat" not in st.session_state:
        from src.talk_to_data.nl_to_sql import TalkToData

        try:
            st.session_state["chat"] = TalkToData()
            st.session_state["chat_error"] = None
        except Exception as error:  # noqa: BLE001 - typically the DB being absent
            st.session_state["chat"] = None
            st.session_state["chat_error"] = str(error)
    return st.session_state["chat"]


# --------------------------------------------------------------------------- #
# Shared fragments
# --------------------------------------------------------------------------- #
def risk_card(probability: float, score: float, band: str, decision: str) -> None:
    """Render the headline result for one applicant."""
    columns = st.columns([1.4, 1, 1, 1.2])
    with columns[0]:
        st.markdown(
            f"""<div class="risk-card band-{band.lower()}">
            <div class="risk-label">Risk band</div>
            <div class="risk-value">{band}</div></div>""",
            unsafe_allow_html=True,
        )
    columns[1].metric("Default probability", f"{100 * probability:.1f}%")
    columns[2].metric("Risk score", f"{score:.0f} / {settings.risk_score_scale}")
    columns[3].metric("Recommended action", decision)


def missing_model_notice(feature: str) -> None:
    """Explain that the model is untrained, and how to fix it."""
    st.warning(
        f"**No trained model found**, so {feature} is unavailable.\n\n"
        "Train one with:\n```bash\npython -m src.ml.train\n```\n"
        "Under Docker this runs automatically on first start."
    )
    if error := st.session_state.get("model_error"):
        st.caption(f"Loader reported: {error}")


# --------------------------------------------------------------------------- #
# Section: Overview
# --------------------------------------------------------------------------- #
def render_overview(applicants: pd.DataFrame) -> None:
    """Portfolio summary, data quality and the EDA insights."""
    st.header("Portfolio overview")

    summary = load_summary(applicants)
    columns = st.columns(5)
    columns[0].metric("Applicants", f"{summary['n_rows']:,}")
    columns[1].metric("Features", summary["n_columns"])
    columns[2].metric("Defaults", f"{summary.get('n_defaults', 0):,}")
    columns[3].metric("Default rate", f"{100 * summary.get('default_rate', 0):.2f}%")
    columns[4].metric("Class imbalance", f"{summary.get('imbalance_ratio', 0):.0f}:1")

    st.caption(
        f"Data mode **{settings.data_mode.value}** -- reading from `{settings.active_data_dir}`. "
        "At this default rate accuracy is meaningless, so the model is evaluated on PR-AUC "
        "and its thresholds are tuned rather than left at 0.5."
    )

    insights = load_report_json("eda_insights.json")
    findings = load_report_json("data_quality_findings.json")

    if not insights:
        st.info(
            "EDA artifacts have not been generated yet. Run:\n"
            "```bash\npython -m notebooks.eda\n```"
        )
        return

    st.subheader("Business insights")
    st.caption(
        "Each rate carries a 95% Wilson confidence interval. Where intervals overlap, "
        "the takeaway says so rather than asserting a difference."
    )

    for insight in insights:
        with st.expander(insight["title"], expanded=insight is insights[0]):
            figure = settings.figures_dir / (insight.get("figure") or "")
            if figure.exists():
                st.image(str(figure), use_column_width=True)
            st.markdown(f"**Takeaway.** {insight['takeaway']}")

    if findings:
        st.subheader("Data quality")
        st.caption("Every issue below is acted on in the preprocessing pipeline.")
        st.dataframe(
            pd.DataFrame(findings)[
                ["severity", "issue", "n_affected", "pct_affected", "treatment"]
            ],
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------------------------------- #
# Section: Predict
# --------------------------------------------------------------------------- #
def render_predict(applicants: pd.DataFrame) -> None:
    """Score an existing applicant, or a hypothetical one."""
    st.header("Score an applicant")

    bundle = load_model_bundle()
    if bundle is None:
        missing_model_notice("scoring")
        return

    from src.ml.predict import predict_applicant

    mode = st.radio(
        "Applicant source",
        ["Pick an existing applicant", "Enter details manually"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Pick an existing applicant":
        identifiers = applicants["SK_ID_CURR"].tolist()
        chosen = st.selectbox(
            "Applicant ID", identifiers, index=0,
            help="Any applicant from the loaded dataset.",
        )
        row = applicants[applicants["SK_ID_CURR"] == chosen]
    else:
        row = _manual_applicant_form(applicants)

    if row is None or row.empty:
        return

    st.session_state["selected_applicant"] = row

    with st.spinner("Scoring..."):
        result = predict_applicant(row, top_n=10)

    risk_card(result.probability, result.risk_score, result.risk_band, result.decision)

    thresholds = bundle.thresholds
    st.caption(
        f"Bands are tuned from out-of-fold predictions, never 0.5: "
        f"**Low** at or below {thresholds['band_low_max']:.3f} (the portfolio base rate), "
        f"**Medium** up to {thresholds['band_medium_max']:.3f}, **High** above it. "
        f"The decision threshold is {thresholds['decision_threshold']:.3f}."
    )

    with st.expander("Applicant details"):
        display = row.iloc[0]
        fields = [
            "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH",
            "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS",
            "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "BUREAU_HAS_OVERDUE",
        ]
        present = [field for field in fields if field in display.index]
        st.dataframe(
            pd.DataFrame(
                {
                    "field": present,
                    # Rendered as text: the column mixes strings and floats, and
                    # Arrow cannot type a mixed column without coercing it.
                    "value": [_render_value(display[field]) for field in present],
                }
            ),
            use_container_width=True, hide_index=True,
        )

    st.info(
        "See **Explain** for why this applicant scored as they did, and **Rules** for the "
        "policy the model applies across the whole book."
    )


def _render_value(value: Any) -> str:
    """Format one applicant field for display."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "not provided"
    if isinstance(value, float):
        return f"{value:,.4g}"
    if isinstance(value, (int,)):
        return f"{value:,}"
    return str(value)


def _manual_applicant_form(applicants: pd.DataFrame) -> pd.DataFrame | None:
    """Collect the fields that matter most, defaulting to a real applicant.

    Only the highest-impact fields are exposed. Everything else inherits from
    the template row, because the preprocessor fills absent columns with NaN and
    the tree model reads missingness natively -- a partly-specified applicant is
    a legitimate input, not an error.
    """
    template = applicants.iloc[[0]].copy()

    with st.form("applicant_form"):
        columns = st.columns(3)
        income = columns[0].number_input(
            "Annual income", min_value=10_000.0, max_value=5_000_000.0,
            value=float(template["AMT_INCOME_TOTAL"].iloc[0]), step=10_000.0,
        )
        credit = columns[1].number_input(
            "Loan amount", min_value=10_000.0, max_value=5_000_000.0,
            value=float(template["AMT_CREDIT"].iloc[0]), step=10_000.0,
        )
        annuity = columns[2].number_input(
            "Annual instalment", min_value=1_000.0, max_value=500_000.0,
            value=float(template["AMT_ANNUITY"].iloc[0] or 24_000), step=1_000.0,
        )

        columns = st.columns(3)
        age = columns[0].slider("Age", 21, 70, 40)
        employed = columns[1].slider("Years employed", 0.0, 40.0, 5.0, 0.5)
        children = columns[2].slider("Children", 0, 5, 0)

        st.markdown("**External credit scores** (0-1, higher is safer)")
        columns = st.columns(3)
        ext1 = columns[0].slider("EXT_SOURCE_1", 0.0, 1.0, 0.5, 0.01)
        ext2 = columns[1].slider("EXT_SOURCE_2", 0.0, 1.0, 0.5, 0.01)
        ext3 = columns[2].slider("EXT_SOURCE_3", 0.0, 1.0, 0.5, 0.01)

        columns = st.columns(3)
        education = columns[0].selectbox(
            "Education", sorted(applicants["NAME_EDUCATION_TYPE"].dropna().unique())
        )
        income_type = columns[1].selectbox(
            "Income type", sorted(applicants["NAME_INCOME_TYPE"].dropna().unique())
        )
        arrears = columns[2].selectbox("Prior arrears on external credit", ["No", "Yes"])

        submitted = st.form_submit_button("Score this applicant", type="primary")

    if not submitted:
        return None

    template["AMT_INCOME_TOTAL"] = income
    template["AMT_CREDIT"] = credit
    template["AMT_ANNUITY"] = annuity
    template["AMT_GOODS_PRICE"] = credit * 0.9
    template["DAYS_BIRTH"] = -int(age * 365.25)
    template["DAYS_EMPLOYED"] = -int(employed * 365.25)
    template["CNT_CHILDREN"] = children
    template["CNT_FAM_MEMBERS"] = children + 2
    template["EXT_SOURCE_1"] = ext1
    template["EXT_SOURCE_2"] = ext2
    template["EXT_SOURCE_3"] = ext3
    template["NAME_EDUCATION_TYPE"] = education
    template["NAME_INCOME_TYPE"] = income_type
    template["BUREAU_HAS_OVERDUE"] = 1 if arrears == "Yes" else 0
    template["SK_ID_CURR"] = -1  # marks a hypothetical applicant
    return template


# --------------------------------------------------------------------------- #
# Section: Explain
# --------------------------------------------------------------------------- #
def render_explain(applicants: pd.DataFrame) -> None:
    """SHAP explanations, globally and for one applicant."""
    st.header("Explainability")

    bundle = load_model_bundle()
    if bundle is None:
        missing_model_notice("SHAP explanations")
        return

    from src.ml.predict import predict_applicant
    from src.xai.shap_explainer import plot_global_importance, plot_local_explanation

    local_tab, global_tab = st.tabs(["This applicant", "Whole portfolio"])

    with local_tab:
        row = st.session_state.get("selected_applicant")
        if row is None:
            row = applicants.iloc[[0]]
            st.caption("Showing the first applicant. Choose another in **Predict**.")

        with st.spinner("Computing SHAP contributions..."):
            result = predict_applicant(row, top_n=10)

        risk_card(result.probability, result.risk_score, result.risk_band, result.decision)
        figure = plot_local_explanation(
            result.top_contributions, result.probability, result.applicant_id
        )
        st.image(str(figure), use_column_width=True)

        st.markdown("**Reasons, strongest first**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "feature": c.feature,
                        "value": c.value,
                        "effect": c.direction,
                        "contribution": round(c.contribution, 4),
                    }
                    for c in result.top_contributions
                ]
            ),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Contributions are in log-odds. Positive pushes the applicant towards default, "
            "negative away from it; they sum to the model's output for this applicant."
        )

    with global_tab:
        sample_size = st.slider("Applicants sampled", 100, 1000, 400, 100)
        with st.spinner("Computing global importance..."):
            features = bundle.preprocessor.transform(applicants.head(sample_size))
            if bundle.metadata.get("requires_string_categoricals"):
                for column in bundle.metadata["categorical_features"]:
                    features[column] = features[column].astype(str)
            importance = bundle.explainer.global_importance(
                features[bundle.feature_names], sample_size=sample_size, top_n=15
            )
        st.image(str(plot_global_importance(importance)), use_column_width=True)
        st.dataframe(importance, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Section: Rules
# --------------------------------------------------------------------------- #
def render_rules() -> None:
    """The surrogate tree's credit-policy rules."""
    st.header("Derived credit policy")
    st.markdown(
        "A shallow decision tree is fitted to **the model's own predictions**, turning what "
        "the model does into rules a credit committee can read, challenge and audit. "
        "Fidelity is reported because a surrogate that does not track the model would look "
        "authoritative while being wrong."
    )

    if load_model_bundle() is None:
        missing_model_notice("policy rules")
        return

    report = load_policy_rules()
    if report is None:
        st.error("Could not derive the policy rules. See the application logs for details.")
        return

    columns = st.columns(4)
    columns[0].metric("Fidelity (R²)", f"{report.fidelity['r2']:.3f}")
    columns[1].metric("Band agreement", f"{report.fidelity['band_agreement_pct']:.1f}%")
    columns[2].metric("Tree depth", f"{report.fidelity['depth']:.0f}")
    columns[3].metric("Rules", len(report.rules))

    st.subheader("Rules table")
    frame = report.to_frame()
    st.dataframe(
        frame, use_container_width=True, hide_index=True,
        column_config={
            "predicted_default_pct": st.column_config.NumberColumn(
                "Predicted default %", format="%.2f"
            ),
            "observed_default_pct": st.column_config.NumberColumn(
                "Observed default %", format="%.2f"
            ),
            "coverage_pct": st.column_config.NumberColumn("Coverage %", format="%.1f"),
        },
    )
    st.caption(
        "Predicted is what the surrogate says; observed is the realised default rate of the "
        "applicants the rule actually covers. Close agreement is the evidence that a rule "
        "means what it claims."
    )

    st.subheader("Policy statements")
    for rule in sorted(report.rules, key=lambda r: -r.predicted_probability):
        st.code(rule.describe(), language="text")

    st.download_button(
        "Download rules (CSV)", frame.to_csv(index=False),
        file_name="credit_policy_rules.csv", mime="text/csv",
    )


# --------------------------------------------------------------------------- #
# Section: Chat
# --------------------------------------------------------------------------- #
SUGGESTED_QUESTIONS: list[str] = [
    "What is the overall default rate?",
    "Which education level has the highest default rate?",
    "Compare average income between applicants who defaulted and those who repaid.",
    "How does the default rate vary across external credit score bands?",
    "Do applicants with prior arrears default more often?",
    "Show me the 10 riskiest applicants the model flagged for review.",
]


def render_chat() -> None:
    """Natural-language question answering over the database."""
    st.header("Talk to your data")

    interface = get_chat_interface()
    if interface is None:
        st.error(
            "**Database unavailable**, so the chat cannot run.\n\n"
            "Under `docker-compose up` the database starts and loads automatically. "
            "Running locally, start PostgreSQL and load it with:\n"
            "```bash\npython -m src.data.db_loader\n```"
        )
        if error := st.session_state.get("chat_error"):
            st.caption(f"Reported: {error}")
        return

    available, message = interface.is_available()
    if not available:
        st.warning(f"**Chat is unavailable.** {message}")
        st.caption("Every other section of the app continues to work.")
        return

    st.caption(
        f"{message} Questions become read-only SQL, validated before execution: "
        "single SELECT only, tables and columns checked against the live schema, "
        "row cap enforced, and answers checked to quote only figures present in the result."
    )

    st.markdown("**Try one of these**")
    columns = st.columns(3)
    for index, question in enumerate(SUGGESTED_QUESTIONS):
        if columns[index % 3].button(question, key=f"suggested_{index}", use_container_width=True):
            st.session_state["pending_question"] = question

    for turn in interface.memory.turns:
        with st.chat_message("user"):
            st.write(turn.question)
        with st.chat_message("assistant"):
            st.write(turn.answer if turn.succeeded else f":warning: {turn.error}")

    typed = st.chat_input("Ask a question about the portfolio...")
    question = typed or st.session_state.pop("pending_question", None)
    if not question:
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating and validating SQL..."):
            result = interface.ask(question)

        if not result.success:
            st.warning(result.answer or result.error)
            if result.refused:
                st.caption(
                    "The model declined rather than inventing a column -- that is the "
                    "intended behaviour when a question cannot be answered from the data."
                )
            return

        st.write(result.answer)
        if result.summary_source.startswith("deterministic_after"):
            st.caption(
                ":warning: The generated summary quoted figures that were not in the result, "
                "so it was discarded and replaced with a description of the actual rows."
            )

        if not result.rows.empty:
            st.dataframe(result.rows, use_container_width=True, hide_index=True)

        with st.expander("SQL and diagnostics"):
            st.code(result.sql, language="sql")
            columns = st.columns(4)
            columns[0].metric("Rows", result.row_count)
            columns[1].metric("Time", f"{result.elapsed_seconds:.1f}s")
            columns[2].metric("Tokens", result.prompt_tokens + result.completion_tokens)
            columns[3].metric("Repaired", "Yes" if result.repair_attempted else "No")
            st.caption(
                f"Tables used: {', '.join(result.tables_used) or 'none'} | "
                f"summary via {result.summary_source} | prompt v{result.prompt_version}"
            )

    st.rerun()


# --------------------------------------------------------------------------- #
# Sidebar and entry point
# --------------------------------------------------------------------------- #
def render_sidebar() -> str:
    """Render the sidebar and return the selected section."""
    with st.sidebar:
        st.title("Credit Risk Intelligence")
        section = st.radio(
            "Section",
            ["Overview", "Predict", "Explain", "Rules", "Chat"],
            label_visibility="collapsed",
        )

        st.divider()
        st.subheader("System status")

        bundle = load_model_bundle()
        if bundle is not None:
            metrics = load_report_json("evaluation.json")
            st.success(f"Model: {bundle.model_name}")
            if metrics:
                st.caption(
                    f"PR-AUC {metrics['ranking']['pr_auc']:.3f} "
                    f"({metrics['ranking']['pr_auc_lift_over_random']:.1f}x baseline) | "
                    f"ROC-AUC {metrics['ranking']['roc_auc']:.3f}"
                )
        else:
            st.warning("Model: not trained")

        # Probe the provider directly rather than reading session state: the
        # sidebar renders before the Chat section, so session state is still
        # empty on the first pass and would report "not initialised" even when
        # the runtime is up.
        from src.talk_to_data.llm_client import get_llm_client

        chat_available, chat_message = get_llm_client().is_available()
        (st.success if chat_available else st.warning)(
            f"Chat: {settings.llm_provider.value}"
            + ("" if chat_available else " (unavailable)")
        )
        st.caption(chat_message if chat_available else "Every other section still works.")

        st.caption(f"Data mode: **{settings.data_mode.value}**")

        if section == "Chat" and st.session_state.get("chat") is not None:
            st.divider()
            if st.button("Clear conversation", use_container_width=True):
                st.session_state["chat"].reset()
                st.rerun()

    return section


def main() -> None:
    """Application entry point."""
    section = render_sidebar()

    try:
        applicants = load_applicants()
    except FileNotFoundError as error:
        st.error(f"**Could not load the dataset.**\n\n{error}")
        st.stop()
        return

    if section == "Overview":
        render_overview(applicants)
    elif section == "Predict":
        render_predict(applicants)
    elif section == "Explain":
        render_explain(applicants)
    elif section == "Rules":
        render_rules()
    elif section == "Chat":
        render_chat()


main()
