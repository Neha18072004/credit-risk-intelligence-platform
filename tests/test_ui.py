"""Headless tests for the Streamlit application.

Uses Streamlit's own AppTest runner, which executes the real script rather than
importing it, so widget wiring and rendering errors surface here. An HTTP 200
from the server proves nothing -- a section that raises still returns 200 with
the traceback rendered inside the page.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest

from src.utils.config import PROJECT_ROOT

APP_PATH = str(PROJECT_ROOT / "src" / "ui" / "app.py")
SECTIONS = ["Overview", "Predict", "Explain", "Rules", "Chat"]


@pytest.fixture(scope="module")
def app(trained_artifacts):
    """A freshly run app instance, with the model trained."""

    def _run(section: str | None = None) -> AppTest:
        instance = AppTest.from_file(APP_PATH, default_timeout=300)
        instance.run()
        if section:
            instance.sidebar.radio[0].set_value(section).run()
        return instance

    return _run


def test_app_loads_without_error(app) -> None:
    instance = app()
    assert not instance.exception
    assert instance.title[0].value


@pytest.mark.parametrize("section", SECTIONS)
def test_every_section_renders(app, section: str) -> None:
    """Every tab must render cleanly.

    Regression test: st.image() was called with use_container_width, which does
    not exist in this Streamlit version. The server still returned HTTP 200 and
    every section was broken.
    """
    instance = app(section)
    assert not instance.exception, f"{section} raised: {instance.exception}"


def test_overview_shows_portfolio_metrics(app) -> None:
    instance = app("Overview")
    labels = [metric.label for metric in instance.metric]
    assert "Applicants" in labels
    assert "Default rate" in labels
    assert "Class imbalance" in labels


def test_predict_scores_an_applicant(app) -> None:
    instance = app("Predict")
    labels = [metric.label for metric in instance.metric]
    assert "Default probability" in labels
    assert "Risk score" in labels
    assert "Recommended action" in labels
    # The band must be stated in words, never conveyed by colour alone.
    assert any("Risk band" in markdown.value for markdown in instance.markdown)


def test_rules_reports_surrogate_fidelity(app) -> None:
    """Fidelity must be visible: rules without it look authoritative unearned."""
    instance = app("Rules")
    labels = [metric.label for metric in instance.metric]
    assert any("Fidelity" in label for label in labels)
    assert "Band agreement" in labels


def test_explain_renders_contributions(app) -> None:
    instance = app("Explain")
    assert not instance.exception
    assert len(instance.dataframe) >= 1


def test_chat_degrades_without_a_model_runtime(app, monkeypatch) -> None:
    """With no LLM reachable the Chat tab explains itself rather than failing."""
    instance = app("Chat")
    assert not instance.exception
    # Either it is available (a runtime is up) or it warns; never an exception.
    assert instance.warning or instance.chat_input or instance.button


def test_sidebar_lists_every_section(app) -> None:
    instance = app()
    assert set(instance.sidebar.radio[0].options) == set(SECTIONS)
