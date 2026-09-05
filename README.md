# AI-Powered Credit Risk Intelligence Platform

An end-to-end, explainable credit-risk scoring platform built on the Home Credit
Default Risk dataset: EDA, a calibrated ML risk model, SHAP explainability,
derived credit-policy rules, and a guarded natural-language "talk to your data"
interface — all runnable with a single `docker-compose up`.

> **Status:** under construction. Full documentation lands in Phase 9.

## Quick start (local, sample data)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m src.data.generate_sample   # regenerate the synthetic fixtures
pytest -q
```

## Data

The real Home Credit CSVs are Kaggle-gated and **never committed**. `data/` is
gitignored; mount the real files there and set `DATA_MODE=real`. A synthetic,
schema-identical fixture set is committed under `data/sample/` so the entire
pipeline runs without the Kaggle download (`DATA_MODE=sample`, the default).
