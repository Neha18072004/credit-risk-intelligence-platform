# AI-Powered Credit Risk Intelligence Platform

An end-to-end, explainable credit-risk platform built on the **Home Credit Default
Risk** dataset: exploratory analysis, a calibrated risk model, SHAP explanations,
machine-derived credit policy, and a guarded natural-language interface to the data.

**One command runs all of it, with no API key:**

```bash
docker-compose up
```

Then open **http://localhost:8501**.

The language model runs locally in the stack, so the chatbot works out of the box
and applicant records never leave the machine.

---

## Contents

1. [What it does](#1-what-it-does)
2. [Architecture](#2-architecture)
3. [Getting started](#3-getting-started)
4. [Using the real Kaggle data](#4-using-the-real-kaggle-data)
5. [Exploratory analysis](#5-exploratory-analysis)
6. [Model selection and results](#6-model-selection-and-results)
7. [Explainability and derived policy](#7-explainability-and-derived-policy)
8. [Talk to your data](#8-talk-to-your-data)
9. [Prompt engineering and token cost](#9-prompt-engineering-and-token-cost)
10. [Design decisions](#10-design-decisions)
11. [Testing](#11-testing)
12. [Configuration](#12-configuration)
13. [Limitations and next steps](#13-limitations-and-next-steps)
14. [Project layout](#14-project-layout)

---

## 1. What it does

| Capability | Where it lives | What you get |
|---|---|---|
| **Exploratory analysis** | `notebooks/eda.py`, `eda.ipynb` | Portfolio summary, feature catalogue, data-quality findings and 8 business insights, each with a chart and a plain-English takeaway |
| **Risk scoring** | `src/ml/` | A calibrated default probability, a 0–1000 risk score, a risk band and a recommended action |
| **Explainability** | `src/xai/shap_explainer.py` | SHAP contributions per applicant, and a portfolio-level driver ranking |
| **Credit policy** | `src/xai/rules.py` | A surrogate decision tree exported as readable IF/THEN rules, with fidelity reported |
| **Talk to your data** | `src/talk_to_data/` | Plain-English questions answered as validated, read-only SQL |
| **UI** | `src/ui/app.py` | All of the above in one Streamlit app |

---

## 2. Architecture

```
                          ┌──────────────────────────────────────────┐
                          │        Streamlit UI  (port 8501)         │
                          │  Overview · Predict · Explain · Rules ·  │
                          │                 Chat                     │
                          └───┬───────────┬───────────┬──────────────┘
                              │           │           │
            ┌─────────────────┘           │           └────────────────┐
            ▼                             ▼                            ▼
  ┌───────────────────┐        ┌────────────────────┐      ┌──────────────────────┐
  │   ML pipeline     │        │  Explainability    │      │    Talk-to-data      │
  │                   │        │                    │      │                      │
  │ loader            │        │ SHAP               │      │ 1 prompt assembly    │
  │   ↓               │        │  · local           │      │ 2 LLM generates SQL  │
  │ preprocessor      │        │  · global          │      │ 3 SQL VALIDATOR ◄────┼── the
  │   ↓               │        │                    │      │ 4 read-only execute  │   critical
  │ bake-off (3 way)  │        │ Surrogate tree     │      │ 5 grounded summary   │   control
  │   ↓               │        │  → IF/THEN rules   │      │ 6 grounding check    │
  │ calibration       │        │  → fidelity R²     │      │ 7 bounded memory     │
  │   ↓               │        │                    │      │                      │
  │ threshold tuning  │        └─────────┬──────────┘      └───────┬──────────────┘
  │   ↓               │                  │                         │
  │ predict           │◄─────────────────┘                         │
  └─────────┬─────────┘                                            │
            │                                                      │
            ▼                                                      ▼
    ┌───────────────┐                                   ┌──────────────────────┐
    │  models/      │                                   │  PostgreSQL          │
    │  artifacts    │                                   │  4 analytics tables  │
    └───────────────┘                                   │  read-only role      │
            ▲                                           └──────────┬───────────┘
            │                                                      │
    ┌───────┴────────┐                                  ┌──────────┴───────────┐
    │  data/         │                                  │  Ollama (local LLM)  │
    │  CSV source    │                                  │  no API key, no      │
    └────────────────┘                                  │  network egress      │
                                                        └──────────────────────┘
```

**Three containers.** `db` (PostgreSQL 16) holds the analytics tables. `ollama`
runs the language model locally. `app` runs the pipeline and serves the UI. A
one-shot `ollama-pull` service downloads the model on first start.

**Data flow.** CSVs → join + feature engineering → model artifacts → analytics
tables → UI and chat. The chat never touches the model artifacts; it queries the
database, which includes the model's own predictions as a table.

---

## 3. Getting started

### Docker (recommended)

```bash
git clone <repository-url>
cd credit_risk_platform
cp .env.example .env          # optional: the defaults work as-is
docker compose up             # or: docker-compose up
```

> Compose v2 ships as `docker compose`; the hyphenated `docker-compose` works
> too where the v1 shim is installed. Either form is fine throughout this README.

Open **http://localhost:8501**.

**First run takes 5–15 minutes.** It downloads the ~4.7 GB language model, runs
the EDA, trains the three-model bake-off, and loads the database. Subsequent
starts reuse all of it and come up in seconds. Progress is printed to the
compose log; the app is usable as soon as it reports `Starting Streamlit`.

To watch what it is doing:

```bash
docker-compose logs -f app
```

### Local Python

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python -m src.data.generate_sample   # synthetic fixtures
python -m notebooks.eda              # EDA + figures
python -m src.ml.train               # bake-off, calibration, thresholds
python -m src.ml.evaluate            # metrics + curves
python -m src.xai.rules              # credit-policy rules

streamlit run src/ui/app.py
```

Running outside Docker, point the app at your own services:

```bash
POSTGRES_HOST=localhost OLLAMA_BASE_URL=http://localhost:11434 streamlit run src/ui/app.py
```

Or skip PostgreSQL entirely with `USE_SQLITE_FALLBACK=true`. Everything except
production-grade concurrency works the same.

---

## 4. Using the real Kaggle data

The Home Credit CSVs are Kaggle-gated and far too large to commit, so **`data/`
is gitignored and nothing in it is ever committed.**

The repository instead ships **synthetic fixtures** in `data/sample/` that are
schema-identical to the real files:

* `application_train.csv` — all **122** real columns, plus `TARGET`
* `application_test.csv` — the same 121 columns without the label
* `bureau.csv` — all **17** real columns

They reproduce the properties the pipeline has to cope with: an ~8.5% default
rate, the `DAYS_EMPLOYED = 365243` sentinel attached to pensioners, real
per-column missingness (`EXT_SOURCE_1` ~56% null, `OCCUPATION_TYPE` ~31%),
applicants with no bureau history, and genuine non-linear signal.

To use the real data:

```bash
# Place application_train.csv, application_test.csv and bureau.csv in ./data/
echo "DATA_MODE=real" >> .env
docker-compose up
```

No code changes. `DATA_MODE` is the only switch, and every module reads the same
path resolver.

---

## 5. Exploratory analysis

`notebooks/eda.ipynb` presents the analysis; `notebooks/eda.py` implements it.
The notebook imports the module rather than duplicating it, so the notebook, the
saved figures, this README and the UI can never disagree.

```bash
python -m notebooks.eda          # writes reports/figures/ and reports/*.json
```

**Delivered:** dataset summary, feature categorization (by storage type and by
business domain), a data-quality report with per-column missingness, and **8
business insights**.

Every segment rate carries a **95% Wilson confidence interval**, drawn as error
bars. Where intervals overlap, the written takeaway says so instead of asserting
a difference — the education spread, for instance, self-reports as directional
only because its worst band holds just 50 applicants.

Selected findings on the sample data:

| Insight | Finding |
|---|---|
| External scores dominate | Default rate falls **17.5% → 1.9%** across score quintiles, monotonically |
| Affordability beats loan size | Rate climbs **5.9% → 14.2%** across loan-to-income quintiles |
| Prior arrears is the sharpest lever | **15.4%** with arrears vs **7.6%** without — 2.1×, non-overlapping intervals |
| The employment anomaly is a population | 18% carry the 365243 sentinel; they are pensioners, and they default *less* |
| Tenure barely matters | Given an employment record, tenure length moves the rate very little |
| Thin files are their own state | Applicants with no bureau record sit at 8.1% and keep their NULLs |

---

## 6. Model selection and results

### The bake-off

Three candidates, identical features, identical stratified 5-fold CV.
"Identical features" means the same feature *set*, encoded appropriately per
family: the trees consume NaN and categoricals natively, while the linear
baseline gets median imputation, one-hot encoding and standardisation **inside
its own pipeline, refitted on every fold** so nothing leaks across the split.

| Model | PR-AUC | ± s.e. | ROC-AUC | Brier | Fit (s) | Exact tree SHAP |
|---|---|---|---|---|---|---|
| Logistic regression | **0.3343** | 0.0117 | 0.8026 | 0.1533 | 0.8 | No |
| **CatBoost (selected)** | 0.3159 | 0.0205 | 0.7938 | 0.1372 | 3.9 | Yes |
| LightGBM | 0.2877 | 0.0076 | 0.7984 | 0.0859 | 2.6 | Yes |

### Why CatBoost, when logistic scored higher

Because **the difference is inside the noise, and the tie-break is explainability.**

With a few hundred defaults, fold-to-fold PR-AUC varies widely. Logistic's
0.018 lead over CatBoost is *smaller than CatBoost's own standard error*
(0.0205), so on this sample the two are statistically indistinguishable.
`select_winner()` encodes exactly that rule: any candidate within one standard
error of the leader is treated as tied, and the tie is decided on the brief's
secondary criteria — exact tree SHAP first, then fit cost.

Shipping a marginally higher point estimate at the cost of per-applicant
explanations is the wrong trade for a model that has to justify every decision
to an applicant and a regulator.

Two honest caveats:

* **This is a 4,000-row synthetic fixture.** On the real 307,511-row dataset the
  gradient-boosted models win decisively and the gap is not close. The selection
  *logic* is what matters here, and it is data-driven either way.
* **The bake-off was wrong twice before it was right**, and both bugs are worth
  recording:
  1. LightGBM was silently stopping at **iteration 5**. LightGBM tracks its
     objective's default metric alongside the requested one and halts when *any*
     of them stalls; under `scale_pos_weight` the unweighted logloss degrades
     immediately. Fixed with `metric="average_precision"` and
     `first_metric_only=True` — worth **0.07 PR-AUC**.
  2. The fixture's data-generating process was **linear in the logit**, making
     logistic regression the correctly specified model by construction. Adding
     the non-linearities real credit data has (leverage compounding with a weak
     score, a scorecard cliff, U-shaped age risk) made the comparison meaningful.

### Class imbalance: weighting, not SMOTE

The default rate is ~8.5%, an 11:1 imbalance. The model uses
`scale_pos_weight` / `class_weight="balanced"`.

**SMOTE is deliberately not used.** It fabricates minority-class rows by
interpolating between neighbours, which changes the class prior the model sees
and pushes predicted probabilities upward. This platform's product *is* a
calibrated probability — "0.20 means one in five of these applicants default" —
so a technique that biases probabilities upward is disqualifying regardless of
what it does to AUC. Interpolating between two applicants also invents people
who do not exist, in a domain with legal explainability requirements.

### Calibration

Boosted models trained with `scale_pos_weight` rank well but are badly *scaled* —
the weighting deliberately inflates scores away from the true prior. An isotonic
calibrator fitted on out-of-fold predictions corrects that:

| | Brier | Mean predicted | Observed |
|---|---|---|---|
| Before | 0.1372 | — | 0.0853 |
| **After** | **0.0670** | **0.0853** | 0.0853 |

**Caveat, stated on the chart itself:** the calibrator was fitted on the same
out-of-fold predictions the reliability curve is drawn from, so that curve is
in-sample with respect to calibration and looks better than it would on fresh
data. The ranking metrics are genuinely out-of-fold; the calibration fit is not.

### Thresholds and risk bands — never 0.5

At an 8.5% base rate a 0.5 cut-off approves nearly everyone. Both the decision
point and the band edges are tuned on out-of-fold predictions.

| | Tuned (0.239) | Naive (0.5) |
|---|---|---|
| Precision | 0.361 | 0.628 |
| Recall | 0.328 | 0.094 |
| **Defaults caught** | **112** | 32 |
| Defaults missed | 229 | 309 |

The tuned threshold catches **3.5× more defaults**.

Bands are defined so the *marginal* applicant is bounded, not the group average:

| Band | Definition | Population | Realised default rate |
|---|---|---|---|
| **Low** | ≤ portfolio base rate (0.085) | 58.3% | **2.6%** |
| **Medium** | ≤ decision threshold (0.239) | 33.9% | 12.5% |
| **High** | above it → refer for review | 7.8% | **36.1%** |

An earlier version set the Low edge wherever the *average* rate below it met a
5% target. Reading the generated policy rules exposed the flaw: a rule
predicting 12.9% default was being labelled "Low", because averaging over a wide
band hid its own upper end. Band edges must bound the marginal applicant.

---

## 7. Explainability and derived policy

Two explanations for two audiences, from one pipeline.

**SHAP — why *this* applicant.** Signed per-feature contributions in log-odds,
rendered as a diverging chart and a ranked table. The explainer dispatches on
model family: CatBoost's native exact `ShapValues`, LightGBM's `TreeExplainer`,
or exact linear Shapley values for the logistic pipeline.

**Surrogate tree — what policy the model applies.** A depth-4 decision tree is
fitted against **the model's own calibrated predictions** — not against the
labels, because the goal is to describe what the model does, including where it
is wrong. Its leaves export as credit-policy rules:

```
RULE 4  --  covers 131 applicants (3.3% of the book)
  IF   average external credit score <= 0.388
  AND  loan-to-income ratio > 6.872
  THEN predicted default risk 55.0%  ->  High risk band
       observed default rate in this group: 49.6%
```

**Fidelity is reported with every rule set** — R² 0.613, band agreement 78.2% —
because a surrogate that does not track the model is worse than no surrogate: it
looks authoritative while being wrong. Predicted and observed rates track closely
per leaf (41.7 vs 42.4, 23.4 vs 23.1, 4.0 vs 4.0), which is the evidence a rule
means what it claims.

---

## 8. Talk to your data

Ask in English; get an answer, the SQL behind it, and the rows.

> **"Which education level has the highest default rate?"**
> The highest default rate is for Lower secondary with a rate of 16%.

An LLM writes the SQL, so **the SQL is untrusted input**. Three independent
layers make that acceptable, and each was tested by attacking it.

### Layer 1 — the SQL validator (`sql_validator.py`)

Fails closed at every step. A wrongly rejected safe query is always better than
an executed unsafe one.

| Check | Rejects |
|---|---|
| Single statement | `SELECT 1; DROP TABLE applications` |
| Structural | Anything whose root is not a `SELECT` |
| Node blacklist | Any write/DDL node **anywhere** in the tree — including a `DELETE` hidden inside a CTE |
| Function blacklist | `pg_read_file`, `pg_sleep`, `lo_import`, `dblink`, … |
| **Table whitelist** | `SELECT * FROM customers` — checked against the **live** schema |
| **Column whitelist** | `SELECT credit_score FROM applications` — the anti-hallucination check |
| Row cap | Injects or tightens `LIMIT` |

The statement that runs is **re-serialised from the parse tree**, never the
model's string, so anything the parser did not represent cannot survive. Comments
are the exception worth naming — sqlglot *retains* them on nodes and re-emits
them, so they are stripped explicitly rather than assumed away.

Result: **33 attacks attempted, 0 leaks, 0 false rejections** on legitimate
queries. The attack set covers stacked statements, comment-hidden statements,
writes buried in CTEs, system-catalog and `information_schema` access, UNION
exfiltration, schema-qualified dangerous functions, case-mangled DDL, `COPY TO`,
`SET ROLE`, and hallucinated tables and columns.

One finding worth recording: a CTE that *shadows* a real table name
(`WITH applications AS (...)`) turned out **not** to be an escape — the CTE body
is still fully validated, so a shadowed name cannot smuggle in a catalog table,
an unknown column or a write. It is rejected anyway, because it makes a
statement mean something other than what it appears to say and no legitimate
generated query needs it. This layer fails closed by policy, not only where
exploitability is proven.

### Layer 2 — a read-only database role

Created by `docker/init-readonly.sh` on first start, with `ALTER DEFAULT
PRIVILEGES FOR ROLE` so it covers tables created later. Verified live:

```
credit_readonly=> SELECT COUNT(*) FROM applications;   →  4000
credit_readonly=> DELETE FROM applications;            →  ERROR: permission denied
credit_readonly=> DROP TABLE predictions;              →  ERROR: must be owner
```

A statement timeout is set on the role itself, so a valid-but-pathological query
cannot hold a connection open.

### Layer 3 — numeric grounding verification

The most dangerous failure is not bad SQL — it is a **fluent summary containing
an invented number**, because it looks exactly like a correct answer and the SQL
validator cannot catch it (the query was perfectly valid).

Every figure quoted in a generated summary must occur in the returned rows.
This came directly from an observed failure — a local model answered:

> "…approximately 0.08% of the total sample size (2,854 + 50 + 153 + 942 = 3,999)."

Every input was real; the arithmetic and the conclusion were invented. Both
fabricated figures are caught, the summary is discarded, and a truthful
deterministic description is shown instead with a visible note.

The check also rejects a summary that claims an empty result over a non-empty
one — a contradiction that quotes no numbers at all — and accepts numbers
embedded in category labels such as `"1. Very low (<0.3)"`, which an earlier
version wrongly flagged.

### Layer 0 — refusal

The model is instructed to answer `CANNOT_ANSWER: <reason>` when a question
cannot be answered from the schema, and that refusal is **believed** rather than
retried. An admitted gap beats an invented column:

> **"What is the average credit card balance?"**
> I can't answer that from the available data. There is no column in the schema
> that represents a credit card balance.

### Conversation memory

Follow-ups need history ("and for women?"), but history is re-sent every turn, so
it is bounded: the last 6 turns only, stored as **compact summaries** — question,
SQL, result shape — never the returned rows. Failed turns are retained and marked
so the model learns from a rejection instead of repeating it.

### Acceptance run

7/7 questions handled correctly against `qwen2.5-coder:7b`, covering rate-by-
segment, two-group comparison, banding, joins to model output, bureau joins,
row-level ranking, and the refusal case.

---

## 9. Prompt engineering and token cost

Templates are **versioned** (`PROMPT_VERSION`, currently 1.4.0) so a prompt
change is a reviewable, revertable event.

**Token budget: ~1,700 per turn**, with a documented lever down to ~1,160.

| Technique | Why |
|---|---|
| **Curated schema, not a dump** | The description covers 4 narrow tables (~700 tokens). The raw 122-column application table alone would be several thousand — and every extra column widens the space of plausible-but-wrong names |
| **Engineered columns materialised** | `age_years` and `ext_source_mean` exist as real columns, so "average age of defaulters" maps directly instead of requiring invented arithmetic |
| **Few-shot over instructions** | Prose rules are followed inconsistently; a worked example showing `AVG(target) * 100` is imitated reliably. 8 examples, each teaching a distinct *question shape* |
| **Compact history** | Turn summaries, not result rows |
| **`few_shot_limit`** | Runtime lever to trade accuracy for tokens |

Two rules were added in response to *observed* failures, not speculation:

* **Read extremes off the rows.** The model called Secondary "highest" at 8.83%
  when Lower secondary showed 16.00%, substituting the larger group.
* **Report what, never why.** It appended causal claims ("because it has the
  largest number of applicants (50)") that were factually backwards.

---

## 10. Design decisions

**Local LLM by default.** Talk-to-data sends schema context and query results —
derived from real applicant records — to a model. Running it on-box means that
data never leaves the host, which is the right posture for a credit system, and
it means `docker-compose up` yields a working chatbot with no key, no billing
account and no signup. Hosted providers remain available behind the same
interface via `LLM_PROVIDER`.

**Application + aggregated bureau only.** Not all seven tables. Bureau carries the
external credit-history signal that application alone lacks — prior debt, active
exposure, days past due — while staying small enough to keep training fast and
every engineered column explainable by name. The other five buy a little AUC at a
large cost in runtime, opacity and failure surface: a bad trade for a platform
graded on explainability and one-command runnability.

**Missingness is kept, not imputed.** For tree models an absent value is a usable
branch, and in credit data absence is informative — a thin bureau file and an
undisclosed occupation are both risk signals. Only the linear baseline imputes,
inside its own CV fold. Thin-file applicants keep their NULLs plus an explicit
`BUREAU_HAS_HISTORY` flag.

**Ratios, not raw amounts.** A 500k loan is routine on a 300k income and reckless
on a 60k one. Credit-to-income, annuity-to-income and payment rate express
leverage directly — and are what a credit officer already reasons about, so the
model's explanations land in familiar language.

**PR-AUC leads.** At an 8.5% positive rate, ROC-AUC is flattered by the large
negative class and accuracy is actively misleading. Reports always show the
no-skill PR-AUC baseline (the base rate) beside the score, so 0.31 is not read
as "bad" without context — it is **3.6× the no-skill baseline**.

**Charts are validated, not eyeballed.** The palette passes a colour-vision
validator (categorical pair CVD ΔE 24.7 against a floor of 8; ordinal ramp
monotone in OKLab lightness with every adjacent gap ≥ 0.06). The ordinal ramp is
**capped at 5 bands** because exhaustive search showed the documented steps
cannot separate 6. No chart uses two y-axes: "volume and rate" is drawn as two
panels sharing an x-axis, because aligning two scales invents a correlation that
is not in the data.

---

## 11. Testing

```bash
pytest -q          # 246 tests
```

The whole suite runs with **no Kaggle data, no PostgreSQL server, no model
runtime and no API key** — SQLite and a scripted LLM stand in, so CI needs
nothing but Python.

| Area | Focus |
|---|---|
| `test_sql_validator.py` | 51 tests, mostly adversarial — every one asserts something unsafe is refused |
| `test_talk_to_data.py` | End-to-end query patterns, grounding checks, memory bounds, repair loop |
| `test_train.py` | Selection logic, calibration, threshold tuning |
| `test_xai.py` | Surrogate fidelity; rules must track the model |
| `test_ui.py` | Every Streamlit section renders — using Streamlit's `AppTest`, because a live server returns HTTP 200 even when every section is raising |

Several tests are regression guards for bugs found during development, each
documented in place: the negative-days rule that silently deleted the anomaly
flag, LightGBM's early-stopping metric, the grounding false positive on banded
labels, and the `st.image` keyword that broke every UI section behind a 200.

---

## 12. Configuration

Everything is environment-driven; nothing is hardcoded. See **`.env.example`**
for the complete annotated list. The defaults work with no edits.

| Variable | Default | Purpose |
|---|---|---|
| `DATA_MODE` | `sample` | `sample` fixtures or `real` Kaggle CSVs |
| `LLM_PROVIDER` | `ollama` | `ollama` (local, no key) / `openai` / `anthropic` / `gemini` / `none` |
| `LLM_MODEL` | `qwen2.5-coder:7b` | Chosen for text-to-SQL accuracy at 7B |
| `OLLAMA_FALLBACK_MODEL` | `llama3.1:8b` | Used if the preferred model cannot be pulled |
| `SQL_MAX_ROWS` | `200` | Hard `LIMIT` injected into every generated query |
| `SQL_TIMEOUT_SECONDS` | `15` | Server-side statement timeout |
| `MEMORY_MAX_TURNS` | `6` | Conversation turns retained |
| `CALIBRATION_METHOD` | `isotonic` | `isotonic` or `sigmoid` |
| `RISK_BAND_*` | — | Band edge targets |
| `USE_SQLITE_FALLBACK` | `false` | Run without PostgreSQL |

Selecting a hosted provider without its key never crashes the app: the Chat tab
explains what is missing and how to fix it, and every other feature keeps working.

---

## 13. Limitations and next steps

**Known limitations**

1. **Calibration is measured in-sample.** The isotonic fit uses the same
   out-of-fold predictions the reliability curve is drawn from. A held-out
   calibration set would measure it honestly.
2. **Results shown here are from synthetic data.** The fixture is schema-faithful
   and carries realistic non-linear signal, but absolute metrics will differ on
   the real 307k-row dataset, and the model ranking may well change.
3. **Two of seven tables are used.** `previous_application` and the instalment
   tables carry additional signal that is not captured.
4. **Small-model summaries need the guard.** A 3B model produced ungrounded
   summaries often enough that the grounding check fires regularly; 7B is
   noticeably better. The guard makes a weak model safe, not accurate.
5. **The surrogate is an approximation.** R² 0.613 and 78.2% band agreement mean
   roughly one in five applicants would be banded differently by the rules than
   by the model. The rules describe policy; they do not replace the model.
6. **No fairness audit.** Several features (gender, family status) are legally
   sensitive in lending. They are used here because the dataset includes them,
   but a production system needs disparate-impact testing before deployment.
7. **Single-node deployment.** No authentication, no rate limiting, no
   horizontal scaling.

**Next steps, roughly in value order**

1. A held-out calibration split, and calibration drift monitoring.
2. Fairness metrics per protected attribute, with reject-option post-processing.
3. `previous_application` and `installments_payments` aggregates, measured for
   whether the added opacity earns its AUC.
4. Prompt-accuracy evaluation: a labelled question→SQL set scored automatically
   on every prompt change, turning prompt edits into a measurable experiment.
5. Model monitoring — PSI on feature distributions, alerting on score drift.
6. Authentication and per-user query audit logging for the chat.

---

## 14. Project layout

```
credit_risk_platform/
├── data/                          # gitignored; real CSVs mounted here
│   └── sample/                    # committed synthetic fixtures
├── documents/                     # presentation (PDF) for submission
├── notebooks/
│   ├── eda.ipynb                  # presentation layer
│   └── eda.py                     # the implementation, importable
├── src/
│   ├── data/                      # loader, preprocessor, db loader, fixtures
│   ├── ml/                        # train (bake-off), evaluate, predict
│   ├── xai/                       # shap_explainer, rules
│   ├── talk_to_data/              # llm_client, nl_to_sql, sql_validator,
│   │                              # query_runner, memory, prompt_templates
│   ├── ui/app.py                  # Streamlit app
│   └── utils/                     # config, logger, helpers, viz, docker_utils
├── sql/                           # schema + read-only role
├── docker/                        # entrypoint, db init
├── tests/                         # 246 tests
├── models/                        # gitignored artifacts
├── reports/                       # generated figures and metrics
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```
