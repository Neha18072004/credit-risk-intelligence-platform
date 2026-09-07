# AI-Powered Credit Risk Intelligence Platform

An end-to-end, explainable credit-risk platform on the **Home Credit Default Risk**
dataset: exploratory analysis, a calibrated ML risk model, SHAP explanations,
machine-derived credit policy, and a guarded natural-language interface to the data.

**Live app:** https://neha-credit-risk.streamlit.app  
**Trained on the full 307,511-row dataset** — the model ships with the repo, so a
fresh clone scores applicants immediately with no training step.

| ROC-AUC | PR-AUC | Brier (calibrated) | Defaults caught vs 0.5 cut-off |
|:---:|:---:|:---:|:---:|
| **0.782** | **0.268** (3.3× baseline) | **0.066** | **10,665 vs 713** |

---

## Quick start

```bash
git clone https://github.com/Neha18072004/credit-risk-intelligence-platform
cd credit-risk-intelligence-platform
docker compose up
```

Open **http://localhost:8501**. That's it — no API key, no configuration.

The language model runs **locally** in the stack, so the chatbot works out of the
box and applicant data never leaves the machine. First run downloads it (~4.7GB),
so allow 5–15 minutes; later runs start in seconds.

<details>
<summary><b>Run without Docker</b></summary>

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # add -r requirements-dev.txt for tests
cp .env.example .env

streamlit run src/ui/app.py              # the model is already trained and committed
```

To rebuild everything from scratch:

```bash
python -m src.data.generate_sample   # synthetic fixtures
python -m notebooks.eda              # EDA + figures
python -m src.ml.train               # bake-off, calibration, thresholds
python -m src.ml.evaluate            # metrics + curves
python -m src.xai.rules              # credit-policy rules
```
</details>

<details>
<summary><b>Lightweight deployment (1.83GB instead of 10.43GB)</b></summary>

```bash
docker compose -f docker-compose.lite.yml up
```

Drops the local model runtime — app + PostgreSQL only. **267MB RAM at rest**
instead of ~6GB. Everything works except the chat, which needs a hosted provider
(`LLM_PROVIDER` + key) or reports itself unavailable with a clear message.

The live app above runs on Streamlit Community Cloud using this shape: SQLite
instead of PostgreSQL, sample fixtures instead of the Kaggle files — but still
scoring with the model trained on all 307,511 real rows. Peak memory 356MB.
</details>

<details>
<summary><b>Using the real Kaggle data</b></summary>

`data/` is gitignored — the real CSVs are 2.5GB and Kaggle-gated. The repo ships
**synthetic fixtures** in `data/sample/` that are schema-identical, including the
~8% default rate, the `DAYS_EMPLOYED = 365243` sentinel, and realistic missingness.

Drop `application_train.csv`, `application_test.csv`, `bureau.csv`,
`previous_application.csv` and `installments_payments.csv` into `data/`, then set
`DATA_MODE=real`. No code changes — that switch is the only difference.
</details>

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │      Streamlit UI · port 8501       │
                    │ Overview · Predict · Explain ·      │
                    │      Rules · Chat · Audit           │
                    └──┬───────────┬────────────┬─────────┘
             ┌─────────┘           │            └──────────┐
             ▼                     ▼                       ▼
    ┌────────────────┐   ┌──────────────────┐   ┌────────────────────┐
    │  ML pipeline   │   │ Explainability   │   │   Talk-to-data     │
    │                │   │                  │   │                    │
    │ load + join    │   │ SHAP local       │   │ 1 prompt assembly  │
    │ preprocess     │   │ SHAP global      │   │ 2 LLM → SQL        │
    │ bake-off ×3    │   │                  │   │ 3 SQL VALIDATOR ◄──┼── the
    │ calibration    │   │ surrogate tree   │   │ 4 read-only exec   │   critical
    │ thresholds     │   │  → IF/THEN rules │   │ 5 grounded summary │   control
    │ predict        │   │  → fidelity      │   │ 6 grounding check  │
    └───────┬────────┘   └────────┬─────────┘   └─────────┬──────────┘
            ▼                     │                       ▼
     ┌─────────────┐              │            ┌────────────────────┐
     │  models/    │◄─────────────┘            │    PostgreSQL      │
     │  (shipped)  │                           │  5 analytics tables│
     └─────────────┘                           │  read-only role    │
                                               └─────────┬──────────┘
                                               ┌─────────▼──────────┐
                                               │  Ollama · local    │
                                               │  no key, no egress │
                                               └────────────────────┘
```

**Three containers.** `db` (PostgreSQL) holds the analytics tables, `ollama` runs
the model locally, `app` runs the pipeline and serves the UI.

**Data flow.** CSVs → join + feature engineering → model artifacts → analytics
tables → UI and chat. The chat never touches the model artifacts; it queries the
database, which includes the model's own predictions as a table.

---

## 1 · Exploratory analysis

`notebooks/eda.py` implements it; `eda.ipynb` imports and presents it, so notebook,
figures, README and UI can never disagree. Every rate carries a **95% Wilson
confidence interval**, and takeaways hedge automatically when intervals overlap.

Nine insights. The six that matter most:

| Finding | Evidence |
|---|---|
| External scores dominate | Default rate falls **18.7% → 2.5%** across quintiles, monotonically |
| **Loan size does *not* separate risk** | Loan-to-income is an inverted U — **7.3% at both extremes**, 8.9% in the middle |
| Instalment burden does | Instalment-to-income rises **7.2% → 8.7%** consistently |
| Prior repayment behaviour | Paid late before: **9.4%** vs 6.8% |
| Prior arrears elsewhere | **15.9%** vs 7.6% — 2.1×, non-overlapping intervals |
| The employment anomaly | 18% carry the `365243` sentinel; they are pensioners, and default **less** (5.4% vs 8.7%) |

![Default rate by external credit score band](reports/figures/04_external_scores.png)

![Affordability ratios](reports/figures/05_affordability.png)

The loan-to-income result is worth dwelling on because it contradicts the obvious
assumption: the most heavily leveraged applicants are **not** the riskiest, most
likely because very high ratios pick up secured and longer-term products rather
than distressed borrowing. So **constrain the instalment, not the loan size.**

<details>
<summary><b>Data quality findings, all acted on in the preprocessor</b></summary>

| Issue | Scale | Treatment |
|---|---|---|
| `DAYS_EMPLOYED = 365243` | 18% of rows | Nulled + explicit flag. It is a sentinel for "no employment record", not 1000 years of work |
| Columns >50% missing | 41 columns | Retained un-imputed — trees branch on missingness, and a thin file is real risk information |
| `CODE_GENDER = 'XNA'` | 4 rows | Mapped to null, not treated as a third gender |
| Extreme income outliers | ~1% | Left in place; trees are rank-based, and ratio features neutralise scale |
| No external credit history | 14% | Nulls preserved + `BUREAU_HAS_HISTORY` flag |

An earlier version of the affordability takeaway asserted the *opposite* conclusion,
because it was written by hand. It now derives its claim by rank correlation and
reaches opposite, correct conclusions on the real and synthetic datasets.
</details>

---

## 2 · Machine learning

### Which tables earn their place

Four of seven, chosen by measurement — `src/ml/ablation.py` switches each on in turn:

| Configuration | Features | PR-AUC | Marginal gain |
|---|---|---|---|
| `application` only | 139 | 0.2507 | — |
| `+ bureau` | 166 | 0.2576 | **+0.0069** |
| `+ previous_application` | 183 | 0.2638 | **+0.0062** |
| `+ installments_payments` | 198 | 0.2708 | **+0.0070** |

Each contributes roughly equally, so each stays. The three monthly panels are
excluded as largely redundant. `installments_payments` matters beyond its AUC: it
is the table that actually contains **repayment behaviour**, which the brief names
as an analysis area and which nothing else substitutes for.

### The bake-off

Three candidates, identical folds, 307,511 rows, 198 features:

| Model | PR-AUC | ± s.e. | ROC-AUC | Brier | Fit | Tree SHAP |
|---|---|---|---|---|---|---|
| **LightGBM — selected** | **0.2720** | 0.0033 | 0.7811 | 0.1711 | 33s | Yes |
| CatBoost | 0.2705 | 0.0039 | 0.7809 | 0.1801 | 209s | Yes |
| Logistic regression | 0.2480 | 0.0026 | 0.7666 | 0.1961 | 36s | No |

**Why LightGBM.** It and CatBoost are *statistically tied* — 0.0015 apart against
standard errors of 0.0033 and 0.0039. Any candidate within one standard error of
the leader is treated as tied, and the tie goes to exact tree-SHAP support, then
fit cost. Same performance in a sixth of the time.

Hyperparameters come from a randomised search over 12 settings per model, scored on
**average precision** — the same metric the bake-off selects on, because tuning for
one objective and selecting on another produces a model that looks better and
performs worse. The searched values are baked in as defaults.

### Class imbalance: weighting, not SMOTE

An 11.4:1 ratio, handled with `scale_pos_weight` / `class_weight="balanced"`.

**SMOTE is deliberately not used.** It fabricates minority rows by interpolation,
which shifts the class prior and biases probabilities upward. This platform's
product *is* a calibrated probability, so that is disqualifying regardless of what
it does to AUC — and interpolating between two applicants invents people who do not
exist, in a domain with legal explainability requirements.

### Calibration and thresholds

| | Brier | Mean predicted | Observed |
|---|---|---|---|
| Before | 0.1711 | — | 0.0807 |
| **After isotonic** | **0.0663** | **0.0807** | 0.0807 |

![Calibration curve](reports/figures/11_calibration.png)

*Stated on the chart itself: the calibrator was fitted on these same out-of-fold
predictions, so this curve is in-sample with respect to calibration.*

At an 8% base rate a 0.5 cut-off approves nearly everyone, so both the decision
point and the band edges are **tuned on out-of-fold predictions**:

| | Tuned (0.165) | Naive (0.5) |
|---|---|---|
| Recall | **0.430** | 0.029 |
| Share of book flagged | 13.0% | 0.4% |
| **Defaults caught** | **10,665** | 713 |

The naive cut-off looks more *precise* only because it flags almost nobody — it
finds 3% of the defaults in the book, which is not a usable credit policy.

| Band | Definition | Population | Realised default | Share of all defaults |
|---|---|---|---|---|
| **Low** | ≤ portfolio base rate | 66.3% | **3.34%** | 27% |
| **Medium** | ≤ decision threshold | 21.0% | 11.65% | 30% |
| **High** | above it → refer | 12.6% | **26.96%** | **42%** |

The High band is an eighth of the book and holds two-fifths of the defaults.

<details>
<summary><b>Two bugs found before the bake-off could be trusted</b></summary>

**1 · LightGBM was silently stopping at iteration 5.** It tracks its objective's
default metric alongside the requested one and halts when *any* stalls; under
`scale_pos_weight` the unweighted logloss degrades immediately. Fixed with
`metric="average_precision"` and `first_metric_only`. Worth **0.07 PR-AUC** — the
bake-off had been comparing a crippled model.

**2 · The synthetic fixture flattered logistic regression.** Its data-generating
process was linear in the logit, making the linear model correctly specified *by
construction*. Adding the non-linearities real credit data has made the comparison
meaningful, and the real data then confirmed the direction — the boosters win by
roughly seven standard errors.

**Band edges were also wrong at first.** They were set where the *average* default
rate below the edge met a target, which let a wide band hide its own upper end and
put a rule predicting 12.9% default into the band labelled "Low". Reading the
generated policy rules exposed it. Edges now bound the *marginal* applicant.
</details>

---

## 3 · Explainability

Two explanations for two audiences, from one engine.

**SHAP — why *this* applicant.** Signed contributions, rendered as a diverging chart,
a ranked table, and a plain-English paragraph:

> This applicant has a **87.5%** estimated probability of default, placing them in
> the **High** risk band. Recommended action: refer for review. The main factors
> increasing risk are that their **average external credit score of 0.171** raises
> the risk, their combined external credit score of 3.25e-05 raises the risk, and
> their **share of instalments paid late of 48%** raises the risk.

![Local SHAP explanation](reports/figures/14_shap_local_explanation.png)

A ranked table of log-odds explains things to a modeller. Someone who has been
**declined** is owed something they can act on. SHAP and the policy rules draw from
one shared vocabulary (`src/xai/feature_labels.py`), so the two can never describe
the same feature differently. Raw SHAP values remain in the table for anyone who
wants them.

**No prediction claims certainty.** Isotonic calibration returns exactly 0 and 1 on
pure bins, which produced "100.0% probability of default" — indefensible from a
finite sample. Probabilities are bounded to [0.001, 0.999].

### Derived credit policy

A depth-4 surrogate tree is fitted to the model's **own calibrated predictions** —
not to the labels, because the goal is to describe what the model does, *including
where it is wrong*. Its leaves export as rules:

```
RULE 1  --  covers 1,930 applicants (3.9% of the book)
  IF   average external credit score <= 0.395
  AND  combined external credit score <= 0.023
  AND  average external credit score <= 0.247
  THEN predicted default risk 31.2%  ->  High risk band
       observed default rate in this group: 30.7%
```

| Rule | Predicted | Observed | Coverage | Band |
|---|---|---|---|---|
| 1 | 31.2% | **30.7%** | 3.9% | High |
| 2 | 20.8% | **20.2%** | 3.4% | High |
| 3 | 20.6% | **19.4%** | 3.4% | High |

Predicted and observed agree to within a percentage point on every leaf — that
agreement is the evidence a rule means what it claims.

**Fidelity is reported with every rule set: R² 0.556, band agreement 73.1%.** Roughly
one applicant in four would be banded differently by the rules than by the model.
Stated plainly, because a surrogate that does not track the model is worse than no
surrogate — it looks authoritative while being wrong.

---

## 4 · Talk to your data

Ask in English; get an answer, the SQL behind it, and the rows.

> **"Which education level has the highest default rate?"**
> The highest default rate is for Lower secondary with a rate of 10.93%.

An LLM writes the SQL, so **the SQL is untrusted input**. Three independent layers
make that acceptable — **33 attacks attempted, 0 leaks, 0 false rejections.**

| Layer | What it does |
|---|---|
| **1 · SQL validator** | Parses with sqlglot. Single `SELECT` only; every write/DDL node rejected *anywhere* in the tree, including a `DELETE` hidden in a CTE; tables and columns whitelisted against the **live** schema; row cap injected. The statement that runs is **re-serialised from the parse tree**, never the model's string |
| **2 · Read-only role** | A PostgreSQL role with `SELECT` and nothing else. Verified live: `SELECT` works, `DELETE` and `DROP` are refused by the database itself. Statement timeout bounds anything pathological |
| **3 · Numeric grounding** | Every figure quoted in an answer must occur in the returned rows |
| **0 · Refusal** | *"What is the average credit card balance?"* → *"I can't answer that from the available data. There is no column in the schema that represents a credit card balance."* An admitted gap beats an invented column |

**Why layer 3 exists.** The most dangerous failure is not bad SQL — it is a fluent
summary containing an **invented number**, which looks exactly like a correct answer
and which no SQL validator can catch. Observed in testing:

> *"…approximately 0.08% of the total sample size (2,854 + 50 + 153 + 942 = **3,999**)."*

Every input real; the arithmetic and conclusion invented. Both fabricated figures
are caught, the summary discarded, and a truthful description shown instead.

### Prompt engineering and token cost

Templates are **versioned** (currently 1.7.0). **~1,700 tokens per turn**, with a
documented lever down to ~1,160.

| Technique | Why |
|---|---|
| **Curated schema, not a dump** | 5 narrow tables (~700 tokens). The raw 122-column application table alone would be several thousand — and every extra column widens the space of plausible-but-wrong names |
| **Engineered columns materialised** | `age_years`, `ext_source_mean` exist as real columns, so "average age of defaulters" maps directly instead of requiring invented arithmetic |
| **Few-shot over instructions** | Prose rules are followed inconsistently; a worked example showing `AVG(target) * 100` is imitated reliably. 9 examples, each teaching a distinct *question shape* |
| **Compact memory** | Last 6 turns, stored as summaries — question, SQL, row count — never the returned rows |

<details>
<summary><b>Rules added in response to observed failures, not speculation</b></summary>

- **Read extremes off the rows.** The model called Secondary "highest" at 8.83% when
  Lower secondary showed 16.00%, substituting the larger group.
- **Report what, never why.** It appended causal claims ("because it has the largest
  number of applicants (50)") that were factually backwards.
- **Never let NULL fall into a CASE ELSE.** Applicants who had never borrowed were
  being reported as having "always paid on time" — 15,868 people credited with good
  behaviour they had not earned.
- **Cast floats before ROUND.** PostgreSQL has no `ROUND(double precision, int)`.
  The rule was in the prompt *and* demonstrated in an example, and the model still
  reverted under context pressure — so the validator now rewrites it
  deterministically. **If a failure mode is deterministic, fix it deterministically
  rather than asking the model more firmly.**
</details>

---

## 5 · Audit trail

Every credit decision and every question is recorded to an append-only log —
including the **reasons given**, so a decision is reconstructable long after it was
made. Refusals and rejections are logged too: a blocked query is a security event,
and a refusal is evidence the grounding controls worked. Viewable in the **Audit**
tab.

---

## Design decisions

**Local LLM by default.** Talk-to-data sends schema context and query results —
derived from real applicant records — to a model. Running it on-box means that data
never leaves the host, which is the right posture for a credit system, and
`docker compose up` yields a working chatbot with no key, no billing account and no
signup. Hosted providers (OpenAI / Anthropic / Gemini) remain available behind the
same interface via `LLM_PROVIDER`.

*This is the one deviation from the suggested stack.* The trade is latency and some
answer quality for **data residency and zero setup friction** — for a credit-risk
system handling personal financial records, the right way round.

**Missingness is kept, not imputed.** For tree models an absent value is a usable
branch, and in credit data absence is informative — a thin bureau file and an
undisclosed occupation are both risk signals. Only the linear baseline imputes,
inside its own CV fold.

**Ratios, not raw amounts.** A 500k loan is routine on a 300k income and reckless on
a 60k one. Credit-to-income and instalment-to-income express leverage directly — and
are what a credit officer already reasons about, so explanations land in familiar
language.

**PR-AUC leads.** At an 8% positive rate, ROC-AUC is flattered by the large negative
class and accuracy is actively misleading. Reports always show the no-skill baseline
beside the score, so 0.268 is not read as "bad" without context — it is **3.3× the
no-skill baseline**.

**Charts are validated, not eyeballed.** The palette passes a colour-vision
validator (categorical pair CVD ΔE 24.7 against a floor of 8). No chart uses two
y-axes: aligning two scales invents a correlation that is not in the data.

<details>
<summary><b>Model size: measured, not assumed</b></summary>

A smaller local model would make the stack far lighter, so both were tested on the
same questions against the real database:

| | qwen2.5-coder:1.5b | qwen2.5-coder:7b |
|---|---|---|
| Size / latency | 986MB · **4.8s** | 4.7GB · 13.1s |
| "Highest-default education level?" | ❌ wrong group | ✅ correct |
| "Share with no credit history?" | ❌ inverted | ✅ correct |
| "Average credit card balance?" | ❌ **fabricated a figure** | ✅ refused |

The 1.5B is 2.7× faster and confidently wrong, so the **7B is the default**.

An initial pass scored both 6/6 and looked like a clear win for the small one. That
metric counted a question as passed if the pipeline *succeeded or refused* — not if
the answer was **right**. Reading the actual answers reversed the conclusion.
</details>

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q          # 302 tests
```

The whole suite runs with **no Kaggle data, no PostgreSQL, no model runtime and no
API key** — SQLite and a scripted LLM stand in, so CI needs nothing but Python.

| Area | Focus |
|---|---|
| `test_sql_validator.py` | 51 tests, mostly adversarial — every one asserts something unsafe is refused |
| `test_talk_to_data.py` | Query patterns, grounding, memory bounds, the repair loop |
| `test_train.py` | Selection logic, calibration, threshold tuning |
| `test_xai.py` | Surrogate fidelity — rules must track the model |
| `test_ui.py` | Every section renders, via Streamlit's `AppTest` — a live server returns HTTP 200 even when every section is raising |

Many tests are regression guards for bugs found during development, each documented
in place.

---

## Configuration

Everything is environment-driven; see **`.env.example`** for the complete annotated
list (a test asserts it stays in 1:1 sync with the settings object). The defaults
work with no edits.

| Variable | Default | Purpose |
|---|---|---|
| `DATA_MODE` | `sample` | `sample` fixtures or `real` Kaggle CSVs |
| `LLM_PROVIDER` | `ollama` | `ollama` (local, no key) / `openai` / `anthropic` / `gemini` / `none` |
| `LLM_MODEL` | `qwen2.5-coder:7b` | Chosen for text-to-SQL accuracy |
| `SQL_MAX_ROWS` | `200` | Hard `LIMIT` injected into every generated query |
| `SQL_TIMEOUT_SECONDS` | `15` | Server-side statement timeout |
| `MEMORY_MAX_TURNS` | `6` | Conversation turns retained |
| `USE_SQLITE_FALLBACK` | `false` | Run without PostgreSQL |
| `INCLUDE_*` | `true` | Which auxiliary tables to join |

Selecting a hosted provider without its key never crashes the app: the Chat tab
explains what is missing and every other feature keeps working.

---

## Limitations and next steps

**Known limitations**

1. **Calibration is measured in-sample.** The isotonic fit uses the same out-of-fold
   predictions the reliability curve is drawn from. A held-out set would measure it
   honestly.
2. **The deployed demo runs on synthetic fixtures.** The real 2.5GB CSVs cannot be
   committed. It still scores with the model trained on all 307,511 real rows, and
   every figure in this README comes from that model — but locally-generated EDA
   charts will differ from the committed ones.
3. **The surrogate is an approximation.** R² 0.556 means roughly one applicant in
   four is banded differently by the rules than by the model.
4. **Three tables excluded on structural judgement**, not measured the way the other
   three were.
5. **No fairness audit.** Gender and family status are legally sensitive in lending;
   disparate-impact testing is required before production use.

**Next, in value order**

1. A held-out calibration split, and calibration drift monitoring.
2. Fairness metrics per protected attribute, with reject-option post-processing.
3. A labelled question→SQL evaluation set scored on every prompt change, turning
   prompt edits into a measurable experiment.
4. Model monitoring — PSI on feature distributions, alerting on score drift.
5. Authentication and per-user query audit for the chat.

---

## Project layout

```
credit_risk_platform/
├── data/sample/                 # committed synthetic fixtures (real CSVs gitignored)
├── documents/                   # project_presentation.pdf
├── notebooks/                   # eda.ipynb (presentation) + eda.py (implementation)
├── src/
│   ├── data/                    # loader, preprocessor, db_loader, fixture generator
│   ├── ml/                      # train (bake-off), evaluate, predict, ablation
│   ├── xai/                     # shap_explainer, rules, feature_labels
│   ├── talk_to_data/            # nl_to_sql, sql_validator, query_runner,
│   │                            # prompt_templates, llm_client, memory
│   ├── ui/app.py                # Streamlit application
│   └── utils/                   # config, logger, helpers, viz, audit, docker_utils
├── sql/schema.sql               # analytics schema
├── docker/                      # entrypoint, read-only role init
├── tests/                       # 302 tests
├── models/                      # trained artifacts (committed)
├── reports/                     # figures and metrics (committed)
├── Dockerfile · docker-compose.yml · docker-compose.lite.yml
└── requirements.txt · requirements-dev.txt · requirements-serve.txt
```
