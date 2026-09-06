# Use-case presentation — what to produce

> **This file is a placeholder.** The finished presentation must be exported to
> PDF and saved in this directory as:
>
> ```
> documents/project_presentation.pdf
> ```
>
> That exact path and filename is a hard submission requirement. Delete this
> placeholder once the PDF is in place.

---

## Purpose

A short deck (10–14 slides) explaining the platform to a mixed audience: a credit
risk owner who cares about the decision, and an engineer who cares about how it
works. It should stand alone — a reader who never runs the code should still
understand what was built, why, and how well it performs.

---

## Required contents

### 1. Problem (1–2 slides)
- The business problem: unsecured lending to applicants with thin or absent
  credit histories, where a wrong approval costs far more than a wrong decline.
- Why it is hard: only ~8.5% of applicants default, so accuracy is meaningless
  and the metric choice drives everything downstream.
- What "good" looks like: a *calibrated* probability, an explanation the
  applicant is entitled to, and a policy a credit committee can audit.

### 2. Data (1–2 slides)
- Home Credit Default Risk: 307,511 applicants, 122 application columns, plus
  six auxiliary tables.
- **Scope decision:** application + aggregated `bureau` only. Say why — bureau
  carries external credit history, the other five buy little signal for a lot of
  opacity and runtime.
- Note the synthetic fixture: schema-identical, so the whole pipeline is
  demonstrable without Kaggle credentials.

### 3. EDA highlights (2–3 slides)
Pick three or four, each as one chart plus one sentence. Use the figures already
generated in `reports/figures/`:

| Figure | Point it makes |
|---|---|
| `01_class_imbalance.png` | 8.5% default rate — why PR-AUC, not accuracy |
| `04_external_scores.png` | 17.5% → 1.9% across score bands; the dominant signal |
| `06_employment_anomaly.png` | The 365243 sentinel is a pensioner population, not an error |
| `08_credit_history.png` | Prior arrears 15.4% vs 7.6% — the most actionable lever |

### 4. Architecture (1 slide)
Reuse the diagram from the README. Emphasise the three containers and the fact
that the language model runs locally, so no applicant data leaves the host.

### 5. Modelling approach (2 slides)
- The three-way bake-off and the **selection rule**: PR-AUC first, and anything
  within one standard error treated as tied and decided on explainability.
  Show the comparison table.
- Class imbalance by weighting, and **why not SMOTE** — it biases the
  probabilities, and a calibrated probability is the product.
- Calibration before and after (Brier 0.137 → 0.067), with the in-sample caveat.
- Thresholds tuned, never 0.5: **112 defaults caught vs 32**.

### 6. Explainability (2 slides)
- SHAP for the individual: use `14_shap_local_explanation.png`.
- The surrogate tree for policy: show two or three rules verbatim, and state the
  fidelity (R² 0.613, 78.2% band agreement) on the same slide. Fidelity is the
  point — a surrogate without it is decoration.

### 7. Talk-to-data (2 slides)
- One screenshot of a real question, answer, and the SQL behind it.
- The three safety layers, and that **18 injection and hallucination attacks
  were attempted with 0 leaks**.
- The grounding check, illustrated with the real observed failure
  ("…0.08% of the total sample (2,854 + 50 + 153 + 942 = 3,999)") — a fluent,
  entirely invented statistic that the SQL validator could not have caught.

### 8. Results (1 slide)
PR-AUC 0.308 (3.6× the no-skill baseline), ROC-AUC 0.801, Brier 0.067, and the
band table: Low 58% of the book at 2.6% default, High 7.8% at 36.1%.

### 9. Limitations and next steps (1 slide)
Be direct. In-sample calibration measurement, synthetic-data caveat, no fairness
audit yet. Naming real limitations reads as competence, not weakness.

---

## Screenshots to include

Capture these from the running app (`docker-compose up`, then
http://localhost:8501):

1. **Overview** — portfolio metrics and one insight chart.
2. **Predict** — an applicant scored, showing band, probability and action.
3. **Explain** — the SHAP diverging chart for that same applicant.
4. **Rules** — the rules table with the fidelity metrics visible.
5. **Chat** — a question, its answer, and the expanded SQL panel.

Using the *same applicant* across slides 2 and 3 makes the story continuous:
here is the score, and here is exactly why.

---

## Presentation tips

- Lead every slide with the finding, not the method. "Prior arrears doubles
  default risk" beats "we joined the bureau table".
- Show the honest numbers. The in-sample calibration caveat and the synthetic
  data note *strengthen* the deck — they show the work was checked rather than
  reported at face value.
- The bugs found and fixed (LightGBM stopping at iteration 5, the grounding
  false positive) make good speaker notes: they demonstrate the engineering was
  verified rather than assumed.
