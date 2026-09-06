"""Versioned prompt templates for natural-language to SQL.

Three techniques carry most of the accuracy here, and all three are about
constraining the model rather than persuading it:

**1. A compact, curated schema.** The prompt describes four narrow tables with
short semantic notes, not a 122-column dump. Every column costs tokens on every
turn and widens the space of plausible-but-wrong names, so the schema the model
sees is the schema an analyst actually needs. The rendered description is about
700 tokens; the raw application table alone would be several thousand.

**2. Few-shot patterns over instructions.** Prose rules like "remember TARGET is
a flag" are followed inconsistently. A worked example showing
``AVG(target) * 100`` teaches the same thing and is imitated reliably. The eight
patterns below were chosen to cover the question *shapes* an analyst asks --
rate by segment, distribution, comparison, ranking, join to model output, data
quality -- rather than specific questions.

**3. Explicit grounding rules.** The model is told to refuse rather than invent:
if a question cannot be answered from these columns, say so. That converts a
hallucination into a clear message, which the validator would otherwise have to
catch as an unknown-column rejection.

Templates are versioned so a prompt change is a reviewable, revertable event
rather than an untracked edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

PROMPT_VERSION: Final[str] = "1.6.0"

# --------------------------------------------------------------------------- #
# Schema description
# --------------------------------------------------------------------------- #
# Hand-written rather than generated from information_schema: the semantic notes
# ("negative offset", "1 = defaulted") are what stop the model misreading a
# column, and those cannot be derived from types alone.
SCHEMA_DESCRIPTION: Final[str] = """\
TABLE applications  -- one row per loan applicant (the main table)
  sk_id_curr INT PK, target SMALLINT (1=defaulted, 0=repaid; NULL if unlabelled)
  amt_income_total, amt_credit, amt_annuity, amt_goods_price  FLOAT (money)
  credit_to_income_ratio, annuity_to_income_ratio, payment_rate  FLOAT (affordability)
  age_years FLOAT, employed_years FLOAT (NULL when no employment record)
  days_employed_anomaly SMALLINT (1 = pensioner/unemployed, no employment record)
  ext_source_1, ext_source_2, ext_source_3, ext_source_mean  FLOAT 0-1
      (external credit scores; HIGHER = SAFER; strongest predictor of default)
  name_contract_type ('Cash loans','Revolving loans')
  code_gender ('M','F'), flag_own_car ('Y','N'), flag_own_realty ('Y','N')
  name_education_type ('Higher education','Secondary / secondary special',
      'Incomplete higher','Lower secondary','Academic degree')
  name_income_type ('Working','Commercial associate','Pensioner','State servant',...)
  name_family_status ('Married','Single / not married','Civil marriage','Separated','Widow')
  name_housing_type, occupation_type, organization_type  VARCHAR
  cnt_children INT, cnt_fam_members FLOAT, region_rating_client INT (1-3)
  credit_enquiry_total FLOAT (recent credit bureau enquiries)

TABLE bureau  -- one row per PRIOR credit held with another institution
  sk_id_bureau INT PK, sk_id_curr INT FK -> applications.sk_id_curr
  credit_active ('Active','Closed'), credit_type VARCHAR
  days_credit INT (negative = days before application)
  credit_day_overdue INT (days past due, positive)
  amt_credit_sum, amt_credit_sum_debt, amt_credit_sum_overdue FLOAT
  cnt_credit_prolong INT

TABLE bureau_summary  -- bureau rolled up to one row per applicant (all applicants)
  sk_id_curr INT PK FK -> applications.sk_id_curr
  bureau_loan_count, bureau_active_count, bureau_closed_count INT
  bureau_credit_sum_total, bureau_debt_total FLOAT
  bureau_debt_credit_ratio FLOAT (share of external credit still outstanding)
  bureau_days_overdue_max FLOAT, bureau_overdue_loan_count INT
  bureau_has_overdue SMALLINT (1 = has prior arrears)
  bureau_has_history SMALLINT (0 = thin file, no external credit record at all)

TABLE credit_behaviour  -- prior conduct with THIS lender (repayment behaviour)
  sk_id_curr INT PK FK -> applications.sk_id_curr
  prev_application_count, prev_refused_count INT
  prev_refused_rate FLOAT (share of prior applications this lender declined)
  prev_ever_refused SMALLINT (1 = declined at least once before)
  prev_credit_to_application FLOAT (granted / requested; below 1 = cut back)
  prev_avg_credit_granted FLOAT
  instalments_paid_count INT
  avg_days_past_due, worst_days_past_due FLOAT (positive = paid late)
  late_payment_count INT, late_payment_rate FLOAT (share of instalments paid late)
  ever_paid_late SMALLINT (1 = has ever paid an instalment late)
  avg_payment_ratio FLOAT (paid / owed; below 1 = underpaid)
  underpaid_rate FLOAT, total_shortfall FLOAT
  NOTE: not every applicant has borrowed here before; rows are absent for those.

TABLE predictions  -- model output, one row per applicant
  sk_id_curr INT PK FK -> applications.sk_id_curr
  probability_of_default FLOAT 0-1 (calibrated)
  risk_score FLOAT 0-1000 (higher = riskier)
  risk_band VARCHAR ('Low','Medium','High')
  decision VARCHAR ('Approve','Refer for review')"""


@dataclass(frozen=True)
class FewShotExample:
    """One worked question-to-SQL pair."""

    question: str
    sql: str
    teaches: str  # why this example is in the prompt, for maintainers


# Each example is here to teach a distinct query *shape*. Adding a ninth should
# require naming a shape none of these cover.
FEW_SHOT_EXAMPLES: Final[tuple[FewShotExample, ...]] = (
    FewShotExample(
        question="What is the overall default rate?",
        sql=(
            "SELECT ROUND(AVG(target) * 100, 2) AS default_rate_pct,\n"
            "       COUNT(*) AS total_applicants\n"
            "FROM applications\n"
            "WHERE target IS NOT NULL"
        ),
        teaches="target is a 0/1 flag, so AVG*100 is the rate; always exclude NULL labels",
    ),
    FewShotExample(
        question="Which education level has the highest default rate?",
        sql=(
            "SELECT name_education_type,\n"
            "       COUNT(*) AS applicants,\n"
            "       ROUND(AVG(target) * 100, 2) AS default_rate_pct\n"
            "FROM applications\n"
            "WHERE target IS NOT NULL\n"
            "GROUP BY name_education_type\n"
            "HAVING COUNT(*) >= 50\n"
            "ORDER BY default_rate_pct DESC"
        ),
        teaches="rate by segment; always return the group size, and suppress tiny groups",
    ),
    FewShotExample(
        question="Compare average income between applicants who defaulted and those who repaid.",
        sql=(
            "SELECT CASE WHEN target = 1 THEN 'Defaulted' ELSE 'Repaid' END AS outcome,\n"
            "       COUNT(*) AS applicants,\n"
            "       ROUND(AVG(amt_income_total)::numeric, 0) AS avg_income,\n"
            "       ROUND(AVG(amt_credit)::numeric, 0) AS avg_loan\n"
            "FROM applications\n"
            "WHERE target IS NOT NULL\n"
            "GROUP BY target"
        ),
        teaches="two-group comparison via CASE; label the groups readably",
    ),
    FewShotExample(
        question="How does the default rate vary across external credit score bands?",
        sql=(
            "SELECT CASE\n"
            "         WHEN ext_source_mean < 0.3 THEN '1. Very low (<0.3)'\n"
            "         WHEN ext_source_mean < 0.45 THEN '2. Low (0.3-0.45)'\n"
            "         WHEN ext_source_mean < 0.6 THEN '3. Medium (0.45-0.6)'\n"
            "         ELSE '4. High (>=0.6)'\n"
            "       END AS score_band,\n"
            "       COUNT(*) AS applicants,\n"
            "       ROUND(AVG(target) * 100, 2) AS default_rate_pct\n"
            "FROM applications\n"
            "WHERE target IS NOT NULL AND ext_source_mean IS NOT NULL\n"
            "GROUP BY score_band\n"
            "ORDER BY score_band"
        ),
        teaches="banding a continuous column; numeric label prefixes keep the ordering right",
    ),
    FewShotExample(
        question="What does the model's risk banding look like, and is it accurate?",
        sql=(
            "SELECT p.risk_band,\n"
            "       COUNT(*) AS applicants,\n"
            "       ROUND(AVG(p.probability_of_default)::numeric * 100, 2) AS predicted_default_pct,\n"
            "       ROUND(AVG(a.target) * 100, 2) AS actual_default_pct\n"
            "FROM predictions p\n"
            "JOIN applications a ON a.sk_id_curr = p.sk_id_curr\n"
            "WHERE a.target IS NOT NULL\n"
            "GROUP BY p.risk_band\n"
            "ORDER BY predicted_default_pct"
        ),
        teaches=(
            "joining model output to outcomes; and the ::numeric cast that ROUND "
            "requires on a float column"
        ),
    ),
    FewShotExample(
        question="Do applicants with prior arrears default more often?",
        sql=(
            "SELECT CASE WHEN b.bureau_has_overdue = 1 THEN 'Prior arrears'\n"
            "            ELSE 'No arrears' END AS credit_history,\n"
            "       COUNT(*) AS applicants,\n"
            "       ROUND(AVG(a.target) * 100, 2) AS default_rate_pct\n"
            "FROM applications a\n"
            "JOIN bureau_summary b ON b.sk_id_curr = a.sk_id_curr\n"
            "WHERE a.target IS NOT NULL AND b.bureau_has_history = 1\n"
            "GROUP BY b.bureau_has_overdue"
        ),
        teaches="joining to bureau_summary; filter to applicants who actually have a history",
    ),
    FewShotExample(
        question="Show me the 10 riskiest applicants the model flagged for review.",
        sql=(
            "SELECT p.sk_id_curr,\n"
            "       p.risk_score,\n"
            "       p.risk_band,\n"
            "       a.amt_credit,\n"
            "       a.amt_income_total,\n"
            "       a.ext_source_mean\n"
            "FROM predictions p\n"
            "JOIN applications a ON a.sk_id_curr = p.sk_id_curr\n"
            "WHERE p.decision = 'Refer for review'\n"
            "ORDER BY p.risk_score DESC\n"
            "LIMIT 10"
        ),
        teaches="row-level ranking with a filter; an explicit LIMIT for top-N questions",
    ),
    FewShotExample(
        question="Do applicants who paid late on previous loans default more often?",
        sql=(
            "SELECT CASE WHEN c.ever_paid_late = 1 THEN 'Paid late before'\n"
            "            ELSE 'Always paid on time' END AS repayment_history,\n"
            "       COUNT(*) AS applicants,\n"
            "       ROUND(AVG(a.target) * 100, 2) AS default_rate_pct,\n"
            "       ROUND(AVG(c.late_payment_rate)::numeric * 100, 2) AS avg_late_instalment_pct\n"
            "FROM applications a\n"
            "JOIN credit_behaviour c ON c.sk_id_curr = a.sk_id_curr\n"
            "WHERE a.target IS NOT NULL\n"
            "GROUP BY c.ever_paid_late"
        ),
        teaches="the repayment-behaviour table; joining it, and casting a float before ROUND",
    ),
    FewShotExample(
        question="How much data is missing for the external credit scores?",
        sql=(
            "SELECT COUNT(*) AS total_rows,\n"
            "       COUNT(ext_source_1) AS ext_source_1_present,\n"
            "       ROUND(100.0 * (COUNT(*) - COUNT(ext_source_1)) / COUNT(*), 2) AS ext_1_missing_pct,\n"
            "       ROUND(100.0 * (COUNT(*) - COUNT(ext_source_3)) / COUNT(*), 2) AS ext_3_missing_pct\n"
            "FROM applications"
        ),
        teaches="data-quality questions; COUNT(col) skips NULLs while COUNT(*) does not",
    ),
)


SYSTEM_PROMPT: Final[str] = """\
You are a careful SQL analyst for a credit-risk platform. You translate an \
analyst's question into ONE PostgreSQL SELECT query.

HARD RULES -- these are enforced by a validator that will reject your output:
1. Output exactly ONE SELECT statement. Never INSERT, UPDATE, DELETE, DROP, \
CREATE, ALTER, GRANT or TRUNCATE. Never multiple statements.
2. Use ONLY the tables and columns listed in the schema below. Never invent a \
column name. If the question cannot be answered from these columns, do not \
guess -- reply with exactly: CANNOT_ANSWER: <short reason>.
3. Return SQL only. No prose, no explanation, no markdown fences.

QUERY GUIDANCE:
- target is 1 for default and 0 for repaid, so a default RATE is \
ROUND(AVG(target) * 100, 2). Always add WHERE target IS NOT NULL.
- Always include COUNT(*) beside any rate, so the reader can see the group size.
- Suppress tiny groups with HAVING COUNT(*) >= 50 when grouping by a category.
- Higher ext_source_* means SAFER, not riskier.
- days_credit is a negative offset in days; credit_day_overdue is a positive \
number of days past due.
- Round money to whole units and rates to 2 decimal places.
- PostgreSQL has no ROUND(double precision, int). When rounding a FLOAT column \
to decimal places, cast first: ROUND(AVG(ext_source_mean)::numeric, 2). Columns \
typed INT or SMALLINT (such as target) need no cast.
- Give every computed column a readable alias.
- Add LIMIT for row-level questions; aggregates do not need one."""


def render_schema() -> str:
    """Return the compact schema description sent to the model."""
    return SCHEMA_DESCRIPTION


def render_few_shots(limit: int | None = None) -> str:
    """Render the worked examples as prompt text.

    Args:
        limit: Use only the first ``limit`` examples. Lowering this is the main
            token lever available at runtime.

    Returns:
        The formatted example block.
    """
    examples = FEW_SHOT_EXAMPLES[:limit] if limit else FEW_SHOT_EXAMPLES
    return "\n\n".join(
        f"Question: {example.question}\nSQL:\n{example.sql}" for example in examples
    )


def build_sql_prompt(
    question: str,
    schema: str | None = None,
    history: str = "",
    few_shot_limit: int | None = None,
) -> tuple[str, str]:
    """Assemble the system and user prompts for SQL generation.

    Args:
        question: The analyst's natural-language question.
        schema: Override the schema description (used when reading the live
            database schema instead of the curated text).
        history: Compact rendering of recent conversation turns, for follow-ups.
        few_shot_limit: Cap the number of worked examples included.

    Returns:
        ``(system_prompt, user_prompt)``.
    """
    sections = [
        "DATABASE SCHEMA:",
        schema or render_schema(),
        "",
        "WORKED EXAMPLES:",
        render_few_shots(few_shot_limit),
    ]
    if history:
        sections += [
            "",
            "RECENT CONVERSATION (for resolving follow-ups like 'and for women?'):",
            history,
        ]
    sections += ["", f"Question: {question}", "SQL:"]
    return SYSTEM_PROMPT, "\n".join(sections)


SUMMARY_SYSTEM_PROMPT: Final[str] = """\
You are a credit-risk analyst summarising a query result for a colleague.

GROUNDING RULES -- these matter more than fluency:
1. State ONLY what the rows below show. Never add a number that is not in them.
2. Report WHAT the numbers show, never WHY. You cannot know the cause, so do \
not write "because", "due to", "driven by", or any explanation of a figure. \
Stating a cause you cannot see in the rows is the single most damaging error \
you can make here.
3. If the result is empty, say plainly that no rows matched. Do not invent a reason.
4. Quote figures exactly as given; do not re-round or recompute them.
5. If the question asks which group is highest, lowest, best or worst, the \
answer is the row holding that extreme value. Read it off the rows. Do NOT \
substitute a different group because it has more applicants -- you may add the \
small sample size as a caveat afterwards, but the direct answer comes first.
6. Two to four sentences. Lead with the direct answer, then the most useful \
supporting detail. Mention group sizes when they affect how much weight the \
finding deserves.
7. Write plainly for a business reader. No preamble such as "Based on the data"."""


def build_summary_prompt(question: str, sql: str, rows_markdown: str, row_count: int) -> tuple[str, str]:
    """Assemble the prompts for summarising a result set.

    The rows are handed over verbatim and the model is told to use nothing else,
    which is what keeps the summary anchored to real numbers.

    Args:
        question: The original question.
        sql: The SQL that was executed.
        rows_markdown: The returned rows rendered as a markdown table.
        row_count: Number of rows returned.

    Returns:
        ``(system_prompt, user_prompt)``.
    """
    user = (
        f"Question: {question}\n\n"
        f"SQL executed:\n{sql}\n\n"
        f"Result ({row_count} row{'s' if row_count != 1 else ''}):\n{rows_markdown}\n\n"
        "Answer the question using only these rows."
    )
    return SUMMARY_SYSTEM_PROMPT, user
