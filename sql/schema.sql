-- ===========================================================================
-- Credit Risk Intelligence Platform -- analytics schema
--
-- This schema is deliberately NOT a dump of the raw Home Credit files. The
-- application table has 122 columns, most of them optional property attributes
-- that no credit analyst asks about, and feeding all of them to an LLM as
-- schema context would cost tokens on every turn while making the model more
-- likely to hallucinate a column.
--
-- Instead the loader materialises a curated analytics layer:
--   * applications    -- the ~35 columns analysts actually query, including the
--                        engineered ones (AGE_YEARS, EXT_SOURCE_MEAN, ratios)
--                        so questions map onto columns in natural language.
--   * bureau          -- raw external credit records, one row per prior credit.
--   * bureau_summary  -- applicant-level aggregates of the above.
--   * predictions     -- model output, so the chat can answer questions about
--                        scores and risk bands, not just raw applicant data.
--
-- Every table is queried through a READ-ONLY role. See sql/init_readonly.sql.
-- ===========================================================================

DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS bureau_summary;
DROP TABLE IF EXISTS bureau;
DROP TABLE IF EXISTS applications;

-- ---------------------------------------------------------------------------
-- Applicants and their loan applications
-- ---------------------------------------------------------------------------
CREATE TABLE applications (
    sk_id_curr                  INTEGER PRIMARY KEY,
    target                      SMALLINT,            -- 1 = defaulted, 0 = repaid
    name_contract_type          VARCHAR(32),
    code_gender                 VARCHAR(8),
    flag_own_car                VARCHAR(4),
    flag_own_realty             VARCHAR(4),
    cnt_children                INTEGER,
    cnt_fam_members             DOUBLE PRECISION,
    amt_income_total            DOUBLE PRECISION,
    amt_credit                  DOUBLE PRECISION,
    amt_annuity                 DOUBLE PRECISION,
    amt_goods_price             DOUBLE PRECISION,
    name_income_type            VARCHAR(64),
    name_education_type         VARCHAR(64),
    name_family_status          VARCHAR(64),
    name_housing_type           VARCHAR(64),
    occupation_type             VARCHAR(64),
    organization_type           VARCHAR(64),
    region_rating_client        INTEGER,
    age_years                   DOUBLE PRECISION,    -- derived from DAYS_BIRTH
    employed_years              DOUBLE PRECISION,    -- NULL when no record exists
    days_employed_anomaly       SMALLINT,            -- 1 = pensioner/unemployed sentinel
    ext_source_1                DOUBLE PRECISION,
    ext_source_2                DOUBLE PRECISION,
    ext_source_3                DOUBLE PRECISION,
    ext_source_mean             DOUBLE PRECISION,
    credit_to_income_ratio      DOUBLE PRECISION,
    annuity_to_income_ratio     DOUBLE PRECISION,
    payment_rate                DOUBLE PRECISION,
    credit_enquiry_total        DOUBLE PRECISION
);

CREATE INDEX idx_applications_target       ON applications (target);
CREATE INDEX idx_applications_education    ON applications (name_education_type);
CREATE INDEX idx_applications_income_type  ON applications (name_income_type);
CREATE INDEX idx_applications_ext_mean     ON applications (ext_source_mean);

-- ---------------------------------------------------------------------------
-- External credit records (one row per prior credit with another institution)
-- ---------------------------------------------------------------------------
CREATE TABLE bureau (
    sk_id_bureau                INTEGER PRIMARY KEY,
    sk_id_curr                  INTEGER NOT NULL,
    credit_active               VARCHAR(16),         -- Active | Closed | Sold | Bad debt
    credit_type                 VARCHAR(64),
    days_credit                 INTEGER,             -- negative offset from application
    credit_day_overdue          INTEGER,             -- days past due (positive)
    amt_credit_sum              DOUBLE PRECISION,
    amt_credit_sum_debt         DOUBLE PRECISION,
    amt_credit_sum_overdue      DOUBLE PRECISION,
    cnt_credit_prolong          INTEGER
);

CREATE INDEX idx_bureau_applicant ON bureau (sk_id_curr);
CREATE INDEX idx_bureau_active    ON bureau (credit_active);

-- ---------------------------------------------------------------------------
-- Applicant-level roll-up of the bureau table
-- ---------------------------------------------------------------------------
CREATE TABLE bureau_summary (
    sk_id_curr                  INTEGER PRIMARY KEY,
    bureau_loan_count           INTEGER,
    bureau_active_count         INTEGER,
    bureau_closed_count         INTEGER,
    bureau_credit_sum_total     DOUBLE PRECISION,
    bureau_debt_total           DOUBLE PRECISION,
    bureau_debt_credit_ratio    DOUBLE PRECISION,    -- share still outstanding
    bureau_days_overdue_max     DOUBLE PRECISION,
    bureau_overdue_loan_count   INTEGER,
    bureau_has_overdue          SMALLINT,            -- 1 = has prior arrears
    bureau_has_history          SMALLINT             -- 0 = thin file, no record
);

-- ---------------------------------------------------------------------------
-- Model output, so the chat can answer questions about scores and bands
-- ---------------------------------------------------------------------------
CREATE TABLE predictions (
    sk_id_curr                  INTEGER PRIMARY KEY,
    probability_of_default      DOUBLE PRECISION,    -- calibrated, 0..1
    risk_score                  DOUBLE PRECISION,    -- 0..1000, higher = riskier
    risk_band                   VARCHAR(8),          -- Low | Medium | High
    decision                    VARCHAR(32)          -- Approve | Refer for review
);

CREATE INDEX idx_predictions_band ON predictions (risk_band);
