-- Cohort analysis against the `encounters` table (data/processed/diabetic_data_features.csv,
-- loaded via `python sql/load_db.py`). SQLite dialect; portable to Postgres with trivial changes
-- (e.g. CAST syntax) — see data/README.md for why Postgres was the original stack choice.

-- 1. Readmission rate by prior-utilization cohort — the single strongest driver
-- (confirmed independently in 02_eda / 03_statistical_analysis / 07_shap_analysis).
SELECT
    CASE
        WHEN number_inpatient = 0 THEN '0 prior inpatient visits'
        WHEN number_inpatient BETWEEN 1 AND 2 THEN '1-2 prior inpatient visits'
        ELSE '3+ prior inpatient visits'
    END AS utilization_cohort,
    COUNT(*) AS n_encounters,
    ROUND(AVG(readmitted_30d) * 100, 1) AS readmission_rate_pct
FROM encounters
GROUP BY utilization_cohort
ORDER BY readmission_rate_pct DESC;

-- 2. Readmission rate by age band and diagnosis group — where does risk concentrate?
SELECT
    age,
    diag_1_group,
    COUNT(*) AS n_encounters,
    ROUND(AVG(readmitted_30d) * 100, 1) AS readmission_rate_pct
FROM encounters
GROUP BY age, diag_1_group
HAVING COUNT(*) >= 100  -- drop cells too small to trust
ORDER BY readmission_rate_pct DESC
LIMIT 20;

-- 3. High-utilizer, high-comorbidity cohort — the population the intervention-allocation
-- model (src/optimization.py) is ultimately trying to prioritize.
SELECT
    COUNT(*) AS n_encounters,
    ROUND(AVG(readmitted_30d) * 100, 1) AS readmission_rate_pct,
    ROUND(AVG(number_diagnoses), 1) AS avg_diagnoses,
    ROUND(AVG(num_medications), 1) AS avg_medications
FROM encounters
WHERE high_utilizer = 1 AND comorbidity_burden >= 9;

-- 4. Discharge disposition breakdown — sanity check for the leakage exclusion documented
-- in 01_data_audit.ipynb (expired-discharge encounters should be entirely absent here).
SELECT
    discharge_disposition_id,
    COUNT(*) AS n_encounters,
    ROUND(AVG(readmitted_30d) * 100, 1) AS readmission_rate_pct
FROM encounters
GROUP BY discharge_disposition_id
ORDER BY n_encounters DESC
LIMIT 10;
