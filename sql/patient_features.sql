-- Patient-level rollup: resolves the 100,114 encounters down to one row per patient
-- (23% of patients have repeat encounters — see 01_data_audit.ipynb), taking each
-- patient's most recent encounter as the representative row. This is the unit-of-action
-- table a real deployment would score and allocate intervention capacity against —
-- the decision is made about a *patient*, not an individual encounter.

CREATE TEMP VIEW ranked_encounters AS
SELECT
    e.*,
    ROW_NUMBER() OVER (
        PARTITION BY patient_nbr
        ORDER BY encounter_id DESC
    ) AS encounter_rank
FROM encounters e;

SELECT
    patient_nbr,
    encounter_id AS most_recent_encounter_id,
    age,
    race,
    gender,
    number_inpatient,
    number_emergency,
    number_outpatient,
    prior_utilization,
    high_utilizer,
    number_diagnoses,
    num_medications,
    medication_complexity,
    readmitted_30d AS most_recent_encounter_readmitted_30d
FROM ranked_encounters
WHERE encounter_rank = 1;
