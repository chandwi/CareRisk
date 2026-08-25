# CareRisk

A decision-analytics prototype: predicting 30-day hospital readmission risk and optimizing how a hospital allocates *limited* care-management intervention capacity across patients.

Built as a portfolio project targeting analytics/decision-science roles (predictive modeling → explainability → resource optimization → measurable business impact), not a plain "trained a classifier" exercise.

## The question

Not just *"who will be readmitted?"* — but:

> Given only enough capacity to intervene on N patients, which N should get the intervention, and what's the expected benefit of that choice versus simpler strategies (random, highest-risk-only)?

## Dataset

[UCI Diabetes 130-US hospitals (1999-2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) — 101,766 encounters, 130 hospitals, 47 features. See [`data/README.md`](data/README.md) for setup and the full data-quality audit summary.

**Honesty note:** this data is 1999-2008 and has no ground-truth intervention outcomes. This is a decision-analytics *prototype and simulation*, not a production readmission model or a causal treatment-effect study — assumptions about intervention cost/effectiveness are explicit and swept via sensitivity analysis, never presented as measured fact.

## Pipeline

```
UCI EHR data → data quality audit → feature engineering → statistical analysis
  → baseline (logistic regression) → tree models (RF / XGBoost) → calibration
  → SHAP explainability → risk scoring
  → decision simulator (cost / effectiveness / capacity assumptions)
  → resource optimization (who gets the intervention, under a capacity constraint)
  → strategy comparison (random vs. highest-risk vs. utility-optimized)
  → fairness audit across demographic subgroups
  → what-if scenario app (Streamlit)
```

## Structure

```
CareRisk/
├── data/                  raw/processed data (gitignored — see data/README.md)
├── notebooks/             01_data_audit → 10_fairness_analysis
├── sql/                   cohort/feature queries
├── src/                   preprocessing.py, features.py, models.py, evaluation.py,
│                          explainability.py, optimization.py
├── app/                   streamlit_app.py
├── reports/               executive_summary.pdf
└── requirements.txt
```

## Status

- [x] Data acquisition + full column-by-column audit ([`notebooks/01_data_audit.ipynb`](notebooks/01_data_audit.ipynb))
- [x] Cleaning decisions implemented ([`src/preprocessing.py`](src/preprocessing.py))
- [ ] EDA
- [ ] Statistical analysis (hypothesis tests, logistic regression baseline)
- [ ] Feature engineering (utilization score, medication complexity, diagnosis grouping)
- [ ] Model comparison (Logistic Regression / Random Forest / XGBoost)
- [ ] Calibration + SHAP explainability
- [ ] Decision simulator + optimization (capacity-constrained allocation)
- [ ] Strategy comparison (random / highest-risk / utility-optimized)
- [ ] Fairness audit
- [ ] Streamlit app

## Setup

```bash
pip install -r requirements.txt
```

Then follow [`data/README.md`](data/README.md) to get the raw CSVs into `data/raw/`.
