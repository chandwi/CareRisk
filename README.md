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
├── sql/                   load_db.py + cohort_analysis.sql, patient_features.sql (SQLite)
├── src/                   preprocessing.py, features.py, rare_category.py, models.py,
│                          evaluation.py, explainability.py, optimization.py, fairness.py
├── app/                   streamlit_app.py — 5-tab decision cockpit
├── reports/               model_card.md
└── requirements.txt
```

## Status — all phases complete

- [x] Data acquisition + full column-by-column audit ([`01_data_audit`](notebooks/01_data_audit.ipynb))
- [x] EDA ([`02_eda`](notebooks/02_eda.ipynb))
- [x] Statistical analysis — chi-square, Mann-Whitney, logistic regression odds ratios ([`03_statistical_analysis`](notebooks/03_statistical_analysis.ipynb))
- [x] Feature engineering — utilization score, medication complexity, diagnosis grouping ([`04_feature_engineering`](notebooks/04_feature_engineering.ipynb))
- [x] Model comparison — Logistic Regression / Random Forest / XGBoost, group-aware split ([`05_model_training`](notebooks/05_model_training.ipynb), [`06_model_evaluation`](notebooks/06_model_evaluation.ipynb))
- [x] SHAP explainability, incl. a real rare-category noise issue found and fixed ([`07_shap_analysis`](notebooks/07_shap_analysis.ipynb))
- [x] Decision simulator ([`08_decision_simulation`](notebooks/08_decision_simulation.ipynb))
- [x] Optimization — LP-verified capacity-constrained allocation, risk-vs-benefit sensitivity sweep ([`09_optimization`](notebooks/09_optimization.ipynb))
- [x] Fairness audit ([`10_fairness_analysis`](notebooks/10_fairness_analysis.ipynb))
- [x] Streamlit decision cockpit ([`app/streamlit_app.py`](app/streamlit_app.py))

See [`reports/model_card.md`](reports/model_card.md) for a one-page summary of performance, key findings, and limitations.

## Headline results

- **XGBoost**: ROC-AUC 0.668, PR-AUC 0.227 (test set, n=20,075) — beats Logistic Regression and Random Forest baselines, in line with published results on this exact dataset.
- **Top driver**: prior inpatient visits (`number_inpatient`) — confirmed independently by EDA, statistical effect size, and SHAP.
- **Capacity-constrained targeting beats random by ~4.5x** in expected net benefit at the same capacity (500 patients).
- **Risk ≠ benefit — but only sometimes helps**: under an assumed effectiveness-varies-by-complexity scenario, utility-optimized targeting selects a meaningfully different patient set than highest-risk targeting (~55% overlap), but still *underperforms* highest-risk on realized outcomes here — documented honestly with a full sensitivity sweep rather than tuned to fit the "utility wins" narrative. See `09_optimization.ipynb`.
- **Fairness**: a real, moderate recall gap between African American and Caucasian patients (~3 points) — flagged, not resolved.

## Setup

```bash
pip install -r requirements.txt
```

Follow [`data/README.md`](data/README.md) to get the raw CSVs into `data/raw/`, then run the notebooks in order (01 → 10) or the equivalent `src/` scripts to regenerate `data/processed/`. `python sql/load_db.py` loads the featured data into a local SQLite db for the cohort queries in `sql/`.

## Run the app

**Live demo:** _add your Streamlit Community Cloud URL here after deploying_

```bash
streamlit run app/streamlit_app.py
```

Five tabs: Overview (model comparison), Resource Allocation (live what-if simulator — capacity/cost/effectiveness sliders), Risk Drivers (SHAP), Patient Explorer (per-patient local explanation), Fairness (subgroup audit).

The app trains fresh at startup (~10s, cached for the app instance's lifetime) from `app/assets/diabetic_data_features.csv` (~23MB, committed to the repo) rather than unpickling pre-fit models — fitted sklearn/xgboost objects aren't guaranteed compatible across library versions, and a deploy environment resolving a newer scikit-learn than the one that trained the pickle will fail to load it. Training fresh sidesteps that entirely. After regenerating `data/processed/diabetic_data_features.csv`, refresh the deploy CSV with:

```bash
python app/prepare_assets.py
```

### Deploying to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub (chandwi).
2. "New app" → repo `chandwi/CareRisk`, branch `master`, main file path `app/streamlit_app.py`.
3. Deploy. First build takes a few minutes (installs `requirements.txt`); subsequent pushes to `master` auto-redeploy.
