# Model Card — CareRisk

## Overview

Predicts probability of 30-day hospital readmission for a diabetic patient encounter, then feeds that risk score into a capacity-constrained intervention allocation model. See [`README.md`](../README.md) for the full pipeline and `notebooks/01`-`10` for the detailed, executed analysis this card summarizes.

## Intended use

A decision-analytics **prototype and simulation** demonstrating how predictive modeling and optimization can support hospital resource-allocation decisions. **Not** a validated clinical tool and not fit for deployment on real patients — see Limitations.

## Data

[UCI Diabetes 130-US Hospitals (1999-2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), 101,766 encounters → 100,114 after excluding expired-discharge encounters (leakage — see `notebooks/01_data_audit.ipynb`). Split by `patient_nbr` (group-aware, 80/20) to prevent the same patient's repeat encounters crossing the train/test boundary.

## Model

XGBoost (300 trees, max_depth=5, lr=0.05), selected over Logistic Regression and Random Forest baselines on ROC-AUC/PR-AUC (`notebooks/06_model_evaluation.ipynb`). Preprocessing: standard-scaled numerics, one-hot encoded categoricals with a rare-category collapser (levels under 300 training-fold observations merged into the column mode — see below). Class imbalance (≈11% positive) handled via `scale_pos_weight`, not resampling.

## Performance (held-out test set, n=20,075)

| Metric | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|
| ROC-AUC | ~0.659 | ~0.662 | **~0.668** |
| PR-AUC | ~0.216 | ~0.209 | **~0.227** |

Consistent with published results on this exact dataset (typical reported AUC 0.63-0.68) — this is a genuinely hard prediction problem from 1999-2008 EHR data with no lab-trend or free-text signal. Accuracy is intentionally not reported as a headline metric (see `README.md`).

## Key drivers (SHAP)

Prior inpatient visits (`number_inpatient`) is by far the dominant driver — confirmed independently by EDA, statistical effect-size analysis, and SHAP. Discharge disposition, prior utilization broadly, number of diagnoses, age, and medication complexity follow.

## A real methodological issue we found and fixed

The first SHAP pass was dominated by categorical levels with under ~50 training observations (e.g. a discharge-disposition code with 21 rows) — tree models fit noise to near-empty categories, and SHAP faithfully reported that unreliable reliance. Fixed with `RareCategoryCollapser` (`src/transformers.py`): rare levels are merged into their column's mode, fit on the training fold only. Documented in detail in `notebooks/07_shap_analysis.ipynb`.

## Decision layer — risk vs. benefit

Explored whether targeting patients by *expected intervention benefit* (risk × assumed effectiveness × readmission cost, minus intervention cost) beats simple highest-risk targeting under a capacity constraint. With uniform effectiveness the two are mathematically identical. Under a scenario where effectiveness is assumed to decline with patient complexity, selection diverges (~55% overlap) but **highest-risk targeting still wins on realized outcomes** — because the assumed effectiveness driver overlaps heavily with the risk model's own strongest signal, so utility-based targeting trades away well-identified true positives for a hypothetical gain that doesn't compensate. Documented honestly rather than tuned to produce a "utility wins" headline — see `notebooks/09_optimization.ipynb` for the full sensitivity sweep. This matches the uplift-modeling literature: utility-based targeting needs an effect-heterogeneity signal that's genuinely independent of risk, which this observational dataset (no ground-truth intervention outcomes) can't provide.

## Fairness

Recall is moderately lower for African American patients (~0.53) than Caucasian patients (~0.56) at the default threshold, with a correspondingly higher false-negative rate. Several subgroups (Asian, Hispanic, youngest age bands) are too small in this test set for trustworthy subgroup metrics. Full detail and a threshold sensitivity sweep in `notebooks/10_fairness_analysis.ipynb`.

## Limitations

- **1999-2008 data.** Clinical practice, medications, and coding conventions have changed substantially since. Not representative of a modern hospital population.
- **No ground-truth intervention outcomes.** The cost/effectiveness/optimization layer is an explicit, labeled simulation with adjustable assumptions — never a causal or measured effect.
- **Moderate discrimination (ROC-AUC ≈ 0.67).** A meaningful share of both false positives and false negatives should be expected at any operating threshold.
- **race/gender were never model features** — used only as post-hoc audit dimensions — but the model can still learn correlated proxies (e.g. `payer_code`, `medical_specialty`) that produce disparate outcomes, as the fairness audit shows.
- **Not validated on any real hospital's population, workflow, or clinical protocol.** Do not deploy as-is.
