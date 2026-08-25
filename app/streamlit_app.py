"""
CareRisk — Hospital Readmission Decision Support

Run from the CareRisk project root:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "app" / "assets"
sys.path.insert(0, str(ROOT / "src"))

from evaluation import compare_models, recall_at_capacity  # noqa: E402
from explainability import explain_patient, get_shap_values, global_importance  # noqa: E402
from fairness import subgroup_metrics  # noqa: E402
from optimization import compare_strategies, expected_net_benefit, select_top_k_greedy, threshold_sweep  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

st.set_page_config(page_title="CareRisk", layout="wide")


@st.cache_resource
def load_models():
    return {
        name: joblib.load(ASSETS / f"{name}.joblib")
        for name in ["logistic_regression", "random_forest", "xgboost"]
    }


@st.cache_resource
def load_test_split():
    return joblib.load(ASSETS / "test_split.joblib")


@st.cache_data
def load_demo():
    return pd.read_csv(ASSETS / "demo_subset.csv", index_col=0)


@st.cache_data
def compute_shap(_pipe, _X_test):
    explainer, shap_values, X_transformed_df, X_sample = get_shap_values(_pipe, _X_test, sample_size=1500)
    imp = global_importance(shap_values, X_transformed_df.columns, top_n=15)
    return imp, shap_values, X_transformed_df, X_sample, explainer


models = load_models()
X_test, y_test = load_test_split()
demo = load_demo()

xgb_pipe = models["xgboost"]
risk = xgb_pipe.predict_proba(X_test)[:, 1]
y_true = y_test.values

st.title("CareRisk")
st.caption("Hospital readmission decision-support — UCI Diabetes 130-US Hospitals (1999-2008). A decision-analytics **prototype and simulation**, not a production clinical model.")

tab_overview, tab_allocation, tab_drivers, tab_patient, tab_fairness = st.tabs(
    ["Overview", "Resource Allocation", "Risk Drivers", "Patient Explorer", "Fairness"]
)

# ---------------------------------------------------------------- Overview
with tab_overview:
    st.subheader("Model performance (test set, n={:,})".format(len(y_true)))
    comparison = compare_models(models, X_test, y_test)
    st.dataframe(comparison, use_container_width=True)
    st.caption(
        "Accuracy is intentionally not the headline metric — at an 11% positive rate, always "
        "predicting 'no readmission' scores ~89% accuracy while being useless. XGBoost is used "
        "as the production model below."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients (test set)", f"{len(y_true):,}")
    c2.metric("Actual 30-day readmissions", f"{int(y_true.sum()):,}")
    c3.metric("XGBoost ROC-AUC", f"{roc_auc_score(y_true, risk):.3f}")
    c4.metric("Base readmission rate", f"{y_true.mean():.1%}")

    st.subheader("Predicted risk distribution")
    hist_df = pd.DataFrame({"predicted_risk": risk})
    st.bar_chart(np.histogram(hist_df["predicted_risk"], bins=30)[0])

# ---------------------------------------------------------- Resource alloc
with tab_allocation:
    st.subheader("What-if: intervention capacity, cost, and effectiveness")
    st.caption(
        "This dataset has no ground-truth intervention outcomes — cost/effectiveness below are "
        "explicit, adjustable assumptions, never measured fact. See notebooks 08-09 for the full "
        "sensitivity analysis, including a scenario where effectiveness varies by patient."
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        capacity = st.slider("Intervention capacity (patients)", 50, min(8000, len(risk)), 500, step=50)
        intervention_cost = st.slider("Intervention cost ($)", 20, 500, 100, step=10)
        effectiveness = st.slider("Intervention effectiveness (%)", 5, 50, 20, step=5) / 100
        readmission_cost = st.slider("Cost of a readmission ($)", 2_000, 20_000, 10_000, step=500)

    net_benefit = expected_net_benefit(risk, effectiveness, readmission_cost, intervention_cost)
    selected = select_top_k_greedy(net_benefit, capacity)
    mask = selected == 1
    actual_readmits_selected = int((y_true[mask] == 1).sum())
    expected_prevented = actual_readmits_selected * effectiveness
    total_cost = capacity * intervention_cost
    expected_savings = expected_prevented * readmission_cost
    net = expected_savings - total_cost
    roi = 100 * net / total_cost if total_cost > 0 else 0

    with col_b:
        m1, m2, m3 = st.columns(3)
        m1.metric("Patients selected", f"{capacity:,}")
        m2.metric("Expected readmissions prevented", f"{expected_prevented:,.1f}")
        m3.metric("Recall at this capacity", f"{recall_at_capacity(y_true, risk, capacity/len(risk)):.1%}")
        m4, m5, m6 = st.columns(3)
        m4.metric("Intervention cost", f"${total_cost:,.0f}")
        m5.metric("Expected savings", f"${expected_savings:,.0f}")
        m6.metric("Expected net benefit", f"${net:,.0f}", delta=f"{roi:.0f}% ROI")

    st.subheader("Strategy comparison at this capacity (uniform effectiveness)")
    strategy_comparison = compare_strategies(
        y_true, risk, capacity=capacity,
        intervention_effectiveness=effectiveness,
        readmission_cost=readmission_cost,
        intervention_cost=intervention_cost,
    )
    st.dataframe(strategy_comparison, use_container_width=True)
    st.caption(
        "With uniform effectiveness, 'highest risk' and 'utility-optimized' selection are the same "
        "patients by construction — random selection is the real baseline being beaten here."
    )

    st.subheader("Capacity sweep — net benefit vs. ROI trade-off")
    sweep = threshold_sweep(
        y_true, risk, capacities=[100, 250, 500, 1000, 2000, 4000, min(6000, len(risk))],
        intervention_effectiveness=effectiveness, readmission_cost=readmission_cost, intervention_cost=intervention_cost,
    )
    sc1, sc2 = st.columns(2)
    sc1.line_chart(sweep.set_index("capacity")["expected_net_benefit"])
    sc1.caption("Expected net benefit rises with capacity...")
    sc2.line_chart(sweep.set_index("capacity")["roi_pct"])
    sc2.caption("...but marginal ROI declines as lower-risk patients get added.")

# ------------------------------------------------------------- Risk drivers
with tab_drivers:
    st.subheader("Global drivers of readmission risk (SHAP)")
    imp, shap_values, X_transformed_df, X_sample, explainer = compute_shap(xgb_pipe, X_test)
    st.bar_chart(imp.set_index("feature")["mean_abs_shap"])
    st.dataframe(imp, use_container_width=True)
    st.caption(
        "Rare categorical levels (e.g. a discharge-disposition code with 21 training rows) were "
        "merged into their column's mode before modeling — without that fix, SHAP importance was "
        "dominated by statistically unreliable rare levels instead of real drivers. See "
        "notebooks/07_shap_analysis.ipynb."
    )

# --------------------------------------------------------- Patient explorer
with tab_patient:
    st.subheader("Why was this patient flagged?")
    imp, shap_values, X_transformed_df, X_sample, explainer = compute_shap(xgb_pipe, X_test)
    sample_proba = xgb_pipe.predict_proba(X_sample)[:, 1]

    sort_choice = st.radio("Sort sample by", ["Highest risk first", "Row order"], horizontal=True)
    order = np.argsort(sample_proba)[::-1] if sort_choice == "Highest risk first" else np.arange(len(sample_proba))
    pos_in_order = st.slider("Patient (ranked)", 0, len(order) - 1, 0)
    row_idx = int(order[pos_in_order])

    risk_pct = sample_proba[row_idx]
    tier = "Critical — immediate outreach" if risk_pct >= 0.6 else "High — priority follow-up" if risk_pct >= 0.4 else "Moderate — standard follow-up" if risk_pct >= 0.2 else "Low — no intervention needed"

    c1, c2 = st.columns([1, 2])
    c1.metric("Predicted 30-day readmission risk", f"{risk_pct:.1%}")
    c1.write(f"**Recommended tier:** {tier}")

    local = explain_patient(explainer, shap_values, X_transformed_df, row_idx=row_idx, top_n=8)
    local["direction"] = np.where(local["shap_contribution"] > 0, "pushes risk UP", "pushes risk DOWN")
    with c2:
        st.write("**Top contributing factors:**")
        st.dataframe(local[["feature", "value", "shap_contribution", "direction"]], use_container_width=True)

# ------------------------------------------------------------------ Fairness
with tab_fairness:
    st.subheader("Subgroup performance audit")
    st.caption(
        "race/gender/age are audit dimensions only — never used as model features. "
        "We measure, we don't claim to have solved fairness."
    )
    dim = st.selectbox("Group by", ["race", "gender", "age"])
    st.dataframe(subgroup_metrics(y_true, risk, demo[dim]), use_container_width=True)
    st.caption(
        "Groups with under 30 rows are dropped from this table — too small for a trustworthy estimate. "
        "See notebooks/10_fairness_analysis.ipynb for the full write-up, including allocation rates "
        "under the capacity-constrained policy and a threshold sensitivity sweep."
    )
