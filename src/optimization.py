"""
Decision simulation + capacity-constrained intervention allocation.

The dataset has no ground-truth intervention outcomes (see data/README.md),
so intervention_effectiveness and readmission_cost below are explicit,
labeled assumptions swept via sensitivity analysis — never presented as
measured fact. This module answers: given only enough capacity to intervene
on N patients, which N maximize expected benefit, and how much better is
that than simpler targeting rules?
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog


def expected_net_benefit(
    risk: np.ndarray,
    intervention_effectiveness: float,
    readmission_cost: float,
    intervention_cost: float,
) -> np.ndarray:
    expected_benefit = risk * intervention_effectiveness * readmission_cost
    return expected_benefit - intervention_cost


def select_top_k_greedy(net_benefit: np.ndarray, capacity: int) -> np.ndarray:
    """Closed-form solution: for a pure cardinality constraint (choose at
    most `capacity` patients, no other constraints), the optimum is simply
    the `capacity` patients with the highest net benefit. Verified against
    the LP formulation in `select_top_k_lp` below — the two must agree."""
    order = np.argsort(net_benefit)[::-1]
    selected = np.zeros_like(net_benefit, dtype=int)
    selected[order[:capacity]] = 1
    return selected


def select_top_k_lp(net_benefit: np.ndarray, capacity: int) -> np.ndarray:
    """Same problem posed as a linear program: maximize sum(net_benefit * x)
    s.t. sum(x) <= capacity, 0 <= x <= 1. This cardinality-constrained
    knapsack with unit weights has a totally-unimodular constraint matrix,
    so the LP relaxation is guaranteed integral — solving it as an LP (via
    scipy.optimize.linprog) still yields a 0/1 solution, matching the greedy
    sort. Included to demonstrate the optimization formulation explicitly,
    not because it's needed for correctness here."""
    n = len(net_benefit)
    c = -net_benefit  # linprog minimizes, we want to maximize
    A_ub = np.ones((1, n))
    b_ub = np.array([capacity])
    bounds = [(0, 1)] * n
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    selected = (result.x > 0.5).astype(int)
    return selected


def select_random(n: int, capacity: int, random_state: int = 42) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=capacity, replace=False)
    selected = np.zeros(n, dtype=int)
    selected[idx] = 1
    return selected


def select_highest_risk(risk: np.ndarray, capacity: int) -> np.ndarray:
    order = np.argsort(risk)[::-1]
    selected = np.zeros_like(risk, dtype=int)
    selected[order[:capacity]] = 1
    return selected


def evaluate_strategy(
    selected: np.ndarray,
    y_true: np.ndarray,
    risk: np.ndarray,
    intervention_effectiveness,
    readmission_cost: float,
    intervention_cost: float,
) -> dict:
    """Assumes intervention reduces readmission probability multiplicatively
    by `intervention_effectiveness` for selected patients who would actually
    have been readmitted (y_true==1) — this is the simulation layer, not an
    observed effect. `intervention_effectiveness` may be a scalar (uniform
    across patients) or a per-patient array (see heterogeneous_effectiveness)."""
    n_selected = int(selected.sum())
    mask = selected == 1
    actual_readmissions_among_selected = int((y_true[mask] == 1).sum())

    eff = np.broadcast_to(np.asarray(intervention_effectiveness, dtype=float), y_true.shape)
    would_be_readmitted_and_selected = mask & (y_true == 1)
    expected_prevented = eff[would_be_readmitted_and_selected].sum()

    total_cost = n_selected * intervention_cost
    expected_savings = expected_prevented * readmission_cost
    net_benefit = expected_savings - total_cost
    roi = (net_benefit / total_cost) if total_cost > 0 else np.nan

    return {
        "patients_selected": n_selected,
        "actual_readmissions_among_selected": actual_readmissions_among_selected,
        "expected_readmissions_prevented": round(expected_prevented, 1),
        "intervention_cost": total_cost,
        "expected_savings": round(expected_savings, 2),
        "expected_net_benefit": round(net_benefit, 2),
        "roi_pct": round(100 * roi, 1) if not np.isnan(roi) else None,
    }


def heterogeneous_effectiveness(
    prior_inpatient: np.ndarray,
    base_effectiveness: float = 0.20,
    complexity_penalty: float = 0.6,
) -> np.ndarray:
    """A pure simulation assumption, not fit from data: patients with many
    prior inpatient stays are also the hardest to actually help with a
    standard care-management call — diminishing marginal effectiveness for
    the most complex, highest-utilization patients. This is what makes
    "highest risk" and "highest expected benefit" diverge; with constant
    effectiveness (see compare_strategies) they're mathematically the same
    ranking. Effectiveness floors at base_effectiveness * (1 - complexity_penalty)."""
    util = np.asarray(prior_inpatient, dtype=float)
    util_norm = np.clip(util / max(util.max(), 1), 0, 1)
    return base_effectiveness * (1 - complexity_penalty * util_norm)


def compare_strategies(
    y_true: np.ndarray,
    risk: np.ndarray,
    capacity: int,
    intervention_effectiveness: float = 0.20,
    readmission_cost: float = 10_000,
    intervention_cost: float = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    n = len(risk)
    net_benefit = expected_net_benefit(risk, intervention_effectiveness, readmission_cost, intervention_cost)

    strategies = {
        "random": select_random(n, capacity, random_state),
        "highest_risk": select_highest_risk(risk, capacity),
        "utility_optimized": select_top_k_greedy(net_benefit, capacity),
    }

    rows = []
    for name, selected in strategies.items():
        row = evaluate_strategy(selected, y_true, risk, intervention_effectiveness, readmission_cost, intervention_cost)
        row["strategy"] = name
        rows.append(row)
    cols = ["strategy"] + [c for c in rows[0] if c != "strategy"]
    return pd.DataFrame(rows)[cols].set_index("strategy")


def compare_strategies_heterogeneous(
    y_true: np.ndarray,
    risk: np.ndarray,
    prior_inpatient: np.ndarray,
    capacity: int,
    readmission_cost: float = 10_000,
    intervention_cost: float = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Same comparison as compare_strategies, but with effectiveness that
    varies per patient (heterogeneous_effectiveness). This is where
    'highest risk' and 'highest expected benefit' actually diverge: a very
    high-risk, very high-utilization patient may have lower realistic
    effectiveness than a moderately-high-risk patient who is more
    actionable — so utility-optimized targeting should select a different,
    higher-value set of patients than pure risk ranking."""
    n = len(risk)
    eff = heterogeneous_effectiveness(prior_inpatient)
    net_benefit = expected_net_benefit(risk, eff, readmission_cost, intervention_cost)

    strategies = {
        "random": (select_random(n, capacity, random_state), eff),
        "highest_risk": (select_highest_risk(risk, capacity), eff),
        "utility_optimized": (select_top_k_greedy(net_benefit, capacity), eff),
    }

    rows = []
    for name, (selected, eff_for_eval) in strategies.items():
        row = evaluate_strategy(selected, y_true, risk, eff_for_eval, readmission_cost, intervention_cost)
        row["strategy"] = name
        row["overlap_with_highest_risk_pct"] = round(
            100 * (selected & strategies["highest_risk"][0]).sum() / capacity, 1
        )
        rows.append(row)
    cols = ["strategy"] + [c for c in rows[0] if c != "strategy"]
    return pd.DataFrame(rows)[cols].set_index("strategy")


def effectiveness_penalty_sweep(
    y_true: np.ndarray,
    risk: np.ndarray,
    prior_inpatient: np.ndarray,
    capacity: int,
    penalties=(0.0, 0.2, 0.4, 0.6, 0.8),
    readmission_cost: float = 10_000,
    intervention_cost: float = 100,
) -> pd.DataFrame:
    """How much does effectiveness need to vary with patient complexity
    before utility-optimized targeting actually beats highest-risk
    targeting on realized (not just expected) outcomes? At penalty=0
    (uniform effectiveness) the two strategies are identical by
    construction. As the penalty grows, utility-optimized increasingly
    diverts capacity away from the highest-risk-but-hardest-to-help
    patients — whether that pays off depends on how much of that risk
    signal was genuine vs. model noise (moderate AUC here means realized
    outcomes don't perfectly track predicted risk)."""
    rows = []
    highest_risk_sel = select_highest_risk(risk, capacity)
    for p in penalties:
        eff = heterogeneous_effectiveness(prior_inpatient, complexity_penalty=p)
        net_benefit = expected_net_benefit(risk, eff, readmission_cost, intervention_cost)
        utility_sel = select_top_k_greedy(net_benefit, capacity)

        hr_result = evaluate_strategy(highest_risk_sel, y_true, risk, eff, readmission_cost, intervention_cost)
        uo_result = evaluate_strategy(utility_sel, y_true, risk, eff, readmission_cost, intervention_cost)
        rows.append({
            "complexity_penalty": p,
            "highest_risk_net_benefit": hr_result["expected_net_benefit"],
            "utility_optimized_net_benefit": uo_result["expected_net_benefit"],
            "utility_optimized_wins": uo_result["expected_net_benefit"] > hr_result["expected_net_benefit"],
            "overlap_pct": round(100 * (utility_sel & highest_risk_sel).sum() / capacity, 1),
        })
    return pd.DataFrame(rows)


def threshold_sweep(
    y_true: np.ndarray,
    risk: np.ndarray,
    capacities: list[int],
    **kwargs,
) -> pd.DataFrame:
    """How does the utility-optimized strategy's outcome change as capacity
    (how many patients the hospital can actually intervene on) changes?"""
    rows = []
    for cap in capacities:
        net_benefit = expected_net_benefit(
            risk,
            kwargs.get("intervention_effectiveness", 0.20),
            kwargs.get("readmission_cost", 10_000),
            kwargs.get("intervention_cost", 100),
        )
        selected = select_top_k_greedy(net_benefit, cap)
        row = evaluate_strategy(
            selected, y_true, risk,
            kwargs.get("intervention_effectiveness", 0.20),
            kwargs.get("readmission_cost", 10_000),
            kwargs.get("intervention_cost", 100),
        )
        row["capacity"] = cap
        rows.append(row)
    cols = ["capacity"] + [c for c in rows[0] if c != "capacity"]
    return pd.DataFrame(rows)[cols]


if __name__ == "__main__":
    import joblib

    pipe = joblib.load("data/processed/models/xgboost.joblib")
    _, X_test, _, y_test = joblib.load("data/processed/models/splits.joblib")

    risk = pipe.predict_proba(X_test)[:, 1]
    y_true = y_test.values

    # sanity check on a 2,000-patient sample (LP over the full 20k test set
    # is slow with a dense equality-free formulation; the point here is just
    # to confirm the LP and greedy solutions agree, not to solve at scale)
    rng_idx = np.random.default_rng(0).choice(len(risk), size=2000, replace=False)
    net_benefit_sample = expected_net_benefit(risk[rng_idx], 0.20, 10_000, 100)
    greedy_sel = select_top_k_greedy(net_benefit_sample, 200)
    lp_sel = select_top_k_lp(net_benefit_sample, 200)
    assert (greedy_sel == lp_sel).all(), "Greedy and LP solutions disagree!"
    print("Greedy vs LP selection: identical on 2,000-patient/200-capacity check (as expected for a cardinality-constrained knapsack)")
    print()

    comparison = compare_strategies(y_true, risk, capacity=500)
    print("Strategy comparison, capacity=500, UNIFORM effectiveness:")
    print(comparison)
    comparison.to_csv("data/processed/strategy_comparison_uniform.csv")

    print()
    df = pd.read_csv("data/processed/diabetic_data_features.csv")
    prior_inpatient = df.loc[X_test.index, "number_inpatient"].values
    hetero_comparison = compare_strategies_heterogeneous(y_true, risk, prior_inpatient, capacity=500)
    print("Strategy comparison, capacity=500, HETEROGENEOUS effectiveness (risk vs benefit diverge):")
    print(hetero_comparison)
    hetero_comparison.to_csv("data/processed/strategy_comparison_heterogeneous.csv")

    print()
    penalty_sweep = effectiveness_penalty_sweep(y_true, risk, prior_inpatient, capacity=500)
    print("Effectiveness-penalty sensitivity sweep (when does utility-optimized beat highest-risk?):")
    print(penalty_sweep.to_string(index=False))
    penalty_sweep.to_csv("data/processed/effectiveness_penalty_sweep.csv", index=False)

    print()
    sweep = threshold_sweep(y_true, risk, capacities=[100, 250, 500, 1000, 2000, 4000])
    print("Capacity sweep (utility-optimized strategy, uniform effectiveness):")
    print(sweep.to_string(index=False))
    sweep.to_csv("data/processed/capacity_sweep.csv", index=False)
