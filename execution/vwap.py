
# VWAP execution-cost experiment: Poisson vs. Hawkes order flow

from __future__ import annotations

import numpy as np
from scipy import stats

from config import SimulationConfig
from execution.simulator import MarketSimulator


def _run_many(config: SimulationConfig, n_runs: int, base_seed: int = 0) -> np.ndarray:
    slippage = np.empty(n_runs)
    for i in range(n_runs):
        run_cfg = SimulationConfig(**{**config.__dict__, "seed": base_seed + i})
        result = MarketSimulator(run_cfg).run()
        slippage[i] = result.slippage_bps if result.slippage_bps is not None else np.nan
    return slippage


def run_vwap_experiment(n_runs: int = 400, alpha: float = 0.5, beta: float = 1.0):
    
    poisson_cfg = SimulationConfig(process_type="poisson", mu=1.0, alpha=alpha, beta=beta)
    hawkes_cfg = poisson_cfg.matched_hawkes()

    poisson_slippage = _run_many(poisson_cfg, n_runs, base_seed=0)
    hawkes_slippage = _run_many(hawkes_cfg, n_runs, base_seed=10_000)

    poisson_slippage = poisson_slippage[~np.isnan(poisson_slippage)]
    hawkes_slippage = hawkes_slippage[~np.isnan(hawkes_slippage)]

    welch_t, welch_p = stats.ttest_ind(poisson_slippage, hawkes_slippage, equal_var=False)
    mwu_u, mwu_p = stats.mannwhitneyu(poisson_slippage, hawkes_slippage, alternative="two-sided")

    summary = {
        "poisson": {
            "mean_bps": float(np.mean(poisson_slippage)),
            "std_bps": float(np.std(poisson_slippage)),
            "n": len(poisson_slippage),
        },
        "hawkes": {
            "mean_bps": float(np.mean(hawkes_slippage)),
            "std_bps": float(np.std(hawkes_slippage)),
            "n": len(hawkes_slippage),
        },
        "welch_t_test": {"statistic": float(welch_t), "p_value": float(welch_p)},
        "mann_whitney_u": {"statistic": float(mwu_u), "p_value": float(mwu_p)},
    }

    print("=== VWAP Execution Cost: Poisson vs. Hawkes (mean-matched rate) ===")
    print(f"Poisson : mean={summary['poisson']['mean_bps']:.3f} bps, "
          f"std={summary['poisson']['std_bps']:.3f} bps, n={summary['poisson']['n']}")
    print(f"Hawkes  : mean={summary['hawkes']['mean_bps']:.3f} bps, "
          f"std={summary['hawkes']['std_bps']:.3f} bps, n={summary['hawkes']['n']}")
    print(f"Welch's t-test:      t={welch_t:.3f}, p={welch_p:.4f}")
    print(f"Mann-Whitney U test: U={mwu_u:.1f}, p={mwu_p:.4f}")

    return summary


if __name__ == "__main__":
    run_vwap_experiment()
