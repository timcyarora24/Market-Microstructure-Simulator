
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

sys.path.append(".")  # allow running as a plain script from repo root

from core.hawkes_mle import fit_hawkes, goodness_of_fit_residuals
from core.processes import hawkes_process, poisson_process, time_series


def load_trade_times(csv_path: str) -> tuple[np.ndarray, float]:
    df = pd.read_csv(csv_path)
    t = df["time_sec"].values
    t = np.sort(t + np.random.uniform(0, 1e-3, size=len(t)))
    T = t[-1]
    return t, T


def main(csv_path: str):
    t, T = load_trade_times(csv_path)
    print(f"Loaded {len(t):,} real trades spanning {T / 3600:.2f} hours")

    fit = fit_hawkes(t, T)
    print("\n=== Fitted Hawkes parameters (MLE) ===")
    print(f"mu    = {fit['mu']:.5f}  (baseline arrivals/sec)")
    print(f"alpha = {fit['alpha']:.5f}  (excitation per event)")
    print(f"beta  = {fit['beta']:.5f}  (excitation decay rate)")
    print(f"branching ratio (alpha/beta) = {fit['branching_ratio']:.3f}")
    print(f"log-likelihood = {fit['log_likelihood']:.1f}")
    print(f"naive empirical rate = {fit['empirical_rate']:.4f} trades/sec")

    #  Goodness of fit: time-rescaling theorem 
    residuals = goodness_of_fit_residuals(t, fit["mu"], fit["alpha"], fit["beta"])

    # Simulate from fitted params for a visual sanity check
    sim_events, _ = hawkes_process(fit["mu"], fit["alpha"], fit["beta"], T)
    poisson_events = poisson_process(fit["empirical_rate"], T)

    real_iat = np.diff(t)
    sim_iat = np.diff(sim_events)

    bin_width = max(1.0, T / 2000)
    real_counts = time_series(t, T, bin_width)
    sim_counts = time_series(sim_events, T, bin_width)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(real_iat, bins=80, alpha=0.6, label="Real trades", density=True, color="black")
    axes[0].hist(sim_iat, bins=80, alpha=0.5, label="Fitted Hawkes (simulated)", density=True, color="crimson")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Inter-arrival time (s)")
    axes[0].set_title("Inter-Arrival Times: Real vs Fitted")
    axes[0].legend()

    sorted_res = np.sort(residuals)
    theoretical = -np.log(1 - (np.arange(1, len(sorted_res) + 1) - 0.5) / len(sorted_res))
    axes[1].scatter(theoretical, sorted_res, s=4, alpha=0.4, color="steelblue")
    max_q = max(theoretical.max(), sorted_res.max())
    axes[1].plot([0, max_q], [0, max_q], "r--", label="Perfect fit (Exp(1))")
    axes[1].set_xlabel("Theoretical quantile (Exp(1))")
    axes[1].set_ylabel("Empirical quantile (rescaled residuals)")
    axes[1].set_title("Goodness-of-Fit Q-Q Plot")
    axes[1].legend()

    acf_real = acf(real_counts, nlags=50)
    acf_sim = acf(sim_counts, nlags=50)
    axes[2].plot(acf_real, label="Real", color="black")
    axes[2].plot(acf_sim, label="Fitted Hawkes (simulated)", color="crimson")
    axes[2].axhline(0, linestyle="--", color="gray", linewidth=1)
    axes[2].set_xlabel(f"Lag ({bin_width:.1f}s bins)")
    axes[2].set_title("Event-Count Autocorrelation")
    axes[2].legend()

    plt.tight_layout()
    out_path = "data/calibration_diagnostics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved diagnostics plot to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m data.calibrate_hawkes <path_to_trades_csv>")
        sys.exit(1)
    main(sys.argv[1])
