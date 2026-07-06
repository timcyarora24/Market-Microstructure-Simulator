# maximul likelihood calibration of hawkes process parameters
# alpha and beta which were hardcoded in hawkes process simulation (simulation/intraday_simulation_hawkes.ipynb)

from __future__ import annotations

import numpy as np   
# pyrefly: ignore [missing-import]
from scipy.optimize import minimize


def _neg_log_likelihood(theta: np.ndarray, t: np.ndarray, T: float) -> float:
    mu, alpha, beta = theta

    if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
        # Keep the optimizer inside the stable region (branching ratio < 1).
        return 1e10

    n = len(t)
    if n == 0:
        return mu * T

    R = np.zeros(n)
    for i in range(1, n):
        R[i] = np.exp(-beta * (t[i] - t[i - 1])) * (1.0 + R[i - 1])

    lambdas = mu + alpha * R
    if np.any(lambdas <= 0):
        return 1e10

    log_term = np.sum(np.log(lambdas))
    compensator = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - t)))

    ll = log_term - compensator
    return -ll


def fit_hawkes(event_times: np.ndarray, T: float, x0=None) -> dict:
    
    # Fit (mu, alpha, beta) by maximum likelihood to a sequence of event times observed on [0, T].
    
    t = np.asarray(event_times, dtype=float)
    t = np.sort(t)
    n = len(t)
    empirical_rate = n / T if T > 0 else 0.0

    if x0 is None:
        # Start with a modest clustering assumption; optimizer refines it.
        x0 = (empirical_rate * 0.6, empirical_rate * 0.3, empirical_rate * 0.6 + 1e-3)

    bounds = [
        (1e-8, None),      # mu
        (1e-8, None),      # alpha
        (1e-6, 20000),      # beta
    ]

    result = minimize(
        _neg_log_likelihood,
        x0=np.array(x0, dtype=float),
        args=(t, T),
        method="L-BFGS-B",
        bounds=bounds,
    )

    mu, alpha, beta = result.x
    return {
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "branching_ratio": alpha / beta,
        "log_likelihood": -result.fun,
        "converged": result.success,
        "n_events": n,
        "T": T,
        "empirical_rate": empirical_rate,
    }


def goodness_of_fit_residuals(event_times: np.ndarray, mu, alpha, beta) -> np.ndarray:
    
    t = np.asarray(event_times, dtype=float)
    n = len(t)
    R = np.zeros(n)
    for i in range(1, n):
        R[i] = np.exp(-beta * (t[i] - t[i - 1])) * (1.0 + R[i - 1])

    # Compensator Lambda(t_i) = mu*t_i + (alpha/beta) * sum_{j<i} (1 - exp(-beta*(t_i - t_j)))
    # Computed incrementally via R for efficiency.
    compensator = np.zeros(n)
    cum_R_integral = 0.0
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        cum_R_integral += (alpha / beta) * (1 - np.exp(-beta * dt)) * (1 + R[i - 1])
        compensator[i] = mu * t[i] + cum_R_integral

    rescaled = np.diff(compensator)
    return rescaled[rescaled > 0]
