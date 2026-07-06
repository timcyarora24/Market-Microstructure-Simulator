

from __future__ import annotations

import numpy as np
from numpy.random import default_rng

_rng = default_rng()


def poisson_process(rate: float, T: float, rng=None) -> np.ndarray:
    
    rng = rng or _rng
    n_events = rng.poisson(rate * T)
    return np.sort(rng.uniform(0, T, n_events))


def hawkes_process(mu: float, alpha: float, beta: float, T: float, rng=None):
    
    if alpha >= beta:
        raise ValueError(
            f"Unstable Hawkes process: alpha={alpha} >= beta={beta} "
            f"(branching ratio must be < 1)"
        )

    rng = rng or _rng
    events = []
    lambdas = []

    t = 0.0
    lambda_bar = mu
    excitation = 0.0

    while t < T:
        # Candidate inter-arrival time under the current upper bound.
        delta_t = rng.exponential(1.0 / lambda_bar)
        t += delta_t
        if t >= T:
            break

        excitation *= np.exp(-beta * delta_t)
        lambda_t = mu + excitation

        if rng.uniform(0.0, 1.0) <= lambda_t / lambda_bar:
            events.append(t)
            lambdas.append(lambda_t)
            excitation += alpha
            lambda_bar = max(lambda_t + alpha, lambda_bar)

    return np.array(events), np.array(lambdas)


def mu_from_target_rate(target_rate: float, alpha: float, beta: float) -> float:
    
    branching_ratio = alpha / beta
    if branching_ratio >= 1:
        raise ValueError("alpha/beta must be < 1 for a stable process")
    return target_rate * (1 - branching_ratio)


def time_series(events: np.ndarray, T: float, bin_width: float = 1.0) -> np.ndarray:

    n_bins = int(T / bin_width)
    counts = np.zeros(n_bins)
    idx = (events / bin_width).astype(int)
    idx = idx[idx < n_bins]
    np.add.at(counts, idx, 1)
    return counts
