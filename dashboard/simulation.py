

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.random import default_rng
from statsmodels.tsa.stattools import acf

from core.processes import hawkes_process, mu_from_target_rate, poisson_process, time_series

rng = default_rng()

DEFAULT_ALPHA = 0.8
DEFAULT_BETA = 1.2


def run_intraday_simulation(
    process_type="Poisson",
    base_lambda=120,
    impact_coeff=0.03,
    sigma_noise=0.04,
    T=390,
    initial_price=100,
):
    
    rate = base_lambda / 60  # events per minute -> events per second-ish unit used elsewhere
    if process_type == "Poisson":
        events = poisson_process(rate, T, rng=rng)
    else:
        events, _ = hawkes_process(rate, DEFAULT_ALPHA, DEFAULT_BETA, T, rng=rng)

    prices, signs, times = [], [], []
    price = initial_price
    for t in events:
        sign = rng.choice([1, -1])
        price += impact_coeff * sign + rng.normal(0, sigma_noise)
        prices.append(price)
        signs.append(sign)
        times.append(t)

    df = pd.DataFrame({"time": times, "price": prices, "trade_sign": signs})
    return df, events


def run_comparison_analysis(base_lambda=120, T=10000):
    
    mu = base_lambda / 60
    matched_rate = mu / (1 - DEFAULT_ALPHA / DEFAULT_BETA)

    events_poisson = poisson_process(matched_rate, T, rng=rng)
    events_hawkes, _ = hawkes_process(mu, DEFAULT_ALPHA, DEFAULT_BETA, T, rng=rng)

    counts_poisson = time_series(events_poisson, T, bin_width=1)
    counts_hawkes = time_series(events_hawkes, T, bin_width=1)

    return {
        "cv_poisson": counts_poisson.std() / counts_poisson.mean(),
        "cv_hawkes": counts_hawkes.std() / counts_hawkes.mean(),
        "acf_poisson": acf(counts_poisson, nlags=50),
        "acf_hawkes": acf(counts_hawkes, nlags=50),
        "iat_poisson": np.diff(events_poisson),
        "iat_hawkes": np.diff(events_hawkes),
        "n_events_poisson": len(events_poisson),
        "n_events_hawkes": len(events_hawkes),
    }


def compute_order_imbalance(df, window=50):
    df = df.copy()
    df["order_imbalance"] = df["trade_sign"].rolling(window).sum().fillna(0)
    return df


def compute_pnl(df):
    df = df.copy()
    df["returns"] = df["price"].diff().fillna(0)
    df["position"] = np.sign(df["order_imbalance"])
    df["pnl"] = df["position"] * df["returns"]
    df["equity"] = df["pnl"].cumsum()
    return df
