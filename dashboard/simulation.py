import numpy as np
import pandas as pd
from numpy.random import default_rng

rng = default_rng()

# =================================================
# TRUE HAWKES PROCESS (Ogata's Thinning)
# =================================================
def hawkes_process(mu, alpha, beta, T):
    """
    Generate Hawkes process using Ogata's thinning algorithm.
    
    Returns:
    - events: Array of event times
    - times: Same as events (for compatibility)
    - lambdas: Intensity at each event time
    """
    if alpha >= beta:
        raise ValueError(f"Unstable: α={alpha} >= β={beta}")
    
    events = []
    times = []
    lambdas = []
    
    t = 0
    lambda_bar = mu
    excitation_term = 0
    
    while t < T:
        delta_t = rng.exponential(1 / lambda_bar)
        t += delta_t
        
        if t >= T:
            break
        
        uniform_prob = rng.uniform(0, 1)
        excitation_term = excitation_term * np.exp(-beta * delta_t)
        lambda_t = mu + excitation_term
        
        if uniform_prob <= lambda_t / lambda_bar:
            events.append(t)
            times.append(t)
            lambdas.append(lambda_t)
            excitation_term += alpha
            lambda_bar = max(lambda_t + alpha, lambda_bar)
    
    return np.array(events), np.array(times), np.array(lambdas)


def poisson_process(rate, T):
    """
    Generate Poisson process with constant rate.
    
    Returns:
    - events: Array of event times
    """
    n_events = rng.poisson(rate * T)
    events = np.sort(rng.uniform(0, T, n_events))
    return events


# =================================================
# TIME SERIES BINNING
# =================================================
def time_series(events, T, bin_width=1):
    """
    Bin events into time series for ACF calculation.
    """
    bin_size = int(T / bin_width)
    counts = np.zeros(bin_size)
    
    for t in events:
        bin_index = int(t / bin_width)
        if bin_index < bin_size:
            counts[bin_index] += 1
    
    return counts


# =================================================
# INTRADAY SIMULATION (for dashboard visuals)
# =================================================
def run_intraday_simulation(
    process_type="Poisson",
    base_lambda=120,
    impact_coeff=0.03,
    sigma_noise=0.04,
    T=390,
    initial_price=100
):
    """
    Run intraday simulation with price impact.
    
    Returns:
    - df: DataFrame with prices, signs, times
    - events: Array of event times
    """
    # Generate event times
    if process_type == "Poisson":
        rate = base_lambda / 60  # Convert to events per minute
        events = poisson_process(rate, T)
    else:  # Hawkes
        mu = base_lambda / 60
        alpha = 0.8
        beta = 1.2
        events, _, _ = hawkes_process(mu, alpha, beta, T)
    
    # Generate prices with market impact
    prices = []
    trade_signs = []
    trade_times = []
    
    price = initial_price
    
    for t in events:
        sign = rng.choice([1, -1])
        noise = rng.normal(0, sigma_noise)
        price += impact_coeff * sign + noise
        
        prices.append(price)
        trade_signs.append(sign)
        trade_times.append(t)
    
    df = pd.DataFrame({
        "time": trade_times,
        "price": prices,
        "trade_sign": trade_signs
    })
    
    return df, events


# =================================================
# COMPARISON ANALYSIS (for diagnostics)
# =================================================
def run_comparison_analysis(base_lambda=120, T=10000):
    """
    Run both Poisson and Hawkes for comparison.
    
    Returns dict with:
    - events_poisson, events_hawkes
    - counts_poisson, counts_hawkes
    - cv_poisson, cv_hawkes
    - acf_poisson, acf_hawkes
    """
    from statsmodels.tsa.stattools import acf
    
    # Parameters
    mu = base_lambda / 60
    alpha = 0.8
    beta = 1.2
    rate = mu / (1 - alpha / beta)
    
    # Generate processes
    events_poisson = poisson_process(rate, T)
    events_hawkes, _, _ = hawkes_process(mu, alpha, beta, T)
    
    # Bin into time series
    counts_poisson = time_series(events_poisson, T, bin_width=1)
    counts_hawkes = time_series(events_hawkes, T, bin_width=1)
    
    # Calculate CV
    cv_poisson = np.std(counts_poisson) / np.mean(counts_poisson)
    cv_hawkes = np.std(counts_hawkes) / np.mean(counts_hawkes)
    
    # Calculate ACF
    acf_poisson = acf(counts_poisson, nlags=50)
    acf_hawkes = acf(counts_hawkes, nlags=50)
    
    # Inter-arrival times
    iat_poisson = np.diff(events_poisson)
    iat_hawkes = np.diff(events_hawkes)
    
    return {
        'events_poisson': events_poisson,
        'events_hawkes': events_hawkes,
        'counts_poisson': counts_poisson,
        'counts_hawkes': counts_hawkes,
        'cv_poisson': cv_poisson,
        'cv_hawkes': cv_hawkes,
        'acf_poisson': acf_poisson,
        'acf_hawkes': acf_hawkes,
        'iat_poisson': iat_poisson,
        'iat_hawkes': iat_hawkes,
        'n_events_poisson': len(events_poisson),
        'n_events_hawkes': len(events_hawkes)
    }


# =================================================
# ORDER IMBALANCE
# =================================================
def compute_order_imbalance(df, window=50):
    df = df.copy()
    df["order_imbalance"] = (
        df["trade_sign"]
        .rolling(window)
        .sum()
        .fillna(0)
    )
    return df


# =================================================
# RETURNS & PNL
# =================================================
def compute_pnl(df):
    df = df.copy()
    df["returns"] = df["price"].diff().fillna(0)
    df["position"] = np.sign(df["order_imbalance"])
    df["pnl"] = df["position"] * df["returns"]
    df["equity"] = df["pnl"].cumsum()
    return df