import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numpy.random import default_rng
from statsmodels.tsa.stattools import acf

rng = default_rng()

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="Market Microstructure Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================================================
# GLOBAL UI STYLING (LARGER, PROFESSIONAL FONTS)
# =================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 16px;
}

h1 { font-size: 2.2rem; }
h2 { font-size: 1.8rem; }
h3 { font-size: 1.5rem; }

section[data-testid="stSidebar"] * {
    font-size: 22px !important;
}

div[data-testid="stMetricValue"] {
    font-size: 40px;
}

div[data-testid="stMetricLabel"] {
    font-size: 80px;
}

button[data-baseweb="tab"] {
    font-size: 50px;
}

div[data-testid="stCaptionContainer"] {
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# =================================================
# PLOT THEME
# =================================================
PLOT_THEME = dict(
    template="plotly_dark",
    hovermode="x unified",
    font=dict(size=15),
    # title=dict(font=dict(size=18)),
    margin=dict(l=50, r=40, t=60, b=50)
)

# =================================================
# HEADER
# =================================================
st.title("Market Microstructure Simulator")
st.caption(
    "Comparative analysis of Poisson and Hawkes order arrival processes "
    "and their impact on intraday price dynamics."
)

# =================================================
# SIDEBAR
# =================================================
st.sidebar.header("Simulation Parameters")

process_type = st.sidebar.radio(
    "Order Arrival Process",
    ["Poisson", "Hawkes"],
    help="Poisson: independent arrivals | Hawkes: self-exciting order flow"
)

base_lambda = st.sidebar.slider(
    "Trade Intensity (λ)",
    50, 200, 120,
    help="Average arrival intensity (events per hour)"
)

impact = st.sidebar.slider(
    "Impact Coefficient",
    0.001, 0.05, 0.03,
    format="%.3f"
)

noise = st.sidebar.slider(
    "Microstructure Noise (σ)",
    0.01, 0.1, 0.04,
    format="%.2f"
)

window = st.sidebar.slider(
    "Order Imbalance Window",
    10, 150, 80
)

st.sidebar.markdown("---")

show_comparison = st.sidebar.checkbox(
    "Show Statistical Comparison",
    value=True
)

# =================================================
# TRUE PROCESSES
# =================================================
def poisson_process(rate, T):
    n_events = rng.poisson(rate * T)
    return np.sort(rng.uniform(0, T, n_events))


def hawkes_process(mu, alpha, beta, T):
    if alpha >= beta:
        raise ValueError("Unstable Hawkes process (α ≥ β)")
    events = []
    t = 0.0
    lambda_bar = mu
    excitation = 0.0
    while t < T:
        t += rng.exponential(1 / lambda_bar)
        if t >= T:
            break
        excitation *= np.exp(-beta * (t - (events[-1] if events else 0)))
        lambda_t = mu + excitation
        if rng.uniform() <= lambda_t / lambda_bar:
            events.append(t)
            excitation += alpha
            lambda_bar = max(lambda_bar, lambda_t + alpha)
    return np.array(events)


# =================================================
# INTRADAY SIMULATION
# =================================================
def run_intraday_simulation(
    process_type="Poisson",
    base_lambda=120,
    impact_coeff=0.03,
    sigma_noise=0.04,
    T=390,
    initial_price=100
):
    rate = base_lambda / 60
    if process_type == "Poisson":
        events = poisson_process(rate, T)
    else:
        events = hawkes_process(rate, 0.8, 1.2, T)

    prices, signs, times = [], [], []
    price = initial_price

    for t in events:
        sign = rng.choice([1, -1])
        price += impact_coeff * sign + rng.normal(0, sigma_noise)
        prices.append(price)
        signs.append(sign)
        times.append(t)

    df = pd.DataFrame({
        "time": times,
        "price": prices,
        "trade_sign": signs
    })
    return df, events


def compute_order_imbalance(df, window):
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


def run_comparison_analysis(base_lambda, T=10000):
    mu = base_lambda / 60
    events_p = poisson_process(mu, T)
    events_h = hawkes_process(mu, 0.8, 1.2, T)

    bins_p = np.histogram(events_p, bins=int(T))[0]
    bins_h = np.histogram(events_h, bins=int(T))[0]

    return {
        "cv_poisson": bins_p.std() / bins_p.mean(),
        "cv_hawkes": bins_h.std() / bins_h.mean(),
        "acf_poisson": acf(bins_p, nlags=50),
        "acf_hawkes": acf(bins_h, nlags=50),
        "iat_poisson": np.diff(events_p),
        "iat_hawkes": np.diff(events_h),
        "n_events_poisson": len(events_p),
        "n_events_hawkes": len(events_h)
    }

# =================================================
# MAIN SIMULATION
# =================================================
df, events = run_intraday_simulation(
    process_type,
    base_lambda,
    impact,
    noise
)
df = compute_order_imbalance(df, window)
df = compute_pnl(df)

# =================================================
# METRICS
# =================================================
st.subheader("Simulation Results")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Trades", f"{len(df):,}")
c2.metric("Final Price", f"${df['price'].iloc[-1]:.2f}")
c3.metric("Max Imbalance", int(df["order_imbalance"].abs().max()))
c4.metric("Final PnL", f"${df['equity'].iloc[-1]:.2f}")

st.markdown("---")

# =================================================
# MAIN PLOTS
# =================================================
col1, col2 = st.columns(2)

with col1:
    fig = go.Figure(go.Scatter(y=df["price"], line=dict(width=1.6)))
    fig.update_layout(**PLOT_THEME, title="Trade-by-Trade Price Formation")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure()
    fig.add_bar(y=df["trade_sign"], opacity=0.25)
    fig.add_scatter(y=df["order_imbalance"], yaxis="y2", line=dict(width=2))
    fig.update_layout(
        **PLOT_THEME,
        title="Order Flow and Imbalance",
        yaxis2=dict(overlaying="y", side="right")
    )
    st.plotly_chart(fig, use_container_width=True)

# =================================================
# COMPARISON SECTION
# =================================================
if show_comparison:
    st.markdown("---")
    st.subheader("Poisson vs Hawkes — Statistical Comparison")

    results = run_comparison_analysis(base_lambda)
    cv_increase = (results["cv_hawkes"] / results["cv_poisson"] - 1) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Poisson CV", f"{results['cv_poisson']:.2f}")
    c2.metric("Hawkes CV", f"{results['cv_hawkes']:.2f}", f"+{cv_increase:.1f}%")
    c3.metric("Relative Clustering Increase", f"{cv_increase:.1f}%")

    tab1, tab2, tab3 = st.tabs([
        "Event Counts",
        "Autocorrelation",
        "Inter-Arrival Times"
    ])

    with tab1:
        fig = make_subplots(1, 2, subplot_titles=["Poisson", "Hawkes"])
        fig.add_histogram(x=results["iat_poisson"], row=1, col=1)
        fig.add_histogram(x=results["iat_hawkes"], row=1, col=2)
        fig.update_layout(**PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()
        fig.add_scatter(y=results["acf_poisson"], name="Poisson")
        fig.add_scatter(y=results["acf_hawkes"], name="Hawkes")
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(**PLOT_THEME, title="Arrival Autocorrelation")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = go.Figure()
        fig.add_histogram(x=results["iat_poisson"], opacity=0.6, name="Poisson")
        fig.add_histogram(x=results["iat_hawkes"], opacity=0.6, name="Hawkes")
        fig.update_layout(**PLOT_THEME, yaxis_type="log")
        st.plotly_chart(fig, use_container_width=True)

    st.success(
        "Hawkes arrivals exhibit higher dispersion and persistent autocorrelation, "
        "consistent with clustered order flow observed in real markets."
    )

# =================================================
# FOOTER
# =================================================
st.caption(
    "This simulator is intended for educational and research purposes. "
    "It illustrates microstructure mechanisms rather than predictive trading performance."
)
