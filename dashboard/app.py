import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.simulation import (
    run_intraday_simulation,
    run_comparison_analysis,
    compute_order_imbalance,
    compute_pnl,
)
from config import SimulationConfig
from execution.simulator import MarketSimulator


# PAGE CONFIG

st.set_page_config(
    page_title="Market Microstructure Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
html, body, [class*="css"] { font-size: 16px; }
h1 { font-size: 2.2rem; }
h2 { font-size: 1.8rem; }
h3 { font-size: 1.5rem; }
section[data-testid="stSidebar"] * { font-size: 22px !important; }
div[data-testid="stMetricValue"] { font-size: 40px; }
div[data-testid="stMetricLabel"] { font-size: 80px; }
button[data-baseweb="tab"] { font-size: 50px; }
div[data-testid="stCaptionContainer"] { font-size: 20px; }
</style>
""", unsafe_allow_html=True)

PLOT_THEME = dict(
    template="plotly_dark",
    hovermode="x unified",
    font=dict(size=15),
    margin=dict(l=50, r=40, t=60, b=50)
)


# HEADER

st.title("Market Microstructure Simulator")
st.caption(
    "Comparative analysis of Poisson and Hawkes order arrival processes, "
    "their impact on intraday price dynamics, and execution cost against a "
    "live limit order book."
)


# SIDEBAR

st.sidebar.header("Simulation Parameters")

process_type = st.sidebar.radio(
    "Order Arrival Process",
    ["Poisson", "Hawkes"],
    help="Poisson: independent arrivals | Hawkes: self-exciting order flow"
)

base_lambda = st.sidebar.slider("Trade Intensity (\u03bb)", 50, 200, 120, help="Average arrival intensity (events per hour)")
impact = st.sidebar.slider("Impact Coefficient", 0.001, 0.05, 0.03, format="%.3f")
noise = st.sidebar.slider("Microstructure Noise (\u03c3)", 0.01, 0.1, 0.04, format="%.2f")
window = st.sidebar.slider("Order Imbalance Window", 10, 150, 80)

st.sidebar.markdown("---")
show_comparison = st.sidebar.checkbox("Show Statistical Comparison", value=True)

st.sidebar.markdown("---")
st.sidebar.header("Order Book / Execution")
order_qty = st.sidebar.number_input("Parent Order Size", 1_000, 200_000, 50_000, step=1_000)
participation = st.sidebar.slider("Participation Rate", 0.01, 0.5, 0.1)
n_levels = st.sidebar.slider("Book Depth (levels)", 5, 30, 10)


# MAIN SIMULATION (illustrative trade-by-trade path)

df, events = run_intraday_simulation(process_type, base_lambda, impact, noise)
df = compute_order_imbalance(df, window)
df = compute_pnl(df)

st.subheader("Simulation Results")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Trades", f"{len(df):,}")
c2.metric("Final Price", f"${df['price'].iloc[-1]:.2f}")
c3.metric("Max Imbalance", int(df["order_imbalance"].abs().max()))
c4.metric("Final PnL", f"${df['equity'].iloc[-1]:.2f}")

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    fig = go.Figure(go.Scatter(y=df["price"], line=dict(width=1.6)))
    fig.update_layout(**PLOT_THEME, title="Trade-by-Trade Price Formation")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = go.Figure()
    fig.add_bar(y=df["trade_sign"], opacity=0.25)
    fig.add_scatter(y=df["order_imbalance"], yaxis="y2", line=dict(width=2))
    fig.update_layout(**PLOT_THEME, title="Order Flow and Imbalance", yaxis2=dict(overlaying="y", side="right"))
    st.plotly_chart(fig, use_container_width=True)


# COMPARISON SECTION

if show_comparison:
    st.markdown("---")
    st.subheader("Poisson vs Hawkes \u2014 Statistical Comparison")

    results = run_comparison_analysis(base_lambda)
    cv_increase = (results["cv_hawkes"] / results["cv_poisson"] - 1) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Poisson CV", f"{results['cv_poisson']:.2f}")
    c2.metric("Hawkes CV", f"{results['cv_hawkes']:.2f}", f"+{cv_increase:.1f}%")
    c3.metric("Relative Clustering Increase", f"{cv_increase:.1f}%")

    tab1, tab2, tab3 = st.tabs(["Event Counts", "Autocorrelation", "Inter-Arrival Times"])
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


# ORDER BOOK DEPTH & EXECUTION (uses the real MarketSimulator)

st.markdown("---")
st.subheader("Order Book Depth & Execution Cost")
st.caption(
    "This section runs the actual two-sided LOB execution engine "
    "(`execution.simulator.MarketSimulator`) \u2014 the same code used for the "
    "VWAP Monte Carlo experiment \u2014 rather than the simplified illustrative "
    "price path above."
)

_ALPHA, _BETA = 0.8, 1.2
cfg = SimulationConfig(
    process_type=process_type.lower(),
    mu=base_lambda / 3600,
    alpha=_ALPHA if process_type == "Hawkes" else 0.0,
    beta=_BETA if process_type == "Hawkes" else 1.0,
    n_levels=n_levels,
    order_qty=order_qty,
    participation_rate=participation,
    seed=42,
)
result = MarketSimulator(cfg).run(record_history=True)

e1, e2, e3, e4 = st.columns(4)
e1.metric("Filled Qty", f"{result.filled_qty:,.0f} / {order_qty:,.0f}")
e2.metric("Avg Fill Price", f"${result.avg_price:.4f}" if result.avg_price else "N/A")
e3.metric("Slippage (bps)", f"{result.slippage_bps:.2f}" if result.slippage_bps is not None else "N/A")
e4.metric("Num Child Trades", f"{result.n_trades:,}")

snapshots = result.lob.snapshots
if snapshots:
    times = [s["time"] for s in snapshots]
    mids = [s["mid_price"] for s in snapshots]
    spreads = [s["spread"] for s in snapshots]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=times, y=mids, name="Mid Price", line=dict(width=1.8)), secondary_y=False)
    fig.add_trace(go.Scatter(x=times, y=spreads, name="Spread", line=dict(width=1.2, dash="dot")), secondary_y=True)
    fig.update_layout(**PLOT_THEME, title="Mid Price & Spread Over the Execution Window")
    st.plotly_chart(fig, use_container_width=True)

    final_book = snapshots[-1]
    depth_df = pd.DataFrame(
        [(p, q, "bid") for p, q in final_book["bids"][:15]]
        + [(p, q, "ask") for p, q in final_book["asks"][:15]],
        columns=["price", "qty", "side"],
    )
    fig2 = go.Figure()
    for side, color in [("bid", "seagreen"), ("ask", "indianred")]:
        d = depth_df[depth_df["side"] == side].sort_values("price")
        fig2.add_bar(x=d["price"], y=d["qty"], name=side, marker_color=color)
    fig2.update_layout(**PLOT_THEME, title="Final Order Book Depth (top 15 levels/side)")
    st.plotly_chart(fig2, use_container_width=True)


# FOOTER

st.caption(
    "This simulator is intended for educational and research purposes. "
    "It illustrates microstructure mechanisms rather than predictive trading performance."
)
