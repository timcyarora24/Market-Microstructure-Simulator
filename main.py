# 
# run the following commands to test the simulator:
#     python main.py experiment              # VWAP Poisson vs Hawkes Monte Carlo
#     python main.py simulate --process hawkes --order-qty 50000
#     python main.py calibrate data/BTCUSDT_2024-06-03_trades.csv
#     python main.py dashboard                # prints the streamlit run command
# 

from __future__ import annotations

import argparse
import sys

from config import SimulationConfig
from execution.simulator import MarketSimulator
from execution.vwap import run_vwap_experiment


def cmd_experiment(args):
    run_vwap_experiment(n_runs=args.n_runs, alpha=args.alpha, beta=args.beta)


def cmd_simulate(args):
    cfg = SimulationConfig(
        process_type=args.process,
        mu=args.mu,
        alpha=args.alpha,
        beta=args.beta,
        order_qty=args.order_qty,
        n_levels=args.levels,
        participation_rate=args.participation,
        seed=args.seed,
    )
    result = MarketSimulator(cfg).run()
    print(f"Process:        {cfg.process_type}")
    print(f"Filled:         {result.filled_qty:,.0f} / {cfg.order_qty:,.0f}")
    print(f"Avg fill price: {result.avg_price:.4f}" if result.avg_price else "Avg fill price: N/A")
    print(f"Slippage:       {result.slippage_bps:.3f} bps" if result.slippage_bps is not None else "Slippage: N/A")
    print(f"Child trades:   {result.n_trades}")


def cmd_calibrate(args):
    from data.calibrate_hawkes import main as calibrate_main
    calibrate_main(args.csv_path)


def cmd_dashboard(args):
    print("Run this from your shell (needs network + streamlit installed):\n")
    print("    streamlit run dashboard/app.py\n")


def main():
    parser = argparse.ArgumentParser(description="Market Microstructure Simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    p_exp = sub.add_parser("experiment", help="Run the Poisson vs Hawkes VWAP Monte Carlo")
    p_exp.add_argument("--n-runs", type=int, default=400)
    p_exp.add_argument("--alpha", type=float, default=0.5)
    p_exp.add_argument("--beta", type=float, default=1.0)
    p_exp.set_defaults(func=cmd_experiment)

    p_sim = sub.add_parser("simulate", help="Run a single simulation")
    p_sim.add_argument("--process", choices=["poisson", "hawkes"], default="poisson")
    p_sim.add_argument("--mu", type=float, default=1.0)
    p_sim.add_argument("--alpha", type=float, default=0.5)
    p_sim.add_argument("--beta", type=float, default=1.0)
    p_sim.add_argument("--order-qty", type=float, default=50_000)
    p_sim.add_argument("--levels", type=int, default=10)
    p_sim.add_argument("--participation", type=float, default=0.1)
    p_sim.add_argument("--seed", type=int, default=None)
    p_sim.set_defaults(func=cmd_simulate)

    p_cal = sub.add_parser("calibrate", help="Fit Hawkes params to a real trade CSV")
    p_cal.add_argument("csv_path")
    p_cal.set_defaults(func=cmd_calibrate)

    p_dash = sub.add_parser("dashboard", help="Show the command to launch the Streamlit dashboard")
    p_dash.set_defaults(func=cmd_dashboard)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
