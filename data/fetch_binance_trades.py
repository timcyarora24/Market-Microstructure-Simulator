

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

BASE_URL = "https://data.binance.vision/data/spot/daily/trades"

# Raw column layout of Binance's daily trades CSV dumps (no header row).
COLUMNS = ["trade_id", "price", "qty", "quote_qty", "time_ms", "is_buyer_maker", "is_best_match"]


def download_day(symbol: str, date: str) -> pd.DataFrame:
    # Download and parse one day of tick trades for `symbol` on `date` (YYYY-MM-DD) 
    import requests  # imported lazily so the rest of the module works without `requests` installed

    url = f"{BASE_URL}/{symbol}/{symbol}-trades-{date}.zip"
    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Could not download {url} (status {resp.status_code}). "
            f"Check the symbol is correct and the date isn't today/in the future - "
            f"Binance publishes each day's archive the following day."
        )

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        inner_name = zf.namelist()[0]
        with zf.open(inner_name) as f:
            df = pd.read_csv(f, header=None, names=COLUMNS)

    return df


def clean(df: pd.DataFrame, hours: float | None = None) -> pd.DataFrame:
    df = df.sort_values("time_ms").reset_index(drop=True)
    t0 = df["time_ms"].iloc[0]
    df["time_sec"] = (df["time_ms"] - t0) / 1000.0

    if hours is not None:
        df = df[df["time_sec"] <= hours * 3600].reset_index(drop=True)

    return df[["time_ms", "time_sec", "price", "qty", "is_buyer_maker"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair, e.g. BTCUSDT")
    parser.add_argument("--date", required=True, help="Date YYYY-MM-DD (must be a past date)")
    parser.add_argument("--hours", type=float, default=None, help="Truncate to first N hours of the day")
    parser.add_argument("--out", default=None, help="Output CSV path (default: data/<symbol>_<date>_trades.csv)")
    args = parser.parse_args()

    df = download_day(args.symbol, args.date)
    df = clean(df, hours=args.hours)

    out_path = Path(args.out) if args.out else Path(__file__).parent / f"{args.symbol}_{args.date}_trades.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df):,} trades to {out_path}")
    print(f"Time span: {df['time_sec'].iloc[-1] / 3600:.2f} hours")
    print(f"Average trade rate: {len(df) / df['time_sec'].iloc[-1]:.3f} trades/sec "
          f"({len(df) / (df['time_sec'].iloc[-1] / 60):.1f} trades/min)")


if __name__ == "__main__":
    sys.exit(main())
