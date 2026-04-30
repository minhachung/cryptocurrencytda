"""Download daily prices for the basket from CoinGecko and cache to parquet.

Usage:
    python scripts/fetch_data.py            # fetch default basket, full history
    python scripts/fetch_data.py --days 1825   # last 5 years only
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cryptotda.data import DEFAULT_BASKET, fetch_basket


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", default="max", help='"max" or integer number of days')
    p.add_argument("--out", default="data/prices.csv")
    p.add_argument("--cache-dir", default="data/cache")
    args = p.parse_args()

    days = args.days if args.days == "max" else int(args.days)
    df = fetch_basket(DEFAULT_BASKET, days=days, cache_dir=Path(args.cache_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"Saved {df.shape} -> {out}")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Coins: {list(df.columns)}")


if __name__ == "__main__":
    main()
