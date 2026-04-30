"""Download daily prices for the basket and cache to CSV.

Usage:
    python scripts/fetch_data.py                          # Yahoo, 5 years
    python scripts/fetch_data.py --days max               # Yahoo, full history
    python scripts/fetch_data.py --source coingecko       # CoinGecko (rate-limited)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cryptotda.data import DEFAULT_BASKET, fetch_basket


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=("yahoo", "coingecko"), default="yahoo",
                   help="data backend (default: yahoo, no key required)")
    p.add_argument("--days", default="1825",
                   help='Integer days, or "max". Coingecko free tier <=365.')
    p.add_argument("--out", default="data/prices.csv")
    p.add_argument("--cache-dir", default="data/cache")
    args = p.parse_args()

    days = args.days if args.days == "max" else int(args.days)
    df = fetch_basket(
        DEFAULT_BASKET, days=days, cache_dir=Path(args.cache_dir), source=args.source,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"Saved {df.shape} -> {out}")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Coins: {list(df.columns)}")


if __name__ == "__main__":
    main()
