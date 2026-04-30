"""Data loading.

Three backends:
    1. yfinance (Yahoo Finance) -- the default. No API key, no rate limits,
       and multi-year daily crypto history is freely available. This is what
       you want for serious validation against historical crashes.
    2. CoinGecko public API -- secondary, useful for coins not on Yahoo. Rate-
       limited to ~5-15 req/min on the free tier; we retry on 429.
    3. Synthetic generator with embedded crash regimes -- used in CI and as a
       fully-reproducible fallback when the network is unavailable.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
COINGECKO_PRO_BASE = "https://pro-api.coingecko.com/api/v3"

# A reasonable basket: large caps with long history. The point cloud lives in
# R^N, so N should be moderate -- 10-30 is plenty for H_0/H_1 to be informative.
DEFAULT_BASKET = [
    "bitcoin", "ethereum", "ripple", "litecoin", "bitcoin-cash",
    "cardano", "stellar", "monero", "dash", "ethereum-classic",
    "tezos", "eos", "tron", "neo", "zcash",
]

# Mapping from CoinGecko coin IDs to Yahoo Finance tickers.
COINGECKO_TO_YAHOO = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "ripple": "XRP-USD",
    "litecoin": "LTC-USD",
    "bitcoin-cash": "BCH-USD",
    "cardano": "ADA-USD",
    "stellar": "XLM-USD",
    "monero": "XMR-USD",
    "dash": "DASH-USD",
    "ethereum-classic": "ETC-USD",
    "tezos": "XTZ-USD",
    "eos": "EOS-USD",
    "tron": "TRX-USD",
    "neo": "NEO-USD",
    "zcash": "ZEC-USD",
}


def fetch_coingecko_history(
    coin_id: str,
    days: int | str = 365,
    vs_currency: str = "usd",
    sleep: float = 6.5,
    max_retries: int = 5,
) -> pd.Series:
    """Daily close prices for one coin. Returns a Series indexed by UTC date.

    The free public CoinGecko API (no key) accepts `days` up to 365 and
    auto-buckets the response (hourly for days <= 90, daily otherwise).
    The `interval=daily` parameter and `days=max` were locked behind a paid
    tier in 2024, so we no longer send them.

    Rate limits on the free tier are ~5-15 req/min. We default to one
    request every 6.5s (~9 req/min) and retry with exponential backoff
    on 429, honoring any Retry-After header the server returns.

    If COINGECKO_API_KEY is set we use the Pro endpoint and unlock `days=max`.
    A Demo (free-tier) key in COINGECKO_DEMO_API_KEY is also supported and
    sent on the public endpoint via the x-cg-demo-api-key header.
    """
    pro_key = os.environ.get("COINGECKO_API_KEY")
    demo_key = os.environ.get("COINGECKO_DEMO_API_KEY")
    if pro_key:
        url = f"{COINGECKO_PRO_BASE}/coins/{coin_id}/market_chart"
        headers = {"x-cg-pro-api-key": pro_key}
        # Pro tier removes the rate-limit pain; we can be quicker
        sleep = min(sleep, 0.6)
    else:
        url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
        headers = {"x-cg-demo-api-key": demo_key} if demo_key else {}
    params = {"vs_currency": vs_currency, "days": days}

    backoff = sleep
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", backoff))
            wait = max(wait, backoff)
            print(f"[{coin_id}] 429 rate-limited, sleeping {wait:.1f}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
            backoff = min(backoff * 2, 120.0)
            continue
        resp.raise_for_status()
        break
    else:
        resp.raise_for_status()  # surface the last error

    prices = resp.json()["prices"]  # [[ms_ts, price], ...]
    ts = pd.to_datetime([p[0] for p in prices], unit="ms", utc=True).normalize()
    s = pd.Series([p[1] for p in prices], index=ts, name=coin_id)
    # Multiple intraday samples on free tier: keep last per day (UTC close)
    s = s.groupby(s.index).last()
    time.sleep(sleep)  # be polite to the public endpoint
    return s


def fetch_yahoo_history(
    coin_id: str,
    days: int | str = 1825,
    vs_currency: str = "usd",
) -> pd.Series:
    """Daily close prices for one coin via Yahoo Finance (yfinance)."""
    import yfinance as yf
    ticker = COINGECKO_TO_YAHOO.get(coin_id)
    if ticker is None:
        raise ValueError(
            f"No Yahoo ticker known for coin_id={coin_id!r}. "
            f"Add it to COINGECKO_TO_YAHOO or use the 'coingecko' source."
        )
    if days == "max":
        period = "max"
    else:
        # yfinance accepts period like "5y", "10y" -- convert from days.
        years = max(1, int(days) // 365)
        period = f"{years}y" if years <= 10 else "max"
    df = yf.download(
        ticker, period=period, interval="1d",
        progress=False, auto_adjust=True, threads=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned empty data for {ticker}")
    # yfinance can return MultiIndex columns when given a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].rename(coin_id)
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s.groupby(s.index).last()
    return s


def fetch_basket(
    coins: Iterable[str] = DEFAULT_BASKET,
    days: int | str = 1825,
    cache_dir: Path | None = None,
    source: str = "yahoo",
) -> pd.DataFrame:
    """Fetch a basket of coins, return aligned DataFrame of close prices.

    Parameters
    ----------
    coins : iterable of CoinGecko coin IDs (also used as the column names).
    days : integer number of days, or "max" (Pro key required for CoinGecko).
    cache_dir : if set, per-coin CSVs are read/written here so re-runs are free.
    source : "yahoo" (default, no key, multi-year) or "coingecko" (rate-limited).
    """
    if source not in ("yahoo", "coingecko"):
        raise ValueError(f"unknown source {source!r}; use 'yahoo' or 'coingecko'")
    fetch_one = fetch_yahoo_history if source == "yahoo" else fetch_coingecko_history
    series = []
    for c in coins:
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{source}_{c}.csv"
            if cache_path.exists():
                series.append(pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0])
                continue
        s = fetch_one(c, days=days)
        if cache_dir is not None:
            s.to_frame().to_csv(cache_path)
        series.append(s)
    df = pd.concat(series, axis=1).sort_index()
    df.index = df.index.tz_convert(None) if df.index.tz else df.index
    return df


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns; first row dropped."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


# --------------------------------------------------------------------------- #
# Synthetic data with crash regimes (reproducible fallback / tests)
# --------------------------------------------------------------------------- #

def synthetic_market(
    n_assets: int = 15,
    n_days: int = 2500,
    crash_dates: list[int] | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]]:
    """Generate a synthetic multi-asset return panel with crash regimes.

    Mechanism: a regime variable s_t in {calm, stressed} drives both the
    average correlation rho_t and the volatility sigma_t. In the stressed
    regime correlations spike toward 1 and volatility doubles -- exactly
    the geometry that persistent homology should pick up (the point cloud
    collapses onto a one-dimensional manifold, killing H_1 features and
    inflating max persistence in H_0).

    Returns
    -------
    prices : DataFrame (n_days, n_assets)
    returns : DataFrame (n_days - 1, n_assets)
    crash_timestamps : list of pd.Timestamp marking the crash *peaks*.
    """
    rng = np.random.default_rng(seed)
    if crash_dates is None:
        # Place crashes pseudo-uniformly so we have multiple events to evaluate
        crash_dates = list(np.linspace(400, n_days - 200, 6, dtype=int))

    # Regime indicator with 30-day pre-crash buildup and 20-day crash window
    regime = np.zeros(n_days)
    for c in crash_dates:
        regime[max(0, c - 30): c] = np.linspace(0, 1, min(30, c))
        regime[c: min(n_days, c + 20)] = 1.0

    rho = 0.15 + 0.75 * regime          # avg correlation: 0.15 -> 0.9
    sigma = 0.02 + 0.04 * regime        # daily vol: 2% -> 6%
    drift_shock = -0.01 * regime        # negative drift inside crashes

    # Common factor + idiosyncratic, mixed by sqrt(rho)
    common = rng.standard_normal(n_days)
    idio = rng.standard_normal((n_days, n_assets))
    weights = np.sqrt(rho)[:, None]
    z = weights * common[:, None] + np.sqrt(1 - rho)[:, None] * idio
    rets = drift_shock[:, None] + sigma[:, None] * z

    dates = pd.date_range("2017-01-01", periods=n_days, freq="D")
    cols = [f"asset_{i:02d}" for i in range(n_assets)]
    returns = pd.DataFrame(rets, index=dates, columns=cols)
    prices = (1 + returns).cumprod() * 100
    crash_timestamps = [dates[c] for c in crash_dates]
    return prices, returns, crash_timestamps
