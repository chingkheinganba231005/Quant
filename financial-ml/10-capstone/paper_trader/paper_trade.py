#!/usr/bin/env python3
"""Paper-trading loop for the Chapter 10 capstone: regime-aware SPY exposure.

Every trading day, a few minutes before the close:
  1. load prices up to today (live: Yahoo Finance and FRED; replay: a local CSV),
  2. on the first trading day of a year, refit the two-state Gaussian HMM on daily SPY
     log returns up to the end of the previous year and save the parameters,
  3. run the forward filter (past data only) to get today's probability of the turbulent state,
  4. target exposure = 1 - (1 - floor) * p_turbulent, changed only if it moves by more than `band`,
  5. send a market-on-close order for the difference in shares, pay `cost_bps` on the traded value,
  6. append one row to the ledger (date, price, probability, exposure, shares, cash, NAV).

The loop is idempotent: dates already in the ledger are skipped, so it can be run again safely.

    python3 paper_trade.py --replay ../datasets/capstone_daily.csv --start 2026-09-01   # replay history day by day
    python3 paper_trade.py                                                              # live: trade today (needs internet)

Educational code, not investment advice. It never sends real orders.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))


# ---------------------------------------------------------------- model
def fit_hmm(x, iters=200, tol=1e-7):
    """Baum-Welch for a 2-state Gaussian HMM; state 0 = calm (lower volatility)."""
    n = len(x)
    mu = np.quantile(x, [0.3, 0.7]); sd = np.std(x) * np.array([0.6, 1.8])
    A = np.array([[0.95, 0.05], [0.05, 0.95]]); pi = np.array([0.5, 0.5]); ll_old = -np.inf
    for _ in range(iters):
        B = norm.pdf(x[:, None], mu, sd)
        al = np.zeros((n, 2)); c = np.zeros(n)
        al[0] = pi * B[0]; c[0] = al[0].sum(); al[0] /= c[0]
        for t in range(1, n):
            al[t] = (al[t - 1] @ A) * B[t]; c[t] = al[t].sum(); al[t] /= c[t]
        be = np.ones((n, 2))
        for t in range(n - 2, -1, -1):
            be[t] = (A @ (B[t + 1] * be[t + 1])) / c[t + 1]
        g = al * be; g /= g.sum(axis=1, keepdims=True)
        xi = al[:-1, :, None] * A[None] * (B[1:] * be[1:])[:, None, :] / c[1:, None, None]
        A = xi.sum(axis=0); A /= A.sum(axis=1, keepdims=True); pi = g[0]
        w = g / g.sum(axis=0); mu = (w * x[:, None]).sum(axis=0); sd = np.sqrt((w * (x[:, None] - mu) ** 2).sum(axis=0))
        ll = np.log(c).sum()
        if ll - ll_old < tol:
            break
        ll_old = ll
    o = np.argsort(sd)
    return {"mu": mu[o].tolist(), "sd": sd[o].tolist(), "A": A[np.ix_(o, o)].tolist(), "pi": pi[o].tolist(), "loglik": float(ll)}


def filter_last(x, p):
    """forward filter; returns P(turbulent | data up to the last observation)"""
    mu, sd, A, pi = (np.asarray(p[k]) for k in ("mu", "sd", "A", "pi"))
    B = norm.pdf(x[:, None], mu, sd)
    a = pi * B[0]; a /= a.sum()
    for t in range(1, len(x)):
        a = (a @ A) * B[t]; a /= a.sum()
    return float(a[1])


# ---------------------------------------------------------------- data
def load_prices(replay_csv=None):
    if replay_csv:
        d = pd.read_csv(replay_csv, index_col=0, parse_dates=True)
        return d[[CONFIG["ticker"], "TBILL"]].dropna(subset=[CONFIG["ticker"]])
    import io, urllib.request
    import yfinance as yf
    px = yf.download(CONFIG["ticker"], start=CONFIG["first_data"], auto_adjust=True, progress=False)["Close"].squeeze()
    raw = urllib.request.urlopen("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3").read().decode()
    tb = pd.to_numeric(pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True).iloc[:, 0], errors="coerce")
    return pd.DataFrame({CONFIG["ticker"]: px, "TBILL": tb.reindex(px.index).ffill()})


def params_for_year(year, x, params_path):
    """HMM parameters fitted on all data before `year`, cached in a JSON file"""
    cache = json.load(open(params_path)) if os.path.exists(params_path) else {}
    if str(year) not in cache:
        cache[str(year)] = fit_hmm(100 * x.loc[: f"{year - 1}-12-31"].to_numpy())
        json.dump(cache, open(params_path, "w"), indent=1)
    return cache[str(year)]


# ---------------------------------------------------------------- one trading day
def trade_day(date, prices, last, params_path):
    tk, cfg = CONFIG["ticker"], CONFIG
    px = prices.loc[:date]
    close = float(px[tk].iloc[-1])
    cash, shares, exposure = (last["cash"], last["shares"], last["exposure"]) if last is not None else (cfg["initial_capital"], 0, 0.0)
    if last is not None:                                           # interest on cash since the previous close
        cash *= (1 + float(px["TBILL"].iloc[-2]) / 100 / 252) if cash > 0 else 1.0   # rate known the day before
    nav = cash + shares * close
    x = np.log(px[tk]).diff().dropna()
    p_turb = filter_last(100 * x.to_numpy(), params_for_year(date.year, x, params_path))
    target = 1 - (1 - cfg["floor"]) * p_turb
    if last is None or abs(target - exposure) > cfg["band"]:
        exposure = target
    trade = int(np.floor(exposure * nav / close)) - shares
    cost = abs(trade) * close * cfg["cost_bps"] / 1e4
    cash -= trade * close + cost; shares += trade
    return {"date": date.strftime("%Y-%m-%d"), "close": round(close, 4), "p_turbulent": round(p_turb, 6), "exposure": round(exposure, 6),
            "shares": shares, "trade": trade, "cost": round(cost, 2), "cash": round(cash, 2), "nav": round(cash + shares * close, 2)}


def run(replay_csv=None, start=None, ledger_path=None, params_path=None):
    ledger_path = ledger_path or os.path.join(HERE, "ledger.csv")
    params_path = params_path or os.path.join(HERE, "hmm_params.json")
    prices = load_prices(replay_csv)
    ledger = pd.read_csv(ledger_path) if os.path.exists(ledger_path) else pd.DataFrame()
    done = set(ledger["date"]) if len(ledger) else set()
    dates = prices.index if replay_csv else prices.index[-1:]
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    rows = ledger.to_dict("records")
    for dt in dates:
        if dt.strftime("%Y-%m-%d") in done:
            continue
        rows.append(trade_day(dt, prices, rows[-1] if rows else None, params_path))
    out = pd.DataFrame(rows)
    out.to_csv(ledger_path, index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replay", help="CSV of daily prices to replay instead of downloading")
    ap.add_argument("--start", help="first date to trade (replay mode)")
    ap.add_argument("--ledger", help="ledger CSV (default: ledger.csv next to this script)")
    a = ap.parse_args()
    led = run(a.replay, a.start, a.ledger)
    print(led.tail(5).to_string(index=False))
