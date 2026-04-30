"""
Expected Returns Trends — Daily snapshots + Plotly trend charts.

Usage:
    python -m scripts.expected_returns.trends            # collect + chart + push
    python -m scripts.expected_returns.trends --backfill  # + historical quarter ends
    python -m scripts.expected_returns.trends --chart-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# ── config ───────────────────────────────────────────────────────────

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "SPY"]

DB_DIR = Path.home() / ".cache" / "expected_returns"
DB_PATH = DB_DIR / "trends.db"

REPO_DIR = Path.home() / "expected-return"
HTML_FILENAME = "index.html"
JSON_FILENAME = "data.json"

COLORS = {
    "AAPL": "#555555", "MSFT": "#00a4ef", "GOOGL": "#34a853",
    "AMZN": "#ff9900", "NVDA": "#76b900", "META": "#1877f2",
    "TSLA": "#e82127", "SPY": "#f0f6fc",
}
REF_COLOR = "#8b949e"

QUARTER_ENDS = ["2025-09-30", "2025-12-31", "2026-03-31"]


def _fmt(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.2f}%"


def get_db() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            ticker     TEXT NOT NULL,
            date       TEXT NOT NULL,
            method     TEXT NOT NULL,
            value      REAL,
            PRIMARY KEY (ticker, date, method)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ref_rates (
            date   TEXT PRIMARY KEY,
            rf_10y REAL,
            erp    REAL
        )
    """)
    return conn


# ── collection ───────────────────────────────────────────────────────

def collect() -> dict[str, dict[str, Any]]:
    """Run all 3 methods on TICKERS for today, store in DB."""
    from .company_data import CompanyData
    from .damodaran_data import get_implied_erp, get_treasury_10y, refresh_all

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = get_db()
    results: dict[str, dict[str, Any]] = {}

    refresh_all()
    erp = get_implied_erp() or 0.048
    rf = get_treasury_10y() or 0.043
    conn.execute("INSERT OR REPLACE INTO ref_rates (date, rf_10y, erp) VALUES (?, ?, ?)",
                 (today, rf, erp))
    conn.commit()

    for ticker in TICKERS:
        row = _collect_one(ticker, today, conn)
        results[ticker] = row

    conn.close()
    return results


def _collect_one(ticker: str, date: str, conn: sqlite3.Connection,
                 capm_fn=None, tr_fn=None, dcf_fn=None) -> dict[str, Any]:
    """Collect a single ticker for a single date."""
    from .company_data import CompanyData
    from .capm import compute as _capm
    from .dcf import compute as _dcf
    from .total_return import compute as _tr

    capm_fn = capm_fn or _capm
    tr_fn = tr_fn or _tr
    dcf_fn = dcf_fn or _dcf

    row: dict[str, Any] = {"ticker": ticker, "date": date, "error": None}
    try:
        cd = CompanyData(ticker)
        if not cd.load():
            row["error"] = "Failed to load company data"
            return row

        capm = capm_fn(cd) if capm_fn else {}
        tr = tr_fn(cd) if tr_fn else {}
        dcf = dcf_fn(cd) if dcf_fn else {}

        capm_er = capm.get("expected_return")
        tr_er = tr.get("expected_return")
        dcf_er = dcf.get("expected_return")

        row["capm"] = capm_er
        row["total_return"] = tr_er
        row["dcf"] = dcf_er if isinstance(dcf_er, (int, float)) else None
        row["error"] = dcf.get("error") if dcf.get("error") else None

        for method, val in [("capm", capm_er), ("total_return", tr_er), ("dcf", row["dcf"])]:
            if val is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO snapshots (ticker, date, method, value) VALUES (?, ?, ?, ?)",
                    (ticker, date, method, float(val)),
                )
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO snapshots (ticker, date, method, value) VALUES (?, ?, ?, NULL)",
                    (ticker, date, method),
                )
        conn.commit()
        print(f"  ok {ticker} ({date}): CAPM={_fmt(capm_er)}  TR={_fmt(tr_er)}  DCF={_fmt(row['dcf'])}")

    except Exception as exc:
        log.warning("FAIL %s (%s): %s", ticker, date, exc)
        row["error"] = str(exc)
        conn.execute(
            "INSERT OR REPLACE INTO snapshots (ticker, date, method, value) VALUES (?, ?, 'error', NULL)",
            (ticker, date),
        )
        conn.commit()

    return row


# ── backfill ─────────────────────────────────────────────────────────

def _tz_aware(date_str: str, hist_index: pd.DatetimeIndex) -> pd.Timestamp:
    """Create a Timestamp that matches the timezone of hist_index."""
    tz = getattr(hist_index, 'tz', None)
    if tz:
        return pd.Timestamp(date_str).tz_localize(tz)
    return pd.Timestamp(date_str)


def _historical_capm(ticker: str, historical_rf: float) -> float | None:
    """Approximate CAPM using current industry data."""
    from .damodaran_data import get_implied_erp, find_industry_beta, refresh_all
    refresh_all()
    erp = get_implied_erp() or 0.048

    try:
        info = yf.Ticker(ticker).info or {}
        industry_raw = info.get("industry", "")
        sector_raw = info.get("sector", "")
        ind_beta_data = find_industry_beta(industry_raw)
        if not ind_beta_data.get("unlevered_beta"):
            ind_beta_data = find_industry_beta(sector_raw)

        unlevered = ind_beta_data.get("unlevered_beta", 1.0)
        ind_de = ind_beta_data.get("de_ratio", 0.25)
        tax = ind_beta_data.get("tax_rate", 0.21)
        levered = unlevered * (1 + (1 - tax) * ind_de)

        return historical_rf + levered * erp
    except Exception as exc:
        log.warning("historical_capm failed for %s: %s", ticker, exc)
        return None


def _historical_tr(ticker: str, date_str: str) -> float | None:
    """Approximate historical Total Return using actual trailing returns.
    
    Uses 1-year price return + trailing dividend yield as a proxy.
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        hist = yf_ticker.history(period="2y")
        if hist.empty:
            return None

        target = _tz_aware(date_str, hist.index)
        target_mask = hist.index <= target
        if not target_mask.any():
            return None
        hist_at = hist.loc[target_mask].iloc[-1]
        close_at = hist_at["Close"]

        # Find close ~1 year before
        one_yr_before = target - pd.DateOffset(years=1)
        before_mask = hist.index <= one_yr_before
        if not before_mask.any():
            # Try 9 months as fallback
            before_mask = hist.index <= (target - pd.DateOffset(months=9))
        if not before_mask.any():
            return None
        close_before = hist.loc[before_mask].iloc[-1]["Close"]

        # Annualized price return
        price_return = (close_at / close_before) - 1.0
        price_return = max(min(price_return, 2.0), -0.80)

        # Trailing dividend yield: sum last ~1yr of dividends
        divs = hist.loc[target_mask, "Dividends"].tail(252)
        ttm_dividends = float(divs.sum())
        div_yield = ttm_dividends / close_at if close_at > 0 else 0.0

        tr = price_return + div_yield
        tr = max(min(tr, 2.5), -0.85)
        return tr
    except Exception as exc:
        log.warning("historical_tr failed for %s (%s): %s", ticker, date_str, exc)
        return None


def _historical_dcf(ticker: str, date_str: str, conn: sqlite3.Connection) -> float | None:
    """Approximate historical DCF IRR using historical price + today's projections.
    
    Uses same projected FCFFs (from today's DCF), but solves IRR against
    the historical enterprise value (historical close * shares + debt - cash).
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        hist = yf_ticker.history(period="2y")
        if hist.empty:
            return None

        target = _tz_aware(date_str, hist.index)
        target_mask = hist.index <= target
        if not target_mask.any():
            return None
        hist_row = hist.loc[target_mask].iloc[-1]
        historical_close = float(hist_row["Close"])

        info = yf_ticker.info or {}
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        debt = float(info.get("totalDebt") or 0)
        cash = float(info.get("totalCash") or 0)
        if not shares:
            return None

        hist_mcap = historical_close * shares
        hist_ev = hist_mcap + debt - cash
        if hist_ev <= 0:
            return None

        # Build FCFF projections using DCF module
        from .dcf import compute as dcf_compute
        from .company_data import CompanyData

        cd = CompanyData(ticker)
        if not cd.load():
            return None

        dcf_result = dcf_compute(cd)
        projections = dcf_result.get("projections", [])
        terminal_growth = dcf_result.get("terminal_growth", 0.025)

        if not projections:
            return None

        # Newton-Raphson solver against historical EV
        def npv(rate: float) -> float:
            pv = 0.0
            for p in projections:
                pv += p["fcff"] / (1 + rate) ** p["year"]
            tf = projections[-1]["fcff"] * (1 + terminal_growth)
            tv = tf / (rate - terminal_growth) if rate > terminal_growth else 0
            pv += tv / (1 + rate) ** len(projections)
            return pv - hist_ev

        fcffs = [p["fcff"] for p in projections]
        if sum(1 for f in fcffs if f > 0) < 2:
            return None

        implied_irr = None
        for guess in np.linspace(0.01, 0.30, 30):
            rate = guess
            for _ in range(50):
                f = npv(rate)
                if abs(f) < 1e-6:
                    implied_irr = rate
                    break
                der = (npv(rate + 1e-6) - npv(rate - 1e-6)) / (2e-6)
                if abs(der) < 1e-12:
                    break
                rate = rate - f / der
            if implied_irr is not None:
                break

        if implied_irr is not None:
            return max(min(implied_irr, 0.50), -0.10)
        return None
    except Exception as exc:
        log.warning("historical_dcf failed for %s (%s): %s", ticker, date_str, exc)
        return None


def backfill():
    """Backfill historical quarter-end data points."""
    from .damodaran_data import refresh_all
    refresh_all()
    conn = get_db()

    # Get historical 10Y rates from ^TNX
    print("\nGetting historical 10Y rates...")
    tnx = yf.Ticker("^TNX")
    tnx_hist = tnx.history(period="1y")
    hist_rf: dict[str, float] = {}
    for qe in QUARTER_ENDS:
        target = _tz_aware(qe, tnx_hist.index)
        mask = tnx_hist.index <= target
        if mask.any():
            close = tnx_hist.loc[mask].iloc[-1]["Close"]
            hist_rf[qe] = close / 100.0
        else:
            hist_rf[qe] = 0.043
        print(f"  {qe}: Rf = {hist_rf[qe]*100:.2f}%")

    from .damodaran_data import get_implied_erp
    erp = get_implied_erp() or 0.048
    for qe in QUARTER_ENDS:
        conn.execute("INSERT OR REPLACE INTO ref_rates (date, rf_10y, erp) VALUES (?, ?, ?)",
                     (qe, hist_rf.get(qe, 0.043), erp))
    conn.commit()

    for ticker in TICKERS:
        print(f"\n  Backfilling {ticker}...")
        for qe in QUARTER_ENDS:
            capm = _historical_capm(ticker, hist_rf.get(qe, 0.043))
            tr = _historical_tr(ticker, qe)
            dcf = _historical_dcf(ticker, qe, conn)

            for method, val in [("capm", capm), ("total_return", tr), ("dcf", dcf)]:
                if val is not None:
                    conn.execute(
                        "INSERT OR REPLACE INTO snapshots (ticker, date, method, value) VALUES (?, ?, ?, ?)",
                        (ticker, qe, method, float(val)),
                    )
                else:
                    conn.execute(
                        "INSERT OR REPLACE INTO snapshots (ticker, date, method, value) VALUES (?, ?, ?, NULL)",
                        (ticker, qe, method),
                    )
            conn.commit()
            print(f"    {qe}: CAPM={_fmt(capm)}  TR={_fmt(tr)}  DCF={_fmt(dcf)}")

    conn.close()
    print("\nBackfill complete")


# ── data loading ──────────────────────────────────────────────────────

def _load_ticker_data(conn: sqlite3.Connection) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT ticker, date, method, value FROM snapshots ORDER BY ticker, date, method"
    ).fetchall()

    data: dict[str, dict[str, list]] = {}
    for ticker, date, method, value in rows:
        if ticker not in data:
            data[ticker] = {"dates": [], "capm": [], "total_return": [], "dcf": []}
        if date not in data[ticker]["dates"]:
            data[ticker]["dates"].append(date)
            data[ticker]["capm"].append(None)
            data[ticker]["total_return"].append(None)
            data[ticker]["dcf"].append(None)
        idx = data[ticker]["dates"].index(date)
        if method == "capm":
            data[ticker]["capm"][idx] = value
        elif method == "total_return":
            data[ticker]["total_return"][idx] = value
        elif method == "dcf":
            data[ticker]["dcf"][idx] = value

    for tk in data:
        z = sorted(zip(data[tk]["dates"], data[tk]["capm"], data[tk]["total_return"], data[tk]["dcf"]))
        data[tk]["dates"] = [x[0] for x in z]
        data[tk]["capm"] = [x[1] for x in z]
        data[tk]["total_return"] = [x[2] for x in z]
        data[tk]["dcf"] = [x[3] for x in z]

    return data


# ── chart generation ─────────────────────────────────────────────────

def _chart_html(data: dict[str, Any]) -> str:
    tickers = [t for t in TICKERS if t in data and any(v is not None for v in data[t].get("capm", []))]
    if not tickers:
        return "<html><body><h2>No data collected yet.</h2></body></html>"

    # Reference rates
    rf_pct, erp_pct, mkt_ret_pct = 4.3, 4.8, 9.1
    try:
        conn = get_db()
        row = conn.execute("SELECT rf_10y, erp FROM ref_rates ORDER BY date DESC LIMIT 1").fetchone()
        if row:
            rf_pct = row[0] * 100
            erp_pct = row[1] * 100
            mkt_ret_pct = rf_pct + erp_pct
        conn.close()
    except Exception:
        pass

    all_dates = sorted(set(d for td in data.values() for d in td["dates"]))

    def _t(tk, dates, vals, is_spy, suffix="%"):
        c = COLORS.get(tk, "#888")
        return {
            "name": tk,
            "x": dates,
            "y": [v * 100 if v is not None else None for v in vals],
            "mode": "lines+markers",
            "line": {"color": c, "width": 1.5 if not is_spy else 1, "dash": "dot" if is_spy else "solid"},
            "marker": {"size": 4, "color": c},
            "hovertemplate": f"{tk}: %{{y:.2f}}{suffix}<extra></extra>",
        }

    def _s(tk, dates, a, b):
        spreads = []
        for i in range(len(dates)):
            va = a[i] if i < len(a) else None
            vb = b[i] if i < len(b) else None
            if va is not None and vb is not None:
                spreads.append((va - vb) * 100)
            else:
                spreads.append(None)
        c = COLORS.get(tk, "#888")
        is_spy = tk == "SPY"
        return {
            "name": tk,
            "x": dates,
            "y": spreads,
            "mode": "lines+markers",
            "line": {"color": c, "width": 1.5 if not is_spy else 1, "dash": "dot" if is_spy else "solid"},
            "marker": {"size": 4, "color": c},
            "hovertemplate": f"{tk}: %{{y:.2f}}%<extra></extra>",
        }

    def _rl(name, dates, val_pct, color, dash):
        return {
            "name": name,
            "x": [dates[0], dates[-1]],
            "y": [val_pct, val_pct],
            "mode": "lines",
            "line": {"color": color, "width": 1, "dash": dash},
            "hovertemplate": f"{name}: {val_pct:.2f}%<extra></extra>",
            "showlegend": True,
        }

    capm_traces = []
    tr_traces = []
    dcf_traces = []
    s1_traces = []  # DCF-CAPM
    s2_traces = []  # TR-DCF
    s3_traces = []  # TR-CAPM

    for tk in tickers:
        td = data[tk]
        spy = tk == "SPY"
        capm_traces.append(_t(tk, td["dates"], td["capm"], spy))
        tr_traces.append(_t(tk, td["dates"], td["total_return"], spy))
        dcf_traces.append(_t(tk, td["dates"], td["dcf"], spy))
        s1_traces.append(_s(tk, td["dates"], td["dcf"], td["capm"]))
        s2_traces.append(_s(tk, td["dates"], td["total_return"], td["dcf"]))
        s3_traces.append(_s(tk, td["dates"], td["total_return"], td["capm"]))

    # Reference lines
    dr = all_dates if len(all_dates) >= 2 else ["2025-09-30", "2026-04-30"]
    capm_traces.append(_rl("Rf (10Y)", dr, rf_pct, REF_COLOR, "dash"))
    capm_traces.append(_rl("ERP", dr, erp_pct, REF_COLOR, "dot"))
    capm_traces.append(_rl("Rf+ERP (Market)", dr, mkt_ret_pct, "#d29922", "dot"))

    # Zero lines on spread charts
    for st in [s1_traces, s2_traces, s3_traces]:
        st.append({
            "name": "_zero", "x": dr, "y": [0] * len(dr), "mode": "lines",
            "line": {"color": "#f85149", "width": 1}, "showlegend": False, "hoverinfo": "skip",
        })

    def _tj(obj): return json.dumps(obj, allow_nan=False)

    def _lyt(title, ylabel):
        return _tj({
            "template": "plotly_dark",
            "paper_bgcolor": "#161b22",
            "plot_bgcolor": "#0d1117",
            "font": {"color": "#c9d1d9"},
            "hovermode": "x unified",
            "title": {"text": title, "font": {"size": 12}, "x": 0.0, "yanchor": "top"},
            "xaxis": {"gridcolor": "#21262d", "zerolinecolor": "#21262d", "tickfont": {"size": 9}},
            "yaxis": {"title": ylabel, "gridcolor": "#21262d", "zerolinecolor": "#30363d", "tickformat": ".1f"},
            "margin": {"t": 28, "b": 40, "r": 8, "l": 48},
            "legend": {"font": {"size": 7.5}, "orientation": "h", "y": -0.35, "x": 0.5, "xanchor": "center"},
            "height": 320, "showlegend": True,
        })

    # Table
    def _pct(v):
        if v is None:
            return '<span class="a">N/A</span>'
        p = v * 100
        cls = "g" if p > 12 else ("o" if p > 7 else "r")
        return f'<span class="{cls}">{p:.2f}%</span>'

    def _spr(v):
        if v is None:
            return '<span class="a">N/A</span>'
        p = v * 100
        cls = "g" if p > 0 else "r"
        return f'<span class="{cls}">{p:+.2f}%</span>'

    def _rv(v):
        return f'<span style="color:{REF_COLOR}">{v:.2f}%</span>'

    trs = ""
    for tk in tickers:
        td = data[tk]
        c = td["capm"][-1] if td["capm"] else None
        t = td["total_return"][-1] if td["total_return"] else None
        d = td["dcf"][-1] if td["dcf"] else None
        s1 = (d - c) if d is not None and c is not None else None
        s2 = (t - d) if t is not None and d is not None else None
        s3 = (t - c) if t is not None and c is not None else None
        trs += (
            f"<tr><td class='tc'>{tk}</td>"
            f"<td>{_pct(c)}</td><td>{_pct(t)}</td><td>{_pct(d)}</td>"
            f"<td>{_spr(s1)}</td><td>{_spr(s2)}</td><td>{_spr(s3)}</td></tr>"
        )

    rr = (
        f'<tr class="rr"><td class="tc">Rf</td><td>{_rv(rf_pct)}</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr>'
        f'<tr class="rr"><td class="tc">ERP</td><td>{_rv(erp_pct)}</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr>'
        f'<tr class="rr"><td class="tc">Market (Rf+ERP)</td><td>{_rv(mkt_ret_pct)}</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr>'
    )

    lu = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mag 7 Expected Returns Trends</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:20px}}
  .c{{max-width:1200px;margin:0 auto}}
  h1{{color:#f0f6fc;font-size:1.4em;margin-bottom:2px}}
  .sub{{color:#8b949e;font-size:.78em;margin-bottom:6px}}
  .grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:16px}}
  .bx{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:3px}}
  .n{{background:rgba(210,153,34,.06);border:1px solid #30363d;border-radius:6px;padding:6px 10px;margin-bottom:12px;font-size:.78em;color:#8b949e}}
  .n strong{{color:#c9d1d9}}
  .tw{{overflow-x:auto}}
  table{{width:100%;border-collapse:collapse;font-size:.8em}}
  th{{background:#161b22;color:#8b949e;text-transform:uppercase;letter-spacing:.04em;font-weight:600;padding:6px 8px;text-align:center;border-bottom:2px solid #30363d;white-space:nowrap}}
  td{{padding:5px 8px;text-align:center;border-bottom:1px solid #161b22;font-variant-numeric:tabular-nums}}
  tr:hover td{{background:rgba(48,54,61,.3)}}
  .tc{{color:#f0f6fc;font-weight:700;text-align:left}}
  .g{{color:#3fb950}}
  .r{{color:#f85149}}
  .o{{color:#d29922}}
  .a{{color:#8b949e}}
  h2{{color:#f0f6fc;font-size:1em;margin:14px 0 6px}}
  .fn{{color:#8b949e;font-size:.72em;text-align:center;margin-top:18px;padding:10px;border-top:1px solid #21262d}}
  .rr td{{color:#8b949e;font-size:.85em}}
  @media(max-width:800px){{.grid{{grid-template-columns:1fr 1fr}}}}
  @media(max-width:500px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="c">
<h1>Mag 7 + SPY  Expected Returns</h1>
<p class="sub">Last updated: {lu} / CAPM+CRP / Total Return Decomposition / FCFF DCF IRR</p>
<div class="n">
<strong>Top row methods:</strong> CAPM (Rf/ERP/Market ref lines) / Total Return (div+buyback+growth) / DCF IRR (implied by price)<br>
<strong>Bottom row spreads:</strong> DCF−CAPM / TR−DCF / TR−CAPM — green = attractive, red = overvalued
</div>

<div class="grid" id="g">
<div class="bx"><div id="c1"></div></div>
<div class="bx"><div id="c2"></div></div>
<div class="bx"><div id="c3"></div></div>
<div class="bx"><div id="c4"></div></div>
<div class="bx"><div id="c5"></div></div>
<div class="bx"><div id="c6"></div></div>
</div>

<h2>Latest Snapshot</h2>
<div class="tw">
<table>
<thead><tr>
<th style="text-align:left">Ticker</th><th>CAPM</th><th>Total Return</th><th>DCF IRR</th><th>DCF−CAPM</th><th>TR−DCF</th><th>TR−CAPM</th>
</tr></thead>
<tbody>
{trs}
{rr}
</tbody>
</table>
</div>
</div>

<script>
Plotly.newPlot('c1',{_tj(capm_traces)},{_lyt("CAPM","Return (%)")},{{responsive:true,displayModeBar:false}});
Plotly.newPlot('c2',{_tj(tr_traces)},{_lyt("Total Return","Return (%)")},{{responsive:true,displayModeBar:false}});
Plotly.newPlot('c3',{_tj(dcf_traces)},{_lyt("DCF IRR","Return (%)")},{{responsive:true,displayModeBar:false}});
Plotly.newPlot('c4',{_tj(s1_traces)},{_lyt("DCF−CAPM","Spread (%)")},{{responsive:true,displayModeBar:false}});
Plotly.newPlot('c5',{_tj(s2_traces)},{_lyt("TR−DCF","Spread (%)")},{{responsive:true,displayModeBar:false}});
Plotly.newPlot('c6',{_tj(s3_traces)},{_lyt("TR−CAPM","Spread (%)")},{{responsive:true,displayModeBar:false}});

var cs=['c1','c2','c3','c4','c5','c6'];
cs.forEach(function(src){{
  document.getElementById(src).on('plotly_relayout',function(ev){{
    var r0=ev['xaxis.range[0]']||null,r1=ev['xaxis.range[1]']||null;
    if(r0||r1||ev['xaxis.autorange']){{
      cs.forEach(function(dst){{if(dst!==src)Plotly.relayout(document.getElementById(dst),{{'xaxis.range':[r0,r1]}});}});
    }}
  }});
}});
</script>

<div class="fn">
Data: Aswath Damodaran (NYU Stern) / Yahoo Finance / Updated daily via OpenClaw<br>
Not investment advice. SPY shown as market benchmark.
</div>
</body>
</html>"""


# ── git ───────────────────────────────────────────────────────────────

def _commit_push():
    repo = REPO_DIR
    if not repo.exists():
        log.warning("Repo dir %s not found", repo)
        return
    try:
        subprocess.run(["git", "-C", str(repo), "add", HTML_FILENAME, JSON_FILENAME], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-m",
                        f"Daily update {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"],
                       check=False, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "push"], check=True, capture_output=True)
        print("  Pushed to GitHub Pages repo")
    except subprocess.CalledProcessError as exc:
        s = exc.stderr.decode() if exc.stderr else ""
        if "nothing to commit" in s:
            print("  Nothing changed, skip commit")
        else:
            log.warning("Git push failed: %s", s[:200])


def write_output(html: str, json_data: dict):
    repo = REPO_DIR
    repo.mkdir(parents=True, exist_ok=True)
    (repo / HTML_FILENAME).write_text(html)
    (repo / JSON_FILENAME).write_text(json.dumps(json_data, indent=2))
    print(f"  {repo / HTML_FILENAME}")
    print(f"  {repo / JSON_FILENAME}")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Expected Returns Trends")
    parser.add_argument("--backfill", action="store_true", help="Backfill past 3 quarter ends")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--chart-only", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    if args.backfill:
        backfill()

    collect_flag = not args.chart_only and not args.backfill
    chart_flag = not args.collect_only

    if collect_flag:
        print(f"\nCollecting {len(TICKERS)} tickers ({datetime.now():%Y-%m-%d})...")
        results = collect()
        ok = sum(1 for v in results.values() if not v.get("error"))
        print(f"  {ok}/{len(TICKERS)} tickers ok")

    if chart_flag:
        conn = get_db()
        data = _load_ticker_data(conn)
        conn.close()

        if not data:
            print("No data in database yet.")
            return

        html = _chart_html(data)
        json_data = {}
        for tk, td in data.items():
            json_data[tk] = {
                "dates": td["dates"],
                "capm": [v * 100 if v is not None else None for v in td["capm"]],
                "total_return": [v * 100 if v is not None else None for v in td["total_return"]],
                "dcf": [v * 100 if v is not None else None for v in td["dcf"]],
            }

        write_output(html, json_data)
        if not args.no_push:
            _commit_push()

    print("\nDone")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
