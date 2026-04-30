#!/usr/bin/env python3
"""
Damodaran Expected Returns — CLI Interface

Usage:
    # Single company
    python -m scripts.expected_returns.cli MSFT
    python -m scripts.expected_returns.cli TM --country Japan
    python -m scripts.expected_returns.cli AAPL --verbose

    # Portfolio (batch)
    python -m scripts.expected_returns.cli AAPL MSFT GOOGL --portfolio

    # Sector portfolio
    python -m scripts.expected_returns.cli --sector Technology --top 5

    # Refresh Damodaran data
    python -m scripts.expected_returns.cli --refresh

    # List available countries
    python -m scripts.expected_returns.cli --list-countries
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from . import company_data as cd
from . import damodaran_data as dd
from . import portfolio as pmod
from . import report as rmod

# ── logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("expected_returns.cli")


def _run_single(ticker: str, country: str | None, verbose: bool) -> dict[str, Any]:
    """Run all three methods for one ticker."""
    company = cd.CompanyData(ticker, country=country)
    success = company.load()
    if not success:
        print(f"❌ Failed to load data for {ticker}")
        return {"ticker": ticker, "error": True}

    summary = company.to_dict()

    # Method 1: CAPM
    from . import capm as capm_mod
    crp_country = summary.get("country", "US")
    capm_result = capm_mod.compute(company)

    # Method 2: Total Return
    from . import total_return as tr_mod
    tr_result = tr_mod.compute(company)

    # Method 3: DCF
    from . import dcf as dcf_mod
    dcf_result = dcf_mod.compute(company)

    result = {
        **summary,
        "capm": capm_result,
        "total_return": tr_result,
        "dcf": dcf_result,
    }

    if verbose or True:
        print(f"\n{'='*60}")
        print(f"  {summary.get('company_name', ticker)} ({ticker})")
        print(f"  {summary.get('sector', 'N/A')} · {summary.get('country', 'US')}")
        print(f"{'='*60}")
        print(f"  Market Cap: {'${:,.0f}'.format(summary.get('market_cap', 0)) if summary.get('market_cap') else 'N/A'}")
        print(f"  Enterprise Value: {'${:,.0f}'.format(summary.get('enterprise_value', 0)) if summary.get('enterprise_value') else 'N/A'}")
        print()
        print(f"  Method 1 — CAPM + CRP:         {capm_result.get('expected_return_label', 'N/A')}")
        print(f"    Rf: {capm_result.get('rf', 0)*100:.2f}% · β: {capm_result.get('levered_beta', 0):.2f} · ERP: {capm_result.get('erp', 0)*100:.2f}%")
        print(f"    CRP: {capm_result.get('crp', 0)*100:.2f}% · λ: {capm_result.get('lambda', 1.0):.1f}")
        print()
        print(f"  Method 2 — Total Return:        {tr_result.get('expected_return_label', 'N/A')}")
        print(f"    Div Yield: {tr_result.get('dividend_yield', 0)*100:.2f}% · Buyback Yield: {tr_result.get('buyback_yield', 0)*100:.2f}%")
        print(f"    Growth: {tr_result.get('expected_growth', 0)*100:.2f}% ({tr_result.get('growth_source', 'N/A')})")
        print()
        print(f"  Method 3 — DCF → Implied IRR:   {dcf_result.get('expected_return_label', 'N/A')}")
        wacc_val = dcf_result.get('wacc')
        tg_val = dcf_result.get('terminal_growth', 0.025)
        wacc_str = f"{wacc_val*100:.2f}%" if wacc_val else "N/A"
        print(f"    WACC: {wacc_str} · Terminal g: {tg_val*100:.2f}%")
        if dcf_result.get("projections") and not dcf_result.get("error"):
            ev = dcf_result.get('dcf_enterprise_value_wacc')
            cev = dcf_result.get('current_enterprise_value')
            if ev:
                print(f"    DCF Value at WACC: ${ev:,.0f}")
            if cev:
                print(f"    Current EV: ${cev:,.0f}")

    return result


def _run_sector(sector: str | None, top_n: int | None) -> list[dict]:
    """Find stocks in a sector and run analysis on the top N by market cap."""
    import yfinance as yf

    if not sector:
        print("❌ --sector required")
        return []

    print(f"🔍 Searching for top stocks in '{sector}' sector...")

    # Use yfinance to find sector stocks via ETFs
    # S&P 500 sector ETFs
    sector_etf_map = {
        "technology": "XLK",
        "tech": "XLK",
        "software": "IGV",
        "healthcare": "XLV",
        "health": "XLV",
        "financial": "XLF",
        "financials": "XLF",
        "energy": "XLE",
        "consumer": "XLP",
        "consumer discretionary": "XLY",
        "industrial": "XLI",
        "materials": "XMB",
        "real estate": "XLRE",
        "utilities": "XLU",
        "utilities": "XLU",
        "communication": "XLC",
        "telecom": "XLC",
    }

    etf_ticker = None
    for kw, etf in sector_etf_map.items():
        if kw in sector.lower():
            etf_ticker = etf
            break

    if etf_ticker is None:
        print(f"❌ Unknown sector '{sector}'. Try: Technology, Healthcare, Financial, Energy, Consumer, Industrial")
        return []

    try:
        etf = yf.Ticker(etf_ticker)
        holdings = etf.info.get("holdings", [])
        top_holdings = etf.info.get("topHoldings", {})

        # Try to get top holdings from the ETF
        tickers = []
        try:
            etf_holdings = etf.funds_data.holdings() if hasattr(etf, 'funds_data') else None
            if etf_holdings is not None:
                for h in etf_holdings:
                    if hasattr(h, 'symbol') and h.symbol:
                        tickers.append(h.symbol)
        except Exception:
            pass

        if not tickers:
            # Fallback: use known top holdings from the ETF info
            try:
                holdings_data = etf.major_holders if hasattr(etf, 'major_holders') else None
            except Exception:
                pass

        if not tickers:
            print(f"⚠️  Could not get holdings for {etf_ticker}. Using top 10 by weight.")
            # Manual fallback for common sectors
            sector_stocks = {
                "technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM", "ADBE", "INTC"],
                "healthcare": ["UNH", "JNJ", "PFE", "ABBV", "MRK", "TMO", "ABT", "LLY", "BMY", "AMGN"],
                "financial": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "V", "MA"],
                "energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PXD", "OXY", "MPC", "VLO", "PSX"],
            }
            for kw, stocks in sector_stocks.items():
                if kw in sector.lower():
                    tickers = stocks
                    break

        if not tickers:
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

        if top_n:
            tickers = tickers[:top_n]

        print(f"  Found {len(tickers)} stocks in {sector}: {', '.join(tickers)}")
        return [ticker for ticker in tickers]

    except Exception as exc:
        print(f"❌ Error fetching sector data: {exc}")
        return []


def _list_countries():
    """Print available countries from CRP data."""
    crp = dd.get_ctryprem()
    print(f"\nAvailable countries with risk premium data ({len(crp)}):")
    print("-" * 60)
    for country, data in sorted(crp.items())[:30]:
        print(f"  {country:<25s} CRP: {data.get('crp', 0)*100:.2f}%")
    if len(crp) > 30:
        print(f"  ... and {len(crp) - 30} more")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Damodaran Expected Returns — Three methodology comparison"
    )
    parser.add_argument("tickers", nargs="*", help="Stock ticker symbols")
    parser.add_argument("--country", "-c", default=None,
                        help="Country override (for non-US companies)")
    parser.add_argument("--portfolio", "-p", action="store_true",
                        help="Treat multiple tickers as a portfolio")
    parser.add_argument("--sector", "-s", default=None,
                        help="Run on a sector (e.g., Technology, Healthcare)")
    parser.add_argument("--top", type=int, default=None,
                        help="Top N stocks in sector (default: all found)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--refresh", "-r", action="store_true",
                        help="Force refresh Damodaran data cache")
    parser.add_argument("--list-countries", action="store_true",
                        help="List available countries with CRP data")

    args = parser.parse_args()

    # Handle special commands
    if args.list_countries:
        _list_countries()
        return

    if args.refresh:
        print("🔄 Refreshing Damodaran data cache...")
        results = dd.refresh_all()
        success = sum(1 for v in results.values() if v)
        print(f"  Refreshed {success}/{len(results)} datasets")
        return

    # Determine what to analyze
    tickers = []
    if args.sector:
        print(f"📊 Running sector analysis for '{args.sector}'...")
        found = _run_sector(args.sector, args.top)
        if not found:
            print("❌ No tickers found for this sector. Falling back to default.")
            tickers = ["AAPL", "MSFT", "GOOGL"]
        else:
            tickers = found
    elif args.tickers:
        tickers = args.tickers
    else:
        # Default demo
        print("ℹ️  No tickers specified. Running demo on AAPL, MSFT.")
        tickers = ["AAPL", "MSFT"]

    has_portfolio = args.portfolio or len(tickers) > 1

    # Run analysis
    results = []
    for ticker in tickers:
        # Only pass --country if given; for batch portfolios, each company's country
        # is auto-detected from yfinance. The --country flag is mostly for demo/debug.
        ticker_country = args.country if len(tickers) == 1 else None
        result = _run_single(ticker, ticker_country, args.verbose)
        results.append(result)

    # Filter errors
    successful = [r for r in results if not r.get("error")]
    failed = [r.get("ticker") for r in results if r.get("error")]

    if failed:
        print(f"\n⚠️  Failed tickers: {', '.join(failed)}")

    if not successful:
        print("\n❌ No successful analyses. Exiting.")
        return

    # Portfolio aggregation
    portfolio_result = None
    if has_portfolio and len(successful) > 1:
        portfolio_result = pmod.aggregate(successful)

    # Write output
    try:
        outputs = rmod.write_output(successful, portfolio_result)
        print(f"\n✅ Output files written:")
        print(f"   📄 HTML:  {outputs['html']}")
        print(f"   📋 JSON:  {outputs['json']}")
        print(f"   📊 CSV:   {outputs['csv']}")
    except Exception as exc:
        log.warning("Failed to write output files: %s", exc)


if __name__ == "__main__":
    main()
