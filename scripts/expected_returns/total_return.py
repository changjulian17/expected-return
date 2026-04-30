"""
Method 2: Total Return Decomposition

Formula:
    E(R) = Dividend Yield + Buyback Yield + Expected Growth

Where:
- Dividend Yield  = TTM dividends / Market Cap (from yfinance)
- Buyback Yield   = Shares outstanding reduction % or repurchases / Market Cap
- Expected Growth = Fundamental Growth (ROE × Retention Ratio)
"""

from __future__ import annotations

import logging
from typing import Any

from . import company_data as cd

log = logging.getLogger(__name__)


def compute(ticker_data: cd.CompanyData) -> dict[str, Any]:
    """Compute Total Return Decomposition for a company.

    Parameters
    ----------
    ticker_data : CompanyData
        Pre-loaded company data object.

    Returns
    -------
    dict with keys: expected_return, dividend_yield, buyback_yield,
                    expected_growth, growth_source, assumptions
    """
    result: dict[str, Any] = {
        "method": "Total Return Decomposition",
        "expected_return": None,
        "expected_return_label": "N/A",
        "assumptions": {},
    }

    # 1. Dividend Yield
    dy = ticker_data.dividend_yield
    if dy is None:
        dy = 0.0
    result["dividend_yield"] = dy
    result["assumptions"]["Dividend Yield"] = f"{dy * 100:.2f}%"
    if dy < 0.001:
        result["assumptions"]["Dividend Yield"] += " (no dividends)"

    # 2. Buyback Yield
    buyback = ticker_data.buyback_yield
    if buyback is None:
        buyback = 0.0
    result["buyback_yield"] = buyback
    result["assumptions"]["Buyback Yield"] = f"{buyback * 100:.2f}%"
    if buyback < 0.001:
        result["assumptions"]["Buyback Yield"] += " (minimal buybacks)"

    # 3. Expected Growth
    # Prefer revenue growth for reliability; fundamental growth (ROE×Retention)
    # tends to be inflated for mature companies with heavy buybacks.
    fund_growth = ticker_data.fundamental_growth
    rev_growth = ticker_data.revenue_growth_3yr()

    expected_growth = 0.0
    growth_source = "None available (defaulted to 0%)"

    if rev_growth is not None:
        expected_growth = rev_growth
        growth_source = "Historical Revenue CAGR (3yr)"
    elif fund_growth is not None:
        expected_growth = min(fund_growth, 0.15)
        growth_source = "Fundamental (ROE × Retention, capped at 15%)"

    # Blend with fundamental when rev growth is low and fundamental suggests more
    if fund_growth is not None and rev_growth is not None:
        if 0.05 <= fund_growth <= 0.25 and rev_growth < 0.05:
            expected_growth = 0.5 * fund_growth + 0.5 * rev_growth
            growth_source = "Blended (50% Fundamental + 50% Historical)"

    expected_growth = max(min(expected_growth, 0.20), -0.10)  # cap -10% to 20%

    result["expected_growth"] = expected_growth
    result["growth_source"] = growth_source
    result["assumptions"]["Expected Growth"] = f"{expected_growth * 100:.2f}%"
    result["assumptions"]["Growth Source"] = growth_source

    # ROE / Retention detail
    roe = ticker_data.roe
    retention = ticker_data.retention_ratio
    if roe is not None:
        result["assumptions"]["ROE"] = f"{roe * 100:.2f}%"
    if retention is not None:
        result["assumptions"]["Retention Ratio"] = f"{retention * 100:.2f}%"

    # 4. Total Return
    expected_return = dy + buyback + expected_growth
    result["expected_return"] = expected_return
    result["expected_return_label"] = f"{expected_return * 100:.2f}%"

    # Components
    result["cash_yield"] = dy + buyback
    result["assumptions"]["Cash Yield (Div + Buyback)"] = f"{(dy + buyback) * 100:.2f}%"

    # Label each component
    result["formula"] = (
        f"E(R) = {dy*100:.2f}% + {buyback*100:.2f}% + {expected_growth*100:.2f}%"
    )
    result["formula_result"] = f"{expected_return * 100:.2f}%"

    return result
