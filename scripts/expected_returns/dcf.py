"""
Method 3: FCFF DCF → Implied IRR

Builds a 5-year DCF using:
- Revenue growth (historical 3yr or analyst estimates)
- Operating margins converging to Damodaran industry average
- Reinvestment via industry Sales/Capital ratio
- WACC as discount rate
- Terminal value via Gordon Growth Model
- Backs out implied expected return (IRR that sets PV = current EV)

Formula:
    EV = Σ(FCFF_t / (1+r)^t) + TV / (1+r)^5
    where r is the implied expected return
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from . import company_data as cd
from . import damodaran_data as dd

log = logging.getLogger(__name__)

# ── defaults ─────────────────────────────────────────────────────────
DEFAULT_PROJECTION_YEARS = 5
DEFAULT_TERMINAL_GROWTH = 0.025  # 2.5%


def compute(
    ticker_data: cd.CompanyData,
    terminal_growth: float | None = None,
    projection_years: int = DEFAULT_PROJECTION_YEARS,
) -> dict[str, Any]:
    """Compute FCFF DCF → implied IRR.

    Parameters
    ----------
    ticker_data : CompanyData
        Pre-loaded company data object.
    terminal_growth : float
        Perpetual growth rate for terminal value (default 2.5%).
    projection_years : int
        Number of years for explicit projection (default 5).

    Returns
    -------
    dict with keys: expected_return (IRR), wacc, terminal_growth,
                    projections (list of year dicts), assumptions
    """
    result: dict[str, Any] = {
        "method": "FCFF DCF → Implied IRR",
        "expected_return": None,
        "expected_return_label": "N/A",
        "wacc": None,
        "terminal_growth": terminal_growth,
        "projections": [],
        "assumptions": {},
    }

    summary = ticker_data.to_dict()

    # Override terminal growth based on geography if not explicitly set
    if terminal_growth is None:
        country = summary.get("country", "US")
        terminal_growth = dd.terminal_growth_for_country(country)
        result["terminal_growth"] = terminal_growth

    # ── 1. Starting values ──────────────────────────────────────────
    revenue = summary.get("revenue")
    ebit = summary.get("operating_income")
    current_margin = summary.get("operating_margin")

    # Geography-aware tax rate: company actual → country statutory → US default
    country = summary.get("country", "US")
    effective_tax = summary.get("effective_tax_rate")
    if effective_tax is not None and effective_tax > 0:
        tax_rate = effective_tax
    elif country:
        tax_rate = dd.find_country_tax_rate(country)
    else:
        tax_rate = dd.FALLBACK_TAX_RATE

    depreciation = summary.get("depreciation", 0) or 0
    capex = summary.get("capex", 0) or 0
    ev = summary.get("enterprise_value")
    mc = summary.get("market_cap")
    cash = summary.get("cash_and_equivalents", 0) or 0
    debt = summary.get("total_debt", 0) or 0

    if ebit is None and current_margin is None:
        result["assumptions"]["Warning"] = "No operating income data; using revenue-based estimation"
        cm = dd.find_industry_margin(summary.get("industry", "Unknown"))
        if isinstance(cm, dict):
            current_margin = cm.get("pre_tax_margin") or cm.get("after_tax_margin") or 0.0
        else:
            current_margin = cm
    elif ebit is not None and current_margin is None and revenue and revenue > 0:
        current_margin = ebit / revenue
    
    if revenue is None or revenue <= 0:
        result["error"] = "No revenue data available for DCF"
        return result
    
    # Sanity checks for data quality
    # If revenue > 1000x market cap, data is likely in wrong currency
    if mc and mc > 0 and revenue / mc > 100:
        result["error"] = "Revenue/market cap ratio is suspicious (likely currency scaling issue with non-US financial data). Try --country to set correct reporting."
        result["assumptions"]["Revenue"] = f"${revenue:,.0f}"
        result["assumptions"]["Market Cap"] = f"${mc:,.0f}"
        result["assumptions"]["Revenue/MC Ratio"] = f"{revenue/mc:.0f}x"
        return result

    if ev is None or ev <= 0:
        # Derive from market cap
        if mc:
            ev = mc + debt - cash
        else:
            result["error"] = "No enterprise value or market cap available"
            return result
    
    # Sanity check: EV shouldn't be > 100x revenue or clearly wrong
    if revenue > 0 and ev / revenue > 50:
        log.warning("EV/revenue ratio of %.0fx seems high; keeping reported EV", ev / revenue)
        result["assumptions"]["Note"] = "High EV/Revenue ratio (may be financial company)"

    # ── 2. Industry averages ────────────────────────────────────────
    industry = summary.get("industry", "Unknown")
    sector = summary.get("sector", "Unknown")
    industry_margin_raw = dd.find_industry_margin(industry)
    if isinstance(industry_margin_raw, dict):
        industry_margin = industry_margin_raw.get("pre_tax_margin") or industry_margin_raw.get("after_tax_margin") or 0.0
    else:
        industry_margin = industry_margin_raw or 0.0
    sales_capital_ratio = dd.find_sales_capital_ratio(industry)
    industry_wacc = dd.find_industry_wacc(industry)

    result["assumptions"]["Industry"] = industry
    result["assumptions"]["Current Operating Margin"] = f"{current_margin * 100:.2f}%" if current_margin else "N/A"
    result["assumptions"]["Industry Target Margin"] = f"{industry_margin * 100:.2f}%"
    result["assumptions"]["Sales/Capital Ratio"] = f"{sales_capital_ratio:.2f}" if sales_capital_ratio is not None else "N/A"

    # ── 3. Growth rate ──────────────────────────────────────────────
    rev_growth = summary.get("revenue_growth_3yr")
    if rev_growth is None or rev_growth <= -0.5:
        rev_growth = 0.05  # default 5%

    # Cap growth
    rev_growth = max(min(rev_growth, 0.40), -0.20)
    result["assumptions"]["Revenue Growth Rate"] = f"{rev_growth * 100:.2f}%"
    result["assumptions"]["Growth Source"] = "3yr Historical CAGR" if summary.get("revenue_growth_3yr") else "Default (5%)"

    # ── 4. Margin convergence ───────────────────────────────────────
    # If company margin exceeds industry target, don't force convergence down
    # (dominant firms can sustain superior margins)
    conv_base = (current_margin if current_margin else industry_margin) or 0.0
    target_margin = industry_margin or 0.0
    # For superior margins: converge only partway (50%) or keep flat if current >> industry
    if current_margin and current_margin > target_margin and current_margin > 0:
        margin_ratio = target_margin / current_margin if current_margin != 0 else 0
        if margin_ratio < 0.3:
            # Very large gap — keep current margin (dominant firm)
            target_margin = conv_base
        else:
            # Moderate gap — converge halfway
            target_margin = conv_base + (target_margin - conv_base) * 0.5
    converged_margin = conv_base
    margin_steps = np.linspace(converged_margin, target_margin, projection_years + 1)[1:]

    # ── 5. Build projections ────────────────────────────────────────
    projections = []
    rev = float(revenue)

    for yr in range(projection_years):
        # Decaying growth: linearly fade to terminal growth by year 5
        growth = rev_growth - (rev_growth - terminal_growth) * (yr / max(projection_years - 1, 1))
        rev_next = rev * (1 + growth)

        margin = margin_steps[yr]
        ebit_next = rev_next * margin
        ebit_at = ebit_next * (1 - tax_rate)

        # Reinvestment = ΔRevenue / Sales-to-Capital ratio
        delta_rev = rev_next - rev
        scr = sales_capital_ratio if sales_capital_ratio and sales_capital_ratio > 0 else 2.0
        reinvestment = delta_rev / scr if scr > 0 else delta_rev * 0.5
        reinvestment = max(reinvestment, 0)  # no negative reinvestment

        fcff = ebit_at - reinvestment

        projections.append({
            "year": yr + 1,
            "revenue": rev_next,
            "growth": growth,
            "margin": margin,
            "ebit": ebit_next,
            "ebit_at": ebit_at,
            "reinvestment": reinvestment,
            "fcff": fcff,
        })

        rev = rev_next

    result["projections"] = projections

    # ── 6. WACC (discount rate) ─────────────────────────────────────
    # Use industry WACC as base
    wacc = industry_wacc
    result["wacc"] = wacc
    result["assumptions"]["WACC"] = f"{wacc * 100:.2f}%"

    # ── 7. DCF at WACC (reference) ──────────────────────────────────
    pv_fcff = sum(p["fcff"] / (1 + wacc) ** (p["year"]) for p in projections)
    # Terminal value: TV = FCFF_6 / (WACC - terminal_growth)
    terminal_fcff = projections[-1]["fcff"] * (1 + terminal_growth)
    tv = terminal_fcff / (wacc - terminal_growth) if wacc > terminal_growth else 0
    pv_tv = tv / (1 + wacc) ** projection_years
    dcf_ev = pv_fcff + pv_tv

    result["pv_fcff_wacc"] = pv_fcff
    result["pv_tv_wacc"] = pv_tv
    result["dcf_enterprise_value_wacc"] = dcf_ev

    # ── 8. Implied IRR ──────────────────────────────────────────────
    # Solve for r where PV of FCFF + TV = current EV
    current_ev = float(ev)

    def npv(rate: float) -> float:
        pv = 0.0
        for p in projections:
            pv += p["fcff"] / (1 + rate) ** p["year"]
        tv = terminal_fcff / (rate - terminal_growth) if rate > terminal_growth else 0
        pv += tv / (1 + rate) ** projection_years
        return pv - current_ev

    # Try to find root using Newton-like approach
    implied_irr = None
    
    # Check if we have mostly positive FCFFs for IRR to be meaningful
    fcffs = [p["fcff"] for p in projections]
    if sum(1 for f in fcffs if f > 0) >= 2 and sum(fcffs) > 0:
        for guess in np.linspace(0.01, 0.30, 30):  # try rates from 1% to 30%
            try:
                irr = _find_irr(npv, guess)
                if irr is not None and irr > -0.5 and irr < 2.0:
                    implied_irr = irr
                    break
            except Exception:
                continue

    if implied_irr is None:
        # Fallback: WACC-based interpretation
        if dcf_ev < current_ev and dcf_ev > 0:
            # Trading above DCF value → expensive
            implied_irr = wacc * (current_ev / dcf_ev) if dcf_ev > 0 else wacc
            result["assumptions"]["IRR Method"] = "Inferred from WACC (above DCF value)"
        elif dcf_ev >= current_ev and dcf_ev > 0:
            implied_irr = wacc
            result["assumptions"]["IRR Method"] = "Using WACC (below or at DCF value)"
        else:
            implied_irr = wacc
            result["assumptions"]["IRR Method"] = "Using WACC (default, DCF value unavailable)"

    # Cap IRR at reasonable bounds (50% max, -10% min)
    if implied_irr is not None:
        implied_irr = max(min(implied_irr, 0.50), -0.10)

    result["expected_return"] = implied_irr
    if implied_irr is not None:
        result["expected_return_label"] = f"{implied_irr * 100:.2f}%"
    else:
        result["expected_return_label"] = "N/A"

    result["current_enterprise_value"] = current_ev
    result["assumptions"]["Current Enterprise Value"] = f"${current_ev:,.0f}"
    result["assumptions"]["Terminal Growth Rate"] = f"{terminal_growth * 100:.2f}%"
    result["assumptions"]["Projection Period"] = f"{projection_years} years"
    result["assumptions"]["DCF Value at WACC"] = f"${dcf_ev:,.0f}"
    if implied_irr:
        result["assumptions"]["Implied IRR"] = f"{implied_irr * 100:.2f}%"

    return result


def _find_irr(npv_func, guess: float, max_iter: int = 100, tol: float = 1e-6) -> float | None:
    """Newton-Raphson solver for IRR."""
    rate = guess
    for _ in range(max_iter):
        f = npv_func(rate)
        if abs(f) < tol:
            return rate
        # Numerical derivative
        eps = max(1e-6, abs(rate) * 1e-4)
        df = (npv_func(rate + eps) - npv_func(rate - eps)) / (2 * eps)
        if abs(df) < 1e-12:
            return None
        rate_new = rate - f / df
        if abs(rate_new - rate) < tol:
            return rate_new
        rate = rate_new
    return None
