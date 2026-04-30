"""
Method 1: Cost of Equity (CAPM + Country Risk Premium)

Formula:
    E(R) = Rf + β_levered × ERP_US + λ × CRP

Where:
- Rf        = Current 10Y US Treasury yield
- β_levered = Bottom-up unlevered beta (industry) → relevered for company's D/E
- ERP_US    = Damodaran's latest implied equity risk premium
- λ         = Revenue exposure to the country (default: 1.0)
- CRP       = Country risk premium from Damodaran's ctryprem file
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from . import company_data as cd
from . import damodaran_data as dd

log = logging.getLogger(__name__)


def compute(ticker_data: cd.CompanyData, lambda_: float = 1.0) -> dict[str, Any]:
    """Compute CAPM + CRP expected return for a company.

    Parameters
    ----------
    ticker_data : CompanyData
        Pre-loaded company data object.
    lambda_ : float
        Revenue exposure to the country (default 1.0, i.e. 100%).

    Returns
    -------
    dict with keys: expected_return, rf, unlevered_beta, levered_beta,
                    de_ratio, tax_rate, erp, crp, lambda_, assumptions
    """
    result: dict[str, Any] = {
        "method": "CAPM + CRP (Cost of Equity)",
        "expected_return": None,
        "expected_return_label": "N/A",
        "assumptions": {},
    }

    # 1. Risk-free rate
    rf = dd.get_treasury_10y()
    result["rf"] = rf
    result["assumptions"]["Risk-Free Rate (10Y Treasury)"] = f"{rf * 100:.2f}%"

    # 2. Industry beta → bottom-up unlevered beta
    industry = ticker_data.industry
    sector = ticker_data.sector
    industry_beta = dd.find_industry_beta(industry)

    # Fallback: try sector
    if industry_beta.get("unlevered_beta", dd.FALLBACK_GENERIC_BETA) == dd.FALLBACK_GENERIC_BETA and sector != "Unknown":
        industry_beta = dd.find_industry_beta(sector)

    unlevered_beta = industry_beta.get("unlevered_beta", dd.FALLBACK_GENERIC_BETA)
    industry_de_ratio = industry_beta.get("de_ratio", 0.25)
    tax_rate = industry_beta.get("tax_rate", dd.FALLBACK_TAX_RATE)

    result["industry_unlevered_beta"] = unlevered_beta
    result["assumptions"]["Industry Unlevered Beta"] = f"{unlevered_beta:.2f}"

    # 3. Relever beta using company's actual D/E
    company_de = ticker_data.debt_to_equity()
    if company_de is None or company_de == 0:
        de_ratio = industry_de_ratio
        result["assumptions"]["D/E Used"] = "Industry average"
    else:
        de_ratio = company_de
        result["assumptions"]["D/E Used"] = "Company actual"

    # Relevering formula: β_levered = β_unlevered × (1 + (1 - t) × D/E)
    levered_beta = unlevered_beta * (1 + (1 - tax_rate) * de_ratio)
    # Also compute with industry D/E for reference
    levered_beta_industry = unlevered_beta * (1 + (1 - tax_rate) * industry_de_ratio)

    result["de_ratio"] = de_ratio
    result["tax_rate"] = tax_rate
    result["levered_beta"] = levered_beta
    result["assumptions"]["Levered Beta"] = f"{levered_beta:.2f}"
    result["assumptions"]["Relevering Formula"] = f"β_u × (1 + (1-t)×D/E)"

    # 4. ERP
    erp = dd.get_implied_erp()
    result["erp"] = erp
    result["assumptions"]["ERP (Damodaran Implied)"] = f"{erp * 100:.2f}%"

    # 5. CRP
    country = ticker_data.country
    country_risk = dd.find_country_risk(country)
    crp = country_risk.get("crp", 0.0)
    result["crp"] = crp
    result["lambda"] = lambda_
    result["assumptions"]["Country"] = country
    result["assumptions"]["CRP"] = f"{crp * 100:.2f}%"
    result["assumptions"]["Lambda (Revenue Exposure)"] = f"{lambda_:.2f}"

    # 6. Final formula
    expected_return = rf + levered_beta * erp + lambda_ * crp
    result["expected_return"] = expected_return
    result["expected_return_label"] = f"{expected_return * 100:.2f}%"

    # Components breakdown
    us_equity_risk = levered_beta * erp
    crp_contribution = lambda_ * crp
    result["us_equity_risk_premium"] = us_equity_risk
    result["crp_contribution"] = crp_contribution
    result["assumptions"]["US Equity Risk Component (β×ERP)"] = f"{us_equity_risk * 100:.2f}%"
    result["assumptions"]["CRP Contribution (λ×CRP)"] = f"{crp_contribution * 100:.2f}%"
    result["formula"] = f"E(R) = {rf*100:.2f}% + {levered_beta:.2f}×{erp*100:.2f}% + {lambda_:.2f}×{crp*100:.2f}%"
    result["formula_result"] = f"{expected_return * 100:.2f}%"

    return result
