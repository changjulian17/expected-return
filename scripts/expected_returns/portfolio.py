"""
Portfolio / Sector-level aggregation of expected returns.

Takes a list of company results and computes:
- Simple average
- Market-cap-weighted average
- Equal-weighted portfolio
- Industry/Sector breakdowns
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate individual expected return results into portfolio view.

    Parameters
    ----------
    results : list[dict]
        Each dict should contain at least:
        - ticker
        - company_name
        - method name keys like 'capm', 'total_return', 'dcf'
        - market_cap (for weighting)

    Returns
    -------
    dict with portfolio-level summary statistics.
    """
    if not results:
        return {"error": "No results to aggregate"}

    portfolio: dict[str, Any] = {
        "num_companies": len(results),
        "companies": [r.get("ticker", r.get("company_name", "Unknown")) for r in results],
        "capm": _aggregate_method(results, "capm"),
        "total_return": _aggregate_method(results, "total_return"),
        "dcf": _aggregate_method(results, "dcf"),
        "sector_breakdown": _sector_breakdown(results),
    }

    # Overall average across all methods
    methods = ["capm", "total_return", "dcf"]
    all_returns = []
    for r in results:
        for m in methods:
            er = r.get(m, {}).get("expected_return")
            if er is not None:
                all_returns.append(er)
    if all_returns:
        portfolio["average_all_methods"] = float(np.mean(all_returns))
        portfolio["median_all_methods"] = float(np.median(all_returns))
        portfolio["min_all_methods"] = float(np.min(all_returns))
        portfolio["max_all_methods"] = float(np.max(all_returns))

    return portfolio


def _aggregate_method(results: list[dict], method: str) -> dict[str, Any]:
    """Aggregate a single method across all companies."""
    returns = []
    mcs = []
    for r in results:
        data = r.get(method, {})
        er = data.get("expected_return")
        mc = r.get("market_cap")
        if er is not None:
            returns.append(er)
            mcs.append(mc if mc else 0)

    if not returns:
        return {"avg": None, "median": None, "min": None, "max": None, "count": 0}

    avg = float(np.mean(returns))
    median = float(np.median(returns))
    min_val = float(np.min(returns))
    max_val = float(np.max(returns))

    # Weighted by market cap
    total_mc = sum(mcs)
    if total_mc > 0:
        weighted = sum(r * w for r, w in zip(returns, mcs)) / total_mc
    else:
        weighted = avg

    return {
        "avg": avg,
        "avg_label": f"{avg * 100:.2f}%",
        "weighted_avg": weighted,
        "weighted_avg_label": f"{weighted * 100:.2f}%",
        "median": median,
        "median_label": f"{median * 100:.2f}%",
        "min": min_val,
        "min_label": f"{min_val * 100:.2f}%",
        "max": max_val,
        "max_label": f"{max_val * 100:.2f}%",
        "count": len(returns),
    }


def _sector_breakdown(results: list[dict]) -> dict[str, Any]:
    """Break down by sector."""
    sectors: dict[str, list[dict]] = {}
    for r in results:
        sector = r.get("sector", "Unknown")
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(r)

    breakdown = {}
    for sector, members in sectors.items():
        capm_rets = [m.get("capm", {}).get("expected_return") for m in members]
        capm_rets = [r for r in capm_rets if r is not None]
        total_rets = [m.get("total_return", {}).get("expected_return") for m in members]
        total_rets = [r for r in total_rets if r is not None]
        dcf_rets = [m.get("dcf", {}).get("expected_return") for m in members]
        dcf_rets = [r for r in dcf_rets if r is not None]

        breakdown[sector] = {
            "count": len(members),
            "companies": [m.get("ticker", m.get("company_name")) for m in members],
            "avg_capm": float(np.mean(capm_rets)) if capm_rets else None,
            "avg_total_return": float(np.mean(total_rets)) if total_rets else None,
            "avg_dcf": float(np.mean(dcf_rets)) if dcf_rets else None,
            "avg_capm_label": f"{float(np.mean(capm_rets))*100:.2f}%" if capm_rets else "N/A",
            "avg_total_return_label": f"{float(np.mean(total_rets))*100:.2f}%" if total_rets else "N/A",
            "avg_dcf_label": f"{float(np.mean(dcf_rets))*100:.2f}%" if dcf_rets else "N/A",
        }

    return breakdown
