"""
Fetch, parse, and cache Aswath Damodaran's published datasets.

Sources
-------
- betas.xls   – Industry unlevered / levered betas
- histimpl.xls – Implied ERP time series
- ctryprem*.xlsx – Country risk premia
- wacc.xls    – WACC by industry
- margin.xls  – Margins by industry
- capex.xls   – Capex / reinvestment by industry
- fundgrEB.xls – Fundamental growth (ROE, retention)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

log = logging.getLogger(__name__)

# ── cache directory ──────────────────────────────────────────────────
CACHE_DIR = Path.home() / ".cache" / "expected_returns"

# ── base URL ─────────────────────────────────────────────────────────
BASE = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/"

# ── dataset descriptors ──────────────────────────────────────────────
DATASETS = {
    "betas": {"filename": "betas.xls", "source": BASE + "betas.xls"},
    "histimpl": {"filename": "histimpl.xls", "source": BASE + "histimpl.xls"},
    "ctryprem": {"filename": "ctryprem.xls", "source": BASE + "ctryprem.xls", "alt_source": BASE + "ctryprem.xlsx"},
    "wacc": {"filename": "wacc.xls", "source": BASE + "wacc.xls"},
    "margin": {"filename": "margin.xls", "source": BASE + "margin.xls"},
    "capex": {"filename": "capex.xls", "source": BASE + "capex.xls"},
    "fundgr": {"filename": "fundgrEB.xls", "source": BASE + "fundgrEB.xls"},
}

# ── hardcoded fallback defaults ──────────────────────────────────────
FALLBACK_ERP = 0.048
FALLBACK_TREASURY_10Y = 0.043
FALLBACK_ERP_YEARS = {"2024": 0.048, "2023": 0.051, "2022": 0.052, "2021": 0.045, "2020": 0.043}

FALLBACK_BETAS: dict[str, dict] = {
    "Technology (Software & Services)": {
        "unlevered_beta": 1.02, "levered_beta": 1.25, "de_ratio": 0.10, "tax_rate": 0.21, "num_firms": 500,
    }
}
FALLBACK_GENERIC_BETA = 1.0
FALLBACK_GENERIC_DE = 0.25
FALLBACK_TAX_RATE = 0.21
FALLBACK_INDUSTRY_MARGIN = 0.15
FALLBACK_SALES_CAPITAL_RATIO = 1.5
FALLBACK_INDUSTRY_WACC = 0.09

FALLBACK_CTRYPREM: dict[str, dict] = {
    "United States": {"crp": 0.0, "default_spread": 0.0, "equity_ratio": 1.0},
    "Japan": {"crp": 0.0068, "default_spread": 0.0045, "equity_ratio": 1.5},
    "China": {"crp": 0.0125, "default_spread": 0.0075, "equity_ratio": 1.67},
    "India": {"crp": 0.0180, "default_spread": 0.0110, "equity_ratio": 1.64},
    "United Kingdom": {"crp": 0.0035, "default_spread": 0.0025, "equity_ratio": 1.4},
    "Germany": {"crp": 0.0015, "default_spread": 0.0010, "equity_ratio": 1.5},
    "France": {"crp": 0.0015, "default_spread": 0.0010, "equity_ratio": 1.5},
    "Brazil": {"crp": 0.0210, "default_spread": 0.0120, "equity_ratio": 1.75},
    "Australia": {"crp": 0.0020, "default_spread": 0.0015, "equity_ratio": 1.3},
    "Canada": {"crp": 0.0020, "default_spread": 0.0015, "equity_ratio": 1.3},
    "Singapore": {"crp": 0.0025, "default_spread": 0.0020, "equity_ratio": 1.3},
    "South Korea": {"crp": 0.0070, "default_spread": 0.0050, "equity_ratio": 1.4},
    "Taiwan": {"crp": 0.0060, "default_spread": 0.0040, "equity_ratio": 1.5},
    "Switzerland": {"crp": 0.0010, "default_spread": 0.0005, "equity_ratio": 2.0},
    "Netherlands": {"crp": 0.0010, "default_spread": 0.0005, "equity_ratio": 2.0},
}

FALLBACK_INDUSTRY_MARGINS: dict[str, float] = {
    "Technology (Software & Services)": 0.22, "Technology (Hardware & Equipment)": 0.14,
    "Financial Services (Banking)": 0.30, "Financial Services (Insurance)": 0.12,
    "Healthcare (Pharmaceuticals & Biotech)": 0.22, "Healthcare (Medical Equipment & Supplies)": 0.18,
    "Consumer Staples": 0.10, "Consumer Discretionary": 0.09,
    "Energy (Oil & Gas)": 0.12, "Energy (Renewable & Green)": 0.08,
    "Industrial": 0.11, "Materials": 0.13,
    "Real Estate (Development & Operations)": 0.35, "Real Estate (REITs)": 0.40,
    "Telecommunication Services": 0.16, "Utilities (General)": 0.16,
}

FALLBACK_SALES_CAPITAL: dict[str, float] = {
    "Technology (Software & Services)": 2.00, "Technology (Hardware & Equipment)": 2.20,
    "Financial Services (Banking)": 0.50, "Financial Services (Insurance)": 0.40,
    "Healthcare (Pharmaceuticals & Biotech)": 2.50, "Consumer Staples": 2.00,
    "Consumer Discretionary": 2.30, "Energy (Oil & Gas)": 0.80,
    "Industrial": 1.80, "Materials": 1.50, "Retail": 3.00,
}

FALLBACK_WACC: dict[str, float] = {
    "Technology (Software & Services)": 0.10, "Technology (Hardware & Equipment)": 0.09,
    "Financial Services (Banking)": 0.08, "Financial Services (Insurance)": 0.07,
    "Healthcare (Pharmaceuticals & Biotech)": 0.09, "Consumer Staples": 0.07,
    "Consumer Discretionary": 0.08, "Energy (Oil & Gas)": 0.09,
    "Industrial": 0.08, "Materials": 0.08, "Retail": 0.08,
}


# ── helpers ──────────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"


def _cache_is_fresh(name: str) -> bool:
    p = _cache_path(name)
    if not p.exists():
        return False
    mtime = datetime.fromtimestamp(p.stat().st_mtime)
    return (datetime.now() - mtime) < timedelta(hours=24)


def _download(url: str, filename: str, timeout: float = 30) -> bytes | None:
    """Download a file; returns raw bytes or None."""
    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (compatible; DamodaranExpectedReturns/0.1)"
        })
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        log.warning("Failed to download %s: %s", url, exc)
        return None


def _to_safe_strs(series: pd.Series) -> list[str]:
    """Convert a pandas Series (row) to a list of cleaned lowercase strings,
    replacing NaN with empty string so join() never chokes."""
    return [
        str(v).strip().lower() if pd.notna(v) else ""
        for v in series
    ]


def _find_sheet_with_data(xl: pd.ExcelFile, min_cols: int = 3) -> str | None:
    """Find the sheet name with the most columns—probably the data sheet."""
    best, best_name = 0, None
    for sheet in xl.sheet_names:
        try:
            df = pd.read_excel(xl, sheet_name=sheet, header=None, nrows=10)
            if df.shape[1] >= best:
                best, best_name = df.shape[1], sheet
        except Exception:
            continue
    return best_name


def _find_header_row(df: pd.DataFrame, keywords: list[str], max_scan: int = 20) -> int | None:
    """Scan the first `max_scan` rows for one that contains all keywords."""
    for i in range(min(max_scan, len(df))):
        vals = _to_safe_strs(df.iloc[i])
        joined = " ".join(vals)
        if all(kw in joined for kw in keywords):
            return i
    return None


def _col_index(df: pd.DataFrame, text: str) -> int | None:
    """Find the index of a column whose name contains `text`."""
    for i, c in enumerate(df.columns):
        if text.lower() in str(c).lower():
            return i
    return None


def _numeric(val: Any) -> float | None:
    """Safely convert a cell value to float."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _set_columns_safe(df: pd.DataFrame, row_idx: int) -> list[str]:
    """Convert a row of mixed types to safe string column names for a DataFrame."""
    safe = []
    for v in df.iloc[row_idx]:
        if isinstance(v, str):
            safe.append(v.strip())
        elif pd.notna(v):
            safe.append(str(v).strip())
        else:
            safe.append("")
    return safe


def _save_cache(name: str, data: Any) -> None:
    with open(_cache_path(name), "w") as f:
        json.dump(data, f, indent=2, default=str)


def _load_cache(name: str) -> Any | None:
    p = _cache_path(name)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ── country-specific constants ──────────────────────────────────────

COUNTRY_TAX_RATES: dict[str, float] = {
    "United States": 0.21, "Canada": 0.15, "Mexico": 0.30,
    "United Kingdom": 0.25, "Germany": 0.30, "France": 0.25,
    "Italy": 0.24, "Spain": 0.25, "Netherlands": 0.26,
    "Switzerland": 0.18, "Sweden": 0.21, "Norway": 0.22,
    "Denmark": 0.22, "Finland": 0.20, "Ireland": 0.13,
    "Japan": 0.30, "South Korea": 0.25, "Singapore": 0.17,
    "China": 0.25, "India": 0.30, "Taiwan": 0.20,
    "Hong Kong": 0.17, "Australia": 0.25, "New Zealand": 0.28,
    "Brazil": 0.34, "Russia": 0.20, "South Africa": 0.28,
    "Saudi Arabia": 0.20, "UAE": 0.09, "Israel": 0.23,
    "Argentina": 0.35, "Turkey": 0.25, "Indonesia": 0.22,
    "Malaysia": 0.24, "Thailand": 0.20, "Philippines": 0.25,
    "Vietnam": 0.20, "Egypt": 0.23, "Nigeria": 0.30,
}


TERMINAL_GROWTH_BY_CRP: dict[str, float] = {
    "very_low": 0.025,   # CRP ≤ 0.5% (US, Switzerland)
    "low": 0.028,        # CRP 0.5-1.5% (Europe, Japan, Aus)
    "medium": 0.035,     # CRP 1.5-3.0% (China, India, Brazil)
    "high": 0.045,       # CRP 3-5% (higher-risk emerging)
    "very_high": 0.055,  # CRP > 5% (Argentina, Turkey)
}


def find_country_tax_rate(country_name: str) -> float:
    """Get statutory corporate tax rate for a country."""
    if country_name in COUNTRY_TAX_RATES:
        return COUNTRY_TAX_RATES[country_name]
    # Fuzzy match
    cl = country_name.lower()
    for k, v in COUNTRY_TAX_RATES.items():
        if cl in k.lower() or k.lower() in cl:
            return v
    return FALLBACK_TAX_RATE


def terminal_growth_for_country(country_name: str) -> float:
    """Derive a reasonable terminal growth rate based on country risk.
    
    Higher-risk countries have higher growth potential (catch-up effect,
    demographic growth, higher inflation). We use CRP as a proxy.
    """
    crp_data = find_country_risk(country_name)
    crp_val = crp_data.get("crp", 0.0) or 0.0

    if crp_val <= 0.005:
        return TERMINAL_GROWTH_BY_CRP["very_low"]
    elif crp_val <= 0.015:
        return TERMINAL_GROWTH_BY_CRP["low"]
    elif crp_val <= 0.03:
        return TERMINAL_GROWTH_BY_CRP["medium"]
    elif crp_val <= 0.05:
        return TERMINAL_GROWTH_BY_CRP["high"]
    else:
        return TERMINAL_GROWTH_BY_CRP["very_high"]


# ── public API ───────────────────────────────────────────────────────

def refresh_all() -> dict[str, bool]:
    """Download and parse every dataset. Returns {name: success}."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, desc in DATASETS.items():
        success = False
        for attempt_url in [desc["source"]] + ([desc["alt_source"]] if "alt_source" in desc else []):
            raw = _download(attempt_url, desc["filename"])
            if raw:
                break
        if not raw:
            log.warning("All downloads failed for %s; using cached or fallback", name)
            continue

        tmp = CACHE_DIR / f"_{desc['filename']}"
        tmp.write_bytes(raw)
        parsed = _parse_dataset(name, tmp)
        if parsed is not None:
            _save_cache(name, parsed)
            success = True
        tmp.unlink(missing_ok=True)
        results[name] = success
    return results


def _parse_dataset(name: str, path: Path) -> Any:
    parsers = {
        "betas": _parse_betas,
        "histimpl": _parse_histimpl,
        "ctryprem": _parse_ctryprem,
        "wacc": _parse_wacc,
        "margin": _parse_margin,
        "capex": _parse_capex_inv,
        "fundgr": _parse_fundgr,
    }
    parser = parsers.get(name)
    if not parser:
        return None
    try:
        return parser(path)
    except Exception as exc:
        log.warning("Parser error for %s: %s", name, exc, exc_info=True)
        return None


# ── individual parsers ───────────────────────────────────────────────

def _read_xl_sheet(path: Path, preferred_sheet: str | None = None) -> pd.DataFrame:
    """Open an xls/xlsx file and return the most likely data sheet as a 
    clean DataFrame with string column names."""
    engine = "xlrd" if str(path).endswith(".xls") else "openpyxl"
    xl = pd.ExcelFile(path, engine=engine)
    sheet = preferred_sheet if preferred_sheet and preferred_sheet in xl.sheet_names else _find_sheet_with_data(xl)
    if sheet is None:
        sheet = xl.sheet_names[0]
    df = pd.read_excel(xl, sheet_name=sheet, header=None)
    return df


def _parse_betas(path: Path) -> dict:
    """Parse betas.xls → dict of industry_name → {unlevered_beta, levered_beta, de_ratio, tax_rate, num_firms}."""
    df = _read_xl_sheet(path, "Industry Averages")

    # Find the header row with "industry" + "beta"
    header_idx = _find_header_row(df, ["industry", "beta"])
    if header_idx is None:
        log.warning("Could not find betas header row, using fallback")
        return dict(FALLBACK_BETAS)

    df.columns = _set_columns_safe(df, header_idx)
    df = df.iloc[header_idx + 1:].reset_index(drop=True)
    df = df.dropna(how="all")

    # Identify column roles — run once, not per row
    unpinned: dict[str, str] = {}
    levered_col = d_e_col = tax_col = firms_col = None
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "unlevered beta":
            unpinned["unlevered_beta"] = c
        elif "corrected for cash" in cl:
            pass  # skip corrected column
        elif "number of firms" in cl:
            firms_col = c
        elif cl == "d/e ratio":
            d_e_col = c
        elif cl == "beta":
            levered_col = c
        elif "tax" in cl:
            tax_col = c

    result = {}
    for _, row in df.iterrows():
        ind_name = str(row[df.columns[0]]).strip()
        if not ind_name or ind_name.lower() in ("nan", "", "industry name"):
            continue

        entry = {"num_firms": 0}
        if unpinned.get("unlevered_beta"):
            entry["unlevered_beta"] = _numeric(row[unpinned["unlevered_beta"]]) or FALLBACK_GENERIC_BETA
        if levered_col:
            entry["levered_beta"] = _numeric(row[levered_col]) or FALLBACK_GENERIC_BETA
        if d_e_col:
            entry["de_ratio"] = _numeric(row[d_e_col]) or FALLBACK_GENERIC_DE
        if tax_col:
            entry["tax_rate"] = _numeric(row[tax_col]) or FALLBACK_TAX_RATE
        if firms_col:
            entry["num_firms"] = int(_numeric(row[firms_col]) or 0)
        entry.setdefault("de_ratio", FALLBACK_GENERIC_DE)
        entry.setdefault("tax_rate", FALLBACK_TAX_RATE)
        entry.setdefault("levered_beta", entry.get("unlevered_beta", FALLBACK_GENERIC_BETA))
        result[ind_name] = entry

    return result if result else dict(FALLBACK_BETAS)


def _parse_histimpl(path: Path) -> dict:
    """Parse histimpl.xls → {year: {erp, t10y}, 'latest_erp': ..., 'latest_10y': ...}."""
    df = _read_xl_sheet(path)

    # Find header: first cell is "year", and some column has "implied" + "premium"
    header_idx = None
    for i in range(min(25, len(df))):
        first_val = str(df.iloc[i, 0]).strip().lower() if pd.notna(df.iloc[i, 0]) else ""
        if first_val != "year":
            continue
        vals = [str(v).strip().lower() if pd.notna(v) else "" for v in df.iloc[i]]
        joined = " ".join(vals)
        if "implied" in joined and "premium" in joined:
            header_idx = i
            break
    if header_idx is None:
        header_idx = _find_header_row(df, ["year", "premium"])
    if header_idx is None:
        log.warning("Could not find histimpl header, using fallback")
        return {"latest_erp": FALLBACK_ERP, "latest_10y": FALLBACK_TREASURY_10Y, "years": dict(FALLBACK_ERP_YEARS)}

    # Convert header row to strings, replacing NaN/float with ""
    col_names = [str(c).strip() if isinstance(c, str) else str(c) if pd.notna(c) else "" for c in df.iloc[header_idx]]
    df.columns = col_names
    df = df.iloc[header_idx + 1:].reset_index(drop=True)
    df = df.dropna(how="all")

    # Find year and ERP columns
    year_col = erp_col = t10y_col = None
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in ("year", "date", ""):
            year_col = c
        elif "implied" in cl and "premium" in cl:
            erp_col = c
        elif "t.bond" in cl or "t bond" in cl or "tbond" in cl or "riskfree" in cl:
            t10y_col = c

    years = {}
    for _, row in df.iterrows():
        yr_val = _numeric(row[year_col]) if year_col else None
        if yr_val is None:
            continue
        year = str(int(yr_val))

        erp_val = _numeric(row[erp_col]) if erp_col else None
        if erp_val is not None and erp_val > 1:
            erp_val /= 100

        t10y_val = _numeric(row[t10y_col]) if t10y_col else None
        if t10y_val is not None and t10y_val > 1:
            t10y_val /= 100

        if erp_val is not None:
            years[year] = {"erp": erp_val, "t10y": t10y_val}

    # Find the most recent year's ERP as the latest
    sorted_years = sorted(years.keys())
    latest_erp = years[sorted_years[-1]]["erp"] if sorted_years else FALLBACK_ERP
    latest_t10y = years.get(sorted_years[-1], {}).get("t10y", FALLBACK_TREASURY_10Y)

    return {"years": years, "latest_erp": latest_erp, "latest_10y": latest_t10y}


def _parse_ctryprem(path: Path) -> dict:
    """Parse ctryprem → dict of country_name → {crp, default_spread, equity_ratio}."""
    # Try preferred sheets
    df = _read_xl_sheet(path, preferred_sheet="Regional Weighted Averages")
    if len(df.columns) < 3 or len(df) < 5:
        df = _read_xl_sheet(path, preferred_sheet="ERPs by country")
    
    # Find header row: look for "country" + "risk" + "premium" all in one row
    header_idx = _find_header_row(df, ["country", "risk", "premium"])
    if header_idx is None:
        header_idx = _find_header_row(df, ["country", "spread"])
    if header_idx is None:
        log.warning("Could not find ctryprem header, using fallback")
        return dict(FALLBACK_CTRYPREM)

    df.columns = _set_columns_safe(df, header_idx)
    df = df.iloc[header_idx + 1:].reset_index(drop=True)
    df = df.dropna(how="all")

    # Identify column roles
    country_col = default_spread_col = crp_col = None
    for c in df.columns:
        cl = c.lower().strip()
        if country_col is None and cl == "country":
            country_col = c
        elif "default" in cl and "spread" in cl:
            default_spread_col = c
        elif "country risk premium" in cl:
            crp_col = c
    if country_col is None:
        country_col = df.columns[0]

    result = {}
    for _, row in df.iterrows():
        name = str(row[country_col]).strip()
        if not name or name.lower() in ("nan", ""):
            continue

        entry: dict = {"crp": 0.0, "default_spread": 0.0, "equity_ratio": 1.5}
        
        if default_spread_col:
            ds = _numeric(row[default_spread_col])
            if ds is not None:
                if ds > 1:
                    ds /= 100
                entry["default_spread"] = ds
        
        if crp_col:
            crp = _numeric(row[crp_col])
            if crp is not None:
                if crp > 1:
                    crp /= 100
                entry["crp"] = crp

        # Skip regional aggregates and rows without a default spread
        _regional = {"africa", "asia", "europe", "north america", "south america",
                     "caribbean", "middle east", "eastern europe", "western europe",
                     "central and south america", "australia & new zealand",
                     "australia, nz & canada", "africa & mid east"}
        if entry.get("crp", 0) < 1 and name.lower() not in _regional:
            result[name] = entry

    return result if result else dict(FALLBACK_CTRYPREM)


def _parse_wacc(path: Path) -> dict:
    """Parse wacc.xls → dict of industry_name → {cost_of_equity, cost_of_debt, wacc}."""
    df = _read_xl_sheet(path)
    header_idx = _find_header_row(df, ["industry", "debt"])
    if header_idx is None:
        log.warning("Could not find wacc header, using fallback")
        return dict(FALLBACK_WACC)

    df.columns = _set_columns_safe(df, header_idx)
    df = df.iloc[header_idx + 1:].reset_index(drop=True)
    df = df.dropna(how="all")

    ind_col = _col_index(df, "industry")
    if ind_col is None:
        ind_col = 0
    cost_equity_col = _col_index(df, "cost of equity")
    cost_debt_col = _col_index(df, "after-tax cost of debt")
    wacc_col = _col_index(df, "cost of capital")

    result = {}
    for _, row in df.iterrows():
        name = str(row.iloc[ind_col]).strip()
        if not name or name.lower() in ("nan", ""):
            continue
        entry = {}
        for key, col_idx in [("cost_of_equity", cost_equity_col), ("cost_of_debt", cost_debt_col), ("wacc", wacc_col)]:
            if col_idx is not None:
                v = _numeric(row.iloc[col_idx])
                if v is not None and v > 1:
                    v /= 100
                entry[key] = v
        if entry:
            result[name] = entry

    return result or dict(FALLBACK_WACC)


def _parse_margin(path: Path) -> dict:
    """Parse margin.xls → dict of industry_name → {pre_tax_margin, after_tax_margin}."""
    df = _read_xl_sheet(path)
    header_idx = _find_header_row(df, ["industry", "margin"])
    if header_idx is None:
        return dict(FALLBACK_INDUSTRY_MARGINS)

    df.columns = _set_columns_safe(df, header_idx)
    df = df.iloc[header_idx + 1:].reset_index(drop=True)
    df = df.dropna(how="all")

    ind_col = _col_index(df, "industry") or 0
    pretax_col = _col_index(df, "pre-tax") or _col_index(df, "pre tax")
    aftertax_col = _col_index(df, "after-tax") or _col_index(df, "after tax")

    result = {}
    for _, row in df.iterrows():
        name = str(row.iloc[ind_col]).strip()
        if not name or name.lower() in ("nan", ""):
            continue
        entry = {}
        for key, col_idx in [("pre_tax_margin", pretax_col), ("after_tax_margin", aftertax_col)]:
            if col_idx is not None:
                v = _numeric(row.iloc[col_idx])
                if v is not None and v > 1:
                    v /= 100
                entry[key] = v
        if entry:
            result[name] = entry

    return result or dict(FALLBACK_INDUSTRY_MARGINS)


def _parse_capex_inv(path: Path) -> dict:
    """Parse capex.xls → dict of industry_name → {sales_capital, capex_capital}."""
    df = _read_xl_sheet(path)
    header_idx = _find_header_row(df, ["industry", "sales"])
    if header_idx is None:
        return dict(FALLBACK_SALES_CAPITAL)

    df.columns = _set_columns_safe(df, header_idx)
    df = df.iloc[header_idx + 1:].reset_index(drop=True)
    df = df.dropna(how="all")

    ind_col = _col_index(df, "industry") or 0
    sales_cap_col = _col_index(df, "invested capital") or _col_index(df, "sales/capital") or _col_index(df, "to capital")
    capex_sales_col = _col_index(df, "capex") or _col_index(df, "reinvestment")

    result = {}
    for _, row in df.iterrows():
        name = str(row.iloc[ind_col]).strip()
        if not name or name.lower() in ("nan", ""):
            continue
        entry = {}
        for key, col_idx in [("sales_capital", sales_cap_col), ("capex_sales", capex_sales_col)]:
            if col_idx is not None:
                v = _numeric(row.iloc[col_idx])
                if v is not None:
                    entry[key] = v
        if entry:
            result[name] = entry

    return result or dict(FALLBACK_SALES_CAPITAL)


def _parse_fundgr(path: Path) -> dict:
    """Parse fundgrEB.xls → dict of industry_name → {roe, retention} (uses ROC, Reinvestment Rate from file)."""
    df = _read_xl_sheet(path)
    header_idx = _find_header_row(df, ["industry", "roc"])
    if header_idx is None:
        return {}

    df.columns = _set_columns_safe(df, header_idx)
    df = df.iloc[header_idx + 1:].reset_index(drop=True)
    df = df.dropna(how="all")

    ind_col = _col_index(df, "industry") or 0
    roe_col = _col_index(df, "roc")  # File uses ROC, not ROE
    ret_col = _col_index(df, "reinvestment")  # File uses Reinvestment Rate
    growth_col = _col_index(df, "expected growth")

    result = {}
    for _, row in df.iterrows():
        name = str(row.iloc[ind_col]).strip()
        if not name or name.lower() in ("nan", ""):
            continue
        entry = {}
        for key, col_idx in [("roe", roe_col), ("retention", ret_col), ("growth", growth_col)]:
            if col_idx is not None:
                v = _numeric(row.iloc[col_idx])
                if v is not None and v > 1:
                    v /= 100
                entry[key] = v
        if entry:
            result[name] = entry

    return result


# ── industry matching ────────────────────────────────────────────────

# Centralized industry-to-industry-name mapping for cross-dataset matching
# Order matters: more specific terms first so broader ones don't steal the match
_INDUSTRY_CATEGORIES: list[tuple[str, str]] = [
    ("consumer electronics", "Electronics (Consumer & Office)"),
    ("electronics", "Electronics (General)"),
    ("computers/peripherals", "Computers/Peripherals"),
    ("computer service", "Computer Services"),
    ("semiconductor equip", "Semiconductor Equip"),
    ("semiconductor", "Semiconductor"),
    ("software", "Technology (Software & Services)"),
    ("tech", "Technology (Software & Services)"),
    ("bank", "Financial Services (Banking)"),
    ("insurance", "Financial Services (Insurance)"),
    ("biotech", "Drugs (Biotechnology)"),
    ("pharma", "Drugs (Biotechnology)"),
    ("healthcare", "Heathcare Information and Technology"),
    ("hospital", "Healthcare Support Services"),
    ("real estate", "Real Estate (Development & Operations)"),
    ("reit", "Real Estate (REITs)"),
    ("telecom", "Telecom. Services"),
    ("utility", "Utility (General)"),
    ("oil", "Oil/Gas (Production and Exploration)"),
    ("energy", "Oil/Gas (Production and Exploration)"),
    ("transport", "Transportation"),
    ("airline", "Air Transport"),
    ("retail", "Retail (Special Lines)"),
    ("consumer", "Business & Consumer Services"),
    ("food", "Food Processing"),
    ("beverage", "Beverage (Soft)"),
    ("industrial", "Industrial"),
    ("manufacturing", "Industrial"),
    ("machinery", "Industrial"),
    ("material", "Materials"),
    ("chemical", "Chemical (Specialty)"),
    ("metal", "Metals & Mining"),
    ("hotel", "Hotel/Gaming"),
    ("gaming", "Hotel/Gaming"),
    ("media", "Entertainment"),
    ("entertainment", "Entertainment"),
    ("auto", "Automotive"),
    ("automotive", "Automotive"),
]


def _match_industry_key(ind_lower: str, data_dict: dict) -> str | None:
    """Find best-matching key in data_dict for a lowercased industry name using category rules."""
    for kw, target in _INDUSTRY_CATEGORIES:
        if kw in ind_lower:
            if target in data_dict:
                return target
            for k in data_dict:
                if kw in k.lower():
                    return k
            break
    return None


def find_industry_beta(industry_name: str) -> dict:
    """Fuzzy-match industry_name against betas data, returning the best match entry."""
    betas = get_betas()
    if not betas:
        return {"unlevered_beta": FALLBACK_GENERIC_BETA, "de_ratio": FALLBACK_GENERIC_DE, "tax_rate": FALLBACK_TAX_RATE}

    if industry_name in betas:
        return betas[industry_name]

    ind_lower = industry_name.lower()
    for key, val in betas.items():
        if ind_lower in key.lower() or key.lower() in ind_lower:
            return val

    matched_key = _match_industry_key(ind_lower, betas)
    if matched_key:
        return betas[matched_key]
    first_key = next(iter(betas))
    return betas[first_key]


# ── public getters ───────────────────────────────────────────────────

def get_erp() -> dict:
    data = _load_cache("histimpl")
    if data:
        return data
    refresh_all()
    data = _load_cache("histimpl")
    return data or {"latest_erp": FALLBACK_ERP, "latest_10y": FALLBACK_TREASURY_10Y, "years": dict(FALLBACK_ERP_YEARS)}


def get_treasury_10y() -> float:
    """Get latest 10-year Treasury rate from ERP data."""
    erp = get_erp()
    return erp.get("latest_10y", FALLBACK_TREASURY_10Y)


def get_ctryprem() -> dict:
    data = _load_cache("ctryprem")
    if not data:
        refresh_all()
        data = _load_cache("ctryprem")
    return data or dict(FALLBACK_CTRYPREM)


def get_betas() -> dict:
    data = _load_cache("betas")
    if data:
        return data
    refresh_all()
    data = _load_cache("betas")
    return data or dict(FALLBACK_BETAS)


def get_implied_erp() -> float:
    """Get the latest implied ERP value."""
    erp = get_erp()
    return erp.get("latest_erp", FALLBACK_ERP)


def find_country_risk(country_name: str) -> dict:
    """Look up country risk premium for a country name."""
    ctry = get_ctryprem()
    # Direct lookup
    if country_name in ctry:
        return ctry[country_name]
    # Fuzzy match
    cl = country_name.lower()
    for k, v in ctry.items():
        if cl in k.lower() or k.lower() in cl:
            return v
    return {"crp": 0.0, "default_spread": 0.0, "equity_ratio": 1.5}


def get_industry_wacc() -> dict:
    data = _load_cache("wacc")
    if not data:
        refresh_all()
        data = _load_cache("wacc")
    return data or dict(FALLBACK_WACC)


def find_industry_wacc(industry_name: str) -> float:
    """Find WACC for an industry, with fuzzy matching."""
    wacc_data = get_industry_wacc()
    if industry_name in wacc_data:
        return wacc_data[industry_name].get("wacc", FALLBACK_INDUSTRY_WACC)
    il = industry_name.lower()
    for k, v in wacc_data.items():
        if il in k.lower() or k.lower() in il:
            return v.get("wacc", FALLBACK_INDUSTRY_WACC)
    matched = _match_industry_key(il, wacc_data)
    if matched:
        return wacc_data[matched].get("wacc", FALLBACK_INDUSTRY_WACC)
    return FALLBACK_INDUSTRY_WACC


def get_industry_margins() -> dict:
    data = _load_cache("margin")
    if not data:
        refresh_all()
        data = _load_cache("margin")
    return data or dict(FALLBACK_INDUSTRY_MARGINS)


def find_industry_margin(industry_name: str) -> float | None:
    """Find operating margin for an industry, with fuzzy matching."""
    margins = get_industry_margins()
    if industry_name in margins:
        return margins[industry_name]
    il = industry_name.lower()
    for k, v in margins.items():
        if il in k.lower() or k.lower() in il:
            return v
    matched = _match_industry_key(il, margins)
    if matched:
        return margins[matched]
    return None


def find_sales_capital_ratio(industry_name: str) -> float | None:
    """Find Sales/Capital ratio for an industry."""
    data = get_sales_capital()
    if industry_name in data:
        return data[industry_name].get("sales_capital")
    il = industry_name.lower()
    for k, v in data.items():
        if il in k.lower() or k.lower() in il:
            return v.get("sales_capital")
    matched = _match_industry_key(il, data)
    if matched:
        return data[matched].get("sales_capital")
    return None


def get_sales_capital() -> dict:
    data = _load_cache("capex")
    if not data:
        refresh_all()
        data = _load_cache("capex")
    return data or dict(FALLBACK_SALES_CAPITAL)


def get_fundamental_growth() -> dict:
    data = _load_cache("fundgr")
    if not data:
        refresh_all()
        data = _load_cache("fundgr")
    return data or {}


def find_fundamental_growth(industry_name: str) -> dict | None:
    """Find fundamental growth (ROC, reinvestment) for an industry."""
    data = get_fundamental_growth()
    if industry_name in data:
        return data[industry_name]
    il = industry_name.lower()
    for k, v in data.items():
        if il in k.lower() or k.lower() in il:
            return v
    matched = _match_industry_key(il, data)
    if matched:
        return data[matched]
    return None


def get_cache_stats() -> dict:
    stats = {}
    for name in DATASETS:
        p = _cache_path(name)
        stats[name] = {"cached": p.exists(), "age_hours": None}
        if p.exists():
            age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
            stats[name]["age_hours"] = round(age.total_seconds() / 3600, 1)
    return stats


def ensure_data_fresh() -> None:
    """Run once on import: refresh any cache that's stale."""
    stale = [name for name in DATASETS if not _cache_is_fresh(name)]
    if stale:
        log.info("Refreshing stale datasets: %s", stale)
        refresh_all()


# Run once on import
CACHE_DIR.mkdir(parents=True, exist_ok=True)
ensure_data_fresh()
