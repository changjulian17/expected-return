"""
Fetch company financial data via yfinance.

Provides:
- Ticker metadata (name, sector, industry, country)
- Financial statements (income, balance sheet, cash flow)
- Market data (price, shares outstanding, market cap, dividends)
- Historical revenue growth rates
- ROE, retention ratio, buyback yield
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import yfinance as yf

log = logging.getLogger(__name__)


class CompanyData:
    """Container for all company-level data needed by expected return models."""

    def __init__(self, ticker: str, country: str | None = None):
        self.ticker = ticker.upper()
        self._country_override = country
        self._ticker_obj = yf.Ticker(ticker)
        self._info: dict = {}
        self._financials: dict[str, Any] = {}
        self._loaded = False
        self.load()

    def load(self) -> bool:
        """Load all data. Returns True if successful."""
        try:
            info = self._ticker_obj.info or {}
            self._info = info

            # Financials
            inc = self._ticker_obj.income_stmt
            self._financials["income"] = inc if inc is not None and not inc.empty else None
            bal = self._ticker_obj.balance_sheet
            self._financials["balance"] = bal if bal is not None and not bal.empty else None
            cf = self._ticker_obj.cashflow
            self._financials["cashflow"] = cf if cf is not None and not cf.empty else None
            qi = self._ticker_obj.quarterly_income_stmt
            self._financials["quarterly_income"] = qi if qi is not None and not qi.empty else None

            self._loaded = True
            return True
        except Exception as exc:
            log.warning("Failed to load data for %s: %s", self.ticker, exc)
            return False

    # ── metadata ─────────────────────────────────────────────────────

    @property
    def company_name(self) -> str:
        return self._info.get("longName") or self._info.get("shortName") or self.ticker

    @property
    def sector(self) -> str:
        return self._info.get("sector") or "Unknown"

    @property
    def industry(self) -> str:
        return self._info.get("industry") or self._info.get("industryKey") or "Unknown"

    @property
    def country(self) -> str:
        if self._country_override:
            return self._country_override
        return self._info.get("country") or "US"

    # ── market data ──────────────────────────────────────────────────

    @property
    def market_cap(self) -> float | None:
        mc = self._info.get("marketCap")
        return float(mc) if mc else None

    @property
    def enterprise_value(self) -> float | None:
        ev = self._info.get("enterpriseValue")
        mc = self.market_cap
        if ev:
            ev_f = float(ev)
            # Sanity check: EV shouldn't be wildly different from market cap for most companies
            # Financial companies (banks, auto financing) can have EV >> MC due to large debt
            if mc and ev_f > mc * 100:
                # Check revenue: if revenue is also huge, this is real (e.g., Toyota)
                rev = self.revenue
                if rev and ev_f / rev > 50:
                    # EV/revenue > 50x is suspicious even for financial cos
                    log.debug("EV $%.0f > 100x MCap $%.0f and EV %.0fx rev for %s, using MC + net debt", 
                              ev_f, mc, ev_f/rev, self.ticker)
                    debt = min(self.total_debt or 0, mc * 10)  # cap debt at 10x MC
                    cash = self.cash_and_equivalents or 0
                    ev_f = mc + debt - cash
                else:
                    # High debt is real (financial company)
                    log.debug("EV $%.0f is high but revenue is proportionally large for %s (financial co)", ev_f, self.ticker)
            elif mc and ev_f > mc * 50:
                # No revenue check available, but EV/MC is very high
                debt = min(self.total_debt or 0, mc * 10)
                cash = self.cash_and_equivalents or 0
                ev_f = mc + debt - cash
            return ev_f
        return None

    @property
    def current_price(self) -> float | None:
        p = self._info.get("currentPrice") or self._info.get("regularMarketPrice")
        return float(p) if p else None

    @property
    def shares_outstanding(self) -> float | None:
        so = self._info.get("sharesOutstanding")
        return float(so) if so else None

    @property
    def beta(self) -> float | None:
        b = self._info.get("beta")
        return float(b) if b else None

    # ── balance sheet ────────────────────────────────────────────────

    def _get_bs(self, field: str, default: Any = None) -> Any:
        if self._financials.get("balance") is not None and not self._financials["balance"].empty:
            try:
                val = self._financials["balance"].loc[field].iloc[0]
                return float(val) if val is not None and not np.isnan(float(val)) else default
            except (KeyError, IndexError, ValueError, TypeError):
                pass
        # Try quarterly
        try:
            qbs = self._ticker_obj.quarterly_balance_sheet
            if qbs is not None and not qbs.empty:
                val = qbs.loc[field].iloc[0]
                return float(val) if val is not None and not np.isnan(float(val)) else default
        except Exception:
            pass
        return default

    @property
    def total_debt(self) -> float | None:
        """Total debt = Long Term Debt + Short Term Debt."""
        lt = self._get_bs("Long Term Debt", 0)
        st = self._get_bs("Short Term Debt", 0)
        # Try alternative names
        if lt == 0 and st == 0:
            lt = self._get_bs("LongTermDebt", 0)
            st = self._get_bs("ShortTermDebt", 0)
        if lt == 0 and st == 0:
            lt = self._get_bs("Total Debt", 0)
            return float(lt) if lt else None
        total = (lt or 0) + (st or 0)
        return total if total > 0 else None

    @property
    def total_equity(self) -> float | None:
        eq = self._get_bs("Stockholders Equity", None)
        if eq is None:
            eq = self._get_bs("Total Equity", None)
        if eq is None:
            eq = self._get_bs("Common Stock Equity", None)
        return float(eq) if eq else None

    @property
    def cash_and_equivalents(self) -> float | None:
        c = self._get_bs("Cash And Cash Equivalents", None)
        if c is None:
            c = self._get_bs("Cash", None)
        return float(c) if c else None

    def debt_to_equity(self) -> float | None:
        debt = self.total_debt
        eq = self.total_equity
        if debt is not None and eq is not None and eq > 0:
            return debt / eq
        return None

    # ── income statement ─────────────────────────────────────────────

    def _get_is(self, field: str, default: Any = None, period: str = "annual") -> Any:
        src = self._financials.get("income")
        if period == "quarterly":
            src = self._financials.get("quarterly_income")
        if src is not None and not src.empty:
            try:
                val = src.loc[field].iloc[0]
                return float(val) if val is not None and not np.isnan(float(val)) else default
            except (KeyError, IndexError, ValueError, TypeError):
                pass
        # Try finding alternate names
        if src is not None and not src.empty:
            for idx_name in src.index:
                if field.lower() in str(idx_name).lower():
                    try:
                        val = src.loc[idx_name].iloc[0]
                        return float(val) if val is not None and not np.isnan(float(val)) else default
                    except (ValueError, TypeError):
                        pass
        return default

    @property
    def revenue(self) -> float | None:
        rev = self._get_is("Total Revenue", None)
        if rev is None:
            rev = self._get_is("Revenue", None)
        return rev

    @property
    def operating_income(self) -> float | None:
        oi = self._get_is("Operating Income", None)
        if oi is None:
            oi = self._get_is("EBIT", None)
        return oi

    @property
    def net_income(self) -> float | None:
        return self._get_is("Net Income", None)

    @property
    def interest_expense(self) -> float | None:
        return self._get_is("Interest Expense", 0)

    @property
    def tax_provision(self) -> float | None:
        return self._get_is("Tax Provision", None)

    @property
    def depreciation(self) -> float | None:
        d = self._get_is("Depreciation And Amortization", None)
        if d is None:
            d = self._get_is("Depreciation", None)
        return d

    @property
    def operating_margin(self) -> float | None:
        rev = self.revenue
        oi = self.operating_income
        if rev and oi and rev > 0:
            return oi / rev
        return None

    @property
    def net_margin(self) -> float | None:
        rev = self.revenue
        ni = self.net_income
        if rev and ni and rev > 0:
            return ni / rev
        return None

    @property
    def effective_tax_rate(self) -> float | None:
        """Effective tax rate from income statement."""
        tax = self.tax_provision
        ni = self.net_income
        oi = self.operating_income
        if tax is not None and oi is not None and oi > 0:
            rate = abs(tax) / oi
            return min(rate, 0.35)
        return 0.21

    @property
    def dividends_paid(self) -> float | None:
        """TTM dividends."""
        div = self._get_is("Dividends Paid", None)
        if div is None:
            div = self._get_is("Common Stock Dividend Paid", None)
        return abs(div) if div else None

    @property
    def dividend_yield(self) -> float | None:
        """TTM dividend yield. Uses dividendRate/price for reliability."""
        # Method 1: dividendRate / currentPrice (most reliable)
        rate = self._info.get("dividendRate")
        price = self.current_price
        if rate is not None and price and price > 0:
            dy = float(rate) / price
            if 0 < dy < 0.25:
                return dy

        # Method 2: trailingAnnualDividendYield
        ty = self._info.get("trailingAnnualDividendYield")
        if ty is not None:
            dy = float(ty)
            if 0 < dy < 0.25:
                return dy

        # Method 3: dividendYield field (may be percentage or decimal)
        dy = self._info.get("dividendYield")
        if dy is not None:
            dy_f = float(dy)
            if dy_f > 1.0:
                dy_f = dy_f / 100.0
            if 0 < dy_f <= 0.25:
                return dy_f

        # Fallback: compute from dividends / market cap
        div = self.dividends_paid
        mc = self.market_cap
        if div and mc and mc > 0:
            dy_f = div / mc
            if 0 < dy_f <= 0.25:
                return dy_f
        return 0.0

    @property
    def buyback_yield(self) -> float | None:
        """Estimate buyback yield from shares outstanding change or from cash flow."""
        # Method 1: Shares outstanding change over 1yr
        try:
            shares_now = self.shares_outstanding
            if shares_now:
                # Try to get last year's shares from info
                shares_1y = self._info.get("sharesOutstanding")
                if shares_1y and shares_1y != shares_now:
                    reduction = (shares_1y - shares_now) / shares_1y
                    if reduction > 0:
                        return min(reduction, 0.15)
        except Exception:
            pass

        # Method 2: From cash flow statement - Common Stock Repurchase
        cf = self._financials.get("cashflow")
        if cf is not None and not cf.empty:
            try:
                repo = None
                for field in ["Common Stock Repurchase", "Repurchase Of Capital Stock",
                              "Stock Repurchased", "Repurchase Of Common Stock"]:
                    try:
                        repo = cf.loc[field].iloc[0]
                        break
                    except KeyError:
                        continue
                if repo is None:
                    # Try fuzzy match
                    for idx_name in cf.index:
                        if "repurchase" in str(idx_name).lower():
                            repo = cf.loc[idx_name].iloc[0]
                            break
                if repo and abs(float(repo)) > 0 and self.market_cap and self.market_cap > 0:
                    return abs(float(repo)) / self.market_cap
            except Exception:
                pass

        return 0.0

    # ── growth rates ─────────────────────────────────────────────────

    def revenue_growth_3yr(self) -> float | None:
        """Average annual revenue growth over the past 3 years."""
        if self._financials.get("income") is None or self._financials["income"].empty:
            return None
        try:
            revs = self._financials["income"].loc["Total Revenue"]
            revs = revs.dropna()
            if len(revs) < 2:
                return None
            # Take the most recent 3 years of data
            vals = revs.values[:min(4, len(revs))]
            vals = [float(v) for v in vals if v > 0]
            if len(vals) < 2:
                return None
            # CAGR over available years
            n = len(vals) - 1
            cagr = (vals[0] / vals[-1]) ** (1 / n) - 1
            return cagr if not np.isnan(cagr) else None
        except Exception:
            return None

    @property
    def roe(self) -> float | None:
        """Return on Equity = Net Income / Shareholders Equity."""
        ni = self.net_income
        eq = self.total_equity
        if ni and eq and eq > 0:
            return ni / eq
        return None

    @property
    def retention_ratio(self) -> float | None:
        """Retention ratio = 1 - (Dividends / Net Income)."""
        ni = self.net_income
        div = self.dividends_paid
        if ni and ni > 0:
            if div and div > 0:
                payout = min(div / ni, 1.0)
                return 1.0 - payout
            return 1.0  # No dividends → full retention
        return 0.5  # default

    @property
    def fundamental_growth(self) -> float | None:
        """g = ROE × Retention Ratio"""
        roe = self.roe
        retention = self.retention_ratio
        if roe is not None and retention is not None:
            return roe * retention
        return None

    # ── cash flow ────────────────────────────────────────────────────

    @property
    def capex(self) -> float | None:
        cf = self._financials.get("cashflow")
        if cf is not None and not cf.empty:
            try:
                capex = cf.loc["Capital Expenditure"].iloc[0]
                return abs(float(capex))
            except (KeyError, IndexError, ValueError, TypeError):
                pass
        return None

    @property
    def fcf(self) -> float | None:
        """Free Cash Flow to Firm."""
        cf = self._financials.get("cashflow")
        if cf is not None and not cf.empty:
            try:
                fcf_val = cf.loc["Free Cash Flow"].iloc[0]
                return float(fcf_val)
            except (KeyError, IndexError, ValueError, TypeError):
                pass
        return None

    def fcff(self) -> float | None:
        """Free Cash Flow to Firm = EBIT(1-t) + D&A - Capex - ΔWC."""
        ebit = self.operating_income
        if ebit is None:
            return None
        tax_rate = self.effective_tax_rate or 0.21
        ebit_at = ebit * (1 - tax_rate)
        da = self.depreciation or 0
        cap = abs(self.capex) if self.capex else 0
        wc_change = 0  # Simplified - would need working capital data

        return ebit_at + da - cap - wc_change

    # ── summary dict ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dict of all computed values for reporting."""
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "sector": self.sector,
            "industry": self.industry,
            "country": self.country,
            "market_cap": self.market_cap,
            "enterprise_value": self.enterprise_value,
            "current_price": self.current_price,
            "shares_outstanding": self.shares_outstanding,
            "beta_raw": self.beta,
            "total_debt": self.total_debt,
            "total_equity": self.total_equity,
            "cash_and_equivalents": self.cash_and_equivalents,
            "debt_to_equity": self.debt_to_equity(),
            "revenue": self.revenue,
            "operating_income": self.operating_income,
            "net_income": self.net_income,
            "operating_margin": self.operating_margin,
            "net_margin": self.net_margin,
            "effective_tax_rate": self.effective_tax_rate,
            "dividend_yield": self.dividend_yield,
            "buyback_yield": self.buyback_yield,
            "revenue_growth_3yr": self.revenue_growth_3yr(),
            "roe": self.roe,
            "retention_ratio": self.retention_ratio,
            "fundamental_growth": self.fundamental_growth,
            "depreciation": self.depreciation,
            "capex": self.capex,
            "free_cash_flow": self.fcf,
            "fcff_estimate": self.fcff(),
            "interest_expense": self.interest_expense,
        }
