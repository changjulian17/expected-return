"""
HTML dashboard report generator.

Produces a dark-themed HTML report with side-by-side comparison tables,
individual company breakdowns, and color-coded expected returns.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── colors ───────────────────────────────────────────────────────────
BG_COLOR = "#0d1117"
CARD_BG = "#161b22"
TEXT_COLOR = "#c9d1d9"
HEADER_COLOR = "#f0f6fc"
GREEN = "#3fb950"
RED = "#f85149"
ORANGE = "#d29922"
BLUE = "#58a6ff"
MUTED = "#8b949e"
BORDER = "#30363d"

OUTPUT_DIR = Path(__file__).parent / "output"


def _pct(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.2f}%"


def _color_class(er: float | None) -> str:
    if er is None:
        return ""
    if er > 0.10:
        return "attractive"
    if er > 0.05:
        return "moderate"
    return "overvalued"


def generate_html(
    results: list[dict[str, Any]],
    portfolio_result: dict[str, Any] | None = None,
    title: str = "Damodaran Expected Returns",
) -> str:
    """Generate a complete HTML report.

    Parameters
    ----------
    results : list[dict]
        Each dict has keys: ticker, company_name, sector, industry, country,
        market_cap, capm, total_return, dcf.
    portfolio_result : dict or None
        Aggregated portfolio data from portfolio.aggregate().
    title : str
        Page title.

    Returns
    -------
    str : Complete HTML document.
    """
    rows_html = ""
    breakdowns_html = ""
    is_portfolio = len(results) > 1

    for i, r in enumerate(results):
        ticker = r.get("ticker", "?")

        capm = r.get("capm", {})
        tr = r.get("total_return", {})
        dcf = r.get("dcf", {})

        capm_er = capm.get("expected_return")
        tr_er = tr.get("expected_return")
        dcf_er = dcf.get("expected_return")

        # Consensus (average of available)
        avail = [v for v in [capm_er, tr_er, dcf_er] if v is not None]
        consensus = sum(avail) / len(avail) if avail else None

        rows_html += f"""
        <tr>
            <td>{ticker}</td>
            <td>{r.get('company_name', 'N/A')}</td>
            <td>{r.get('sector', 'N/A')}</td>
            <td class="num {_color_class(capm_er)}">{_pct(capm_er)}</td>
            <td class="num {_color_class(tr_er)}">{_pct(tr_er)}</td>
            <td class="num {_color_class(dcf_er)}">{_pct(dcf_er)}</td>
            <td class="num consensus {_color_class(consensus)}">{_pct(consensus)}</td>
            <td class="num">{r.get('country', 'US')}</td>
        </tr>"""

        # Individual breakdown card
        breakdowns_html += _individual_breakdown(r, i)

    # ── Portfolio summary ───────────────────────────────────────────
    portfolio_html = ""
    if portfolio_result and is_portfolio:
        portfolio_html = _portfolio_summary(portfolio_result)

    # ── Assumptions table (first company's assumptions as reference) ─
    assumptions_html = _assumptions_table(results)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: {BG_COLOR};
    color: {TEXT_COLOR};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.6;
    padding: 20px;
  }}
  .container {{ max-width: 1400px; margin: 0 auto; }}
  h1 {{ color: {HEADER_COLOR}; font-size: 1.8em; margin-bottom: 8px; }}
  h2 {{ color: {HEADER_COLOR}; font-size: 1.3em; margin: 24px 0 12px; }}
  h3 {{ color: {MUTED}; font-size: 1.05em; margin: 16px 0 8px; }}
  .subtitle {{ color: {MUTED}; font-size: 0.9em; margin-bottom: 20px; }}
  .card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9em;
  }}
  th {{
    text-align: left;
    padding: 10px 12px;
    border-bottom: 2px solid {BORDER};
    color: {MUTED};
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.78em;
    letter-spacing: 0.05em;
  }}
  td {{
    padding: 10px 12px;
    border-bottom: 1px solid {BORDER};
  }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr:hover td {{ background: rgba(255,255,255,0.03); }}
  .attractive {{ color: {GREEN}; font-weight: 600; }}
  .moderate {{ color: {ORANGE}; font-weight: 600; }}
  .overvalued {{ color: {RED}; font-weight: 600; }}
  .consensus {{ font-size: 1.05em; }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: 600;
  }}
  .badge-US {{ background: rgba(88,166,255,0.15); color: {BLUE}; }}
  .badge-non-US {{ background: rgba(210,153,34,0.15); color: {ORANGE}; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }}
  .method-box {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 16px;
  }}
  .method-box h4 {{ color: {HEADER_COLOR}; margin-bottom: 8px; }}
  .method-box .value {{ font-size: 1.6em; font-weight: 700; margin: 4px 0; }}
  .method-box .components {{ font-size: 0.85em; color: {MUTED}; }}
  .method-box .components li {{ list-style: none; padding: 2px 0; }}
  .assumptions {{ font-size: 0.85em; }}
  .assumptions dt {{ color: {MUTED}; float: left; clear: left; width: 200px; padding: 4px 0; }}
  .assumptions dd {{ padding: 4px 0; margin-left: 210px; }}
  .warning {{ color: {ORANGE}; font-size: 0.85em; padding: 8px; background: rgba(210,153,34,0.1); border-radius: 4px; margin: 8px 0; }}
  .error {{ color: {RED}; }}
  .footer {{ color: {MUTED}; font-size: 0.8em; text-align: center; margin-top: 40px; padding: 20px; border-top: 1px solid {BORDER}; }}
  .sector-table {{ margin-top: 8px; }}
  .download-links a {{ color: {BLUE}; text-decoration: none; margin-right: 12px; }}
  .download-links a:hover {{ text-decoration: underline; }}
  @media (max-width: 768px) {{
    .grid-3 {{ grid-template-columns: 1fr; }}
    table {{ font-size: 0.8em; }}
    th, td {{ padding: 6px 8px; }}
  }}
</style>
</head>
<body>
<div class="container">

<h1>📊 {title}</h1>
<p class="subtitle">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Aswath Damodaran Valuation Framework</p>

{portfolio_html}

<h2>📋 Side-by-Side Comparison</h2>
<div class="card">
<table>
<thead>
<tr>
    <th>Ticker</th>
    <th>Name</th>
    <th>Sector</th>
    <th>CAPM + CRP</th>
    <th>Total Return</th>
    <th>DCF → IRR</th>
    <th>Consensus</th>
    <th>Country</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>

<h2>🔍 Individual Company Breakdowns</h2>
{breakdowns_html}

{assumptions_html}

<div class="footer">
    <p>Data sourced from Aswath Damodaran (pages.stern.nyu.edu/~adamodar/) and Yahoo Finance.</p>
    <p>This is not investment advice. Expected returns are estimates based on publicly available data.</p>
</div>

</div>
</body>
</html>"""
    return html


def _individual_breakdown(r: dict, idx: int) -> str:
    ticker = r.get("ticker", "?")
    capm = r.get("capm", {})
    tr = r.get("total_return", {})
    dcf = r.get("dcf", {})

    capm_er = capm.get("expected_return")
    tr_er = tr.get("expected_return")
    dcf_er = dcf.get("expected_return")

    capm_class = _color_class(capm_er)
    tr_class = _color_class(tr_er)
    dcf_class = _color_class(dcf_er)

    # Method boxes
    capm_box = f"""
    <div class="method-box">
        <h4>⚡ CAPM + CRP</h4>
        <div class="value {capm_class}">{_pct(capm_er)}</div>
        <div class="components">
            <li>Rf: {_pct(capm.get('rf'))}</li>
            <li>β: {capm.get('levered_beta', 'N/A'):.2f}</li>
            <li>ERP: {_pct(capm.get('erp'))}</li>
            <li>CRP: {_pct(capm.get('crp'))}</li>
            <li>λ: {capm.get('lambda', 1.0):.1f}</li>
        </div>
    </div>"""

    tr_box = f"""
    <div class="method-box">
        <h4>📈 Total Return Decomposition</h4>
        <div class="value {tr_class}">{_pct(tr_er)}</div>
        <div class="components">
            <li>Div Yield: {_pct(tr.get('dividend_yield'))}</li>
            <li>Buyback Yield: {_pct(tr.get('buyback_yield'))}</li>
            <li>Growth: {_pct(tr.get('expected_growth'))}</li>
            <li>Source: {tr.get('growth_source', 'N/A')}</li>
        </div>
    </div>"""

    dcf_box = f"""
    <div class="method-box">
        <h4>🔮 DCF → Implied IRR</h4>
        <div class="value {dcf_class}">{_pct(dcf_er)}</div>
        <div class="components">
            <li>WACC: {_pct(dcf.get('wacc'))}</li>
            <li>Terminal g: {_pct(dcf.get('terminal_growth'))}</li>
            <li>DCF EV: {dcf.get('dcf_enterprise_value_wacc', 'N/A') if isinstance(dcf.get('dcf_enterprise_value_wacc'), str) else '${:,.0f}'.format(dcf.get('dcf_enterprise_value_wacc', 0))}</li>
            <li>Current EV: {'${:,.0f}'.format(dcf.get('current_enterprise_value', 0)) if dcf.get('current_enterprise_value') else 'N/A'}</li>
        </div>
    </div>"""

    # Assumptions list for this company
    assumptions_html = ""
    for method_name, method_data in [("CAPM", capm), ("Total Return", tr), ("DCF", dcf)]:
        ass = method_data.get("assumptions", {})
        if ass:
            assumptions_html += f"<h4>{method_name} Assumptions</h4><dl class='assumptions'>"
            for k, v in ass.items():
                assumptions_html += f"<dt>{k}</dt><dd>{v}</dd>"
            assumptions_html += "</dl>"

    return f"""
<div class="card" id="{ticker}">
    <h3>{ticker} — {r.get('company_name', 'N/A')}
        <span class="badge {'badge-US' if r.get('country','').upper()=='US' else 'badge-non-US'}">{r.get('country','N/A')}</span>
    </h3>
    <p style="color:{MUTED};font-size:0.85em">{r.get('sector','N/A')} · {r.get('industry','N/A')}
       · MCap: {'${:,.0f}'.format(r.get('market_cap',0)) if r.get('market_cap') else 'N/A'}</p>
    <div class="grid-3">
        {capm_box}
        {tr_box}
        {dcf_box}
    </div>
    <details style="margin-top:12px">
        <summary style="color:{BLUE};cursor:pointer">Assumptions & Details</summary>
        <div class="assumptions">
            {assumptions_html}
        </div>
    </details>
</div>"""


def _portfolio_summary(portfolio: dict[str, Any]) -> str:
    html = "<h2>📊 Portfolio Summary</h2><div class='card'>"
    html += f"<p><strong>Companies:</strong> {portfolio.get('num_companies', 0)}</p>"

    # Method averages
    for method_name, label in [("capm", "CAPM + CRP"), ("total_return", "Total Return"), ("dcf", "DCF → IRR")]:
        data = portfolio.get(method_name, {})
        if data.get("avg") is not None:
            html += f"""<p>{label}: <strong>{data.get('avg_label','N/A')}</strong>
                        (median: {data.get('median_label','N/A')}, 
                        weighted: {data.get('weighted_avg_label','N/A')})</p>"""

    # Overall
    avg_all = portfolio.get("average_all_methods")
    if avg_all:
        html += f"""<p style="margin-top:8px">Overall Average: <strong style="color:{GREEN if avg_all > 0.08 else ORANGE}">{avg_all*100:.2f}%</strong>
                    · Median: {portfolio.get('median_all_methods', 0)*100:.2f}%</p>"""

    html += "</div>"

    # Sector breakdown
    sb = portfolio.get("sector_breakdown", {})
    if sb:
        html += "<h3>Sector Breakdown</h3><div class='card'><table class='sector-table'><thead><tr>"
        html += "<th>Sector</th><th>Count</th><th>Avg CAPM</th><th>Avg Total Return</th><th>Avg DCF</th></tr></thead><tbody>"
        for sector, data in sb.items():
            html += f"""<tr>
                <td>{sector}</td><td>{data['count']}</td>
                <td class="num {_color_class(data.get('avg_capm'))}">{data.get('avg_capm_label','N/A')}</td>
                <td class="num {_color_class(data.get('avg_total_return'))}">{data.get('avg_total_return_label','N/A')}</td>
                <td class="num {_color_class(data.get('avg_dcf'))}">{data.get('avg_dcf_label','N/A')}</td>
            </tr>"""
        html += "</tbody></table></div>"

    return html


def _assumptions_table(results: list[dict]) -> str:
    """Key shared assumptions used in the analysis."""
    # Import here to avoid circular imports at module level
    from . import damodaran_data as dd_inner
    erp = dd_inner.get_implied_erp()
    rf = dd_inner.get_treasury_10y()

    html = f"""
<h2>⚙️ Reference Assumptions</h2>
<div class="card assumptions">
<dl>
    <dt>Risk-Free Rate (10Y Treasury)</dt><dd>{rf*100:.2f}%</dd>
    <dt>Implied ERP (Damodaran)</dt><dd>{erp*100:.2f}%</dd>
    <dt>Terminal Growth Rate</dt><dd>2.50%</dd>
    <dt>Projection Period</dt><dd>5 years</dd>
    <dt>Data Sources</dt><dd>Damodaran Online · Yahoo Finance</dd>
</dl>
</div>"""
    return html


# ── JSON / CSV output ────────────────────────────────────────────────

def generate_json(results: list[dict], portfolio: dict | None = None) -> str:
    """Generate JSON output for API consumption."""
    output = {
        "meta": {
            "generated": datetime.now().isoformat(),
            "methodology": "Aswath Damodaran Expected Returns Framework",
            "version": "0.1.0",
        },
        "companies": [],
        "portfolio": portfolio,
    }

    for r in results:
        company_out = {
            "ticker": r.get("ticker"),
            "name": r.get("company_name"),
            "sector": r.get("sector"),
            "industry": r.get("industry"),
            "country": r.get("country"),
            "market_cap": r.get("market_cap"),
            "enterprise_value": r.get("enterprise_value"),
        }

        methods = {}
        for method in ["capm", "total_return", "dcf"]:
            data = r.get(method, {})
            methods[method] = {
                "expected_return": data.get("expected_return"),
                "assumptions": data.get("assumptions", {}),
            }
        company_out["methods"] = methods
        output["companies"].append(company_out)

    return json.dumps(output, indent=2, default=str)


def generate_csv(results: list[dict]) -> str:
    """Generate CSV output for Excel import."""
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "Ticker", "Name", "Sector", "Industry", "Country",
        "Market Cap", "CAPM+CRP", "Total Return", "DCF→IRR",
        "Consensus"
    ])

    for r in results:
        capm_er = r.get("capm", {}).get("expected_return")
        tr_er = r.get("total_return", {}).get("expected_return")
        dcf_er = r.get("dcf", {}).get("expected_return")
        avail = [v for v in [capm_er, tr_er, dcf_er] if v is not None]
        consensus = sum(avail) / len(avail) if avail else None

        writer.writerow([
            r.get("ticker", ""),
            r.get("company_name", ""),
            r.get("sector", ""),
            r.get("industry", ""),
            r.get("country", ""),
            r.get("market_cap", ""),
            f"{capm_er*100:.2f}%" if capm_er else "N/A",
            f"{tr_er*100:.2f}%" if tr_er else "N/A",
            f"{dcf_er*100:.2f}%" if dcf_er else "N/A",
            f"{consensus*100:.2f}%" if consensus else "N/A",
        ])

    return buf.getvalue()


# ── output writer ────────────────────────────────────────────────────

def write_output(
    results: list[dict[str, Any]],
    portfolio_result: dict[str, Any] | None = None,
    filename_prefix: str = "report",
) -> dict[str, Path]:
    """Write HTML, JSON, and CSV output files.

    Returns
    -------
    dict with keys 'html', 'json', 'csv' mapping to file paths.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # HTML
    html_content = generate_html(results, portfolio_result)
    html_path = OUTPUT_DIR / f"{filename_prefix}.html"
    html_path.write_text(html_content, encoding="utf-8")
    log.info("HTML report written to %s", html_path)

    # JSON
    json_content = generate_json(results, portfolio_result)
    json_path = OUTPUT_DIR / f"{filename_prefix}.json"
    json_path.write_text(json_content, encoding="utf-8")
    log.info("JSON output written to %s", json_path)

    # CSV
    csv_content = generate_csv(results)
    csv_path = OUTPUT_DIR / f"{filename_prefix}.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    log.info("CSV output written to %s", csv_path)

    return {"html": html_path, "json": json_path, "csv": csv_path}
