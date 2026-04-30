# Expected Returns Dashboard

Daily snapshots of expected returns for Mag 7 + SPY using 3 methods:

- **CAPM** — Cost of equity with country risk premium
- **Total Return** — Dividend yield + buyback yield + growth
- **DCF IRR** — Implied expected return from FCFF DCF

## Data

- **Aswath Damodaran (NYU Stern)** — ERP, industry betas, WACC, margins
- **Yahoo Finance** — Prices, financials, shares
- **SQLite** at `~/.cache/expected_returns/trends.db`

## Usage

```bash
pip install -r requirements.txt
python -m scripts.expected_returns.trends            # collect + chart
python -m scripts.expected_returns.trends --backfill  # + historical data
```

Deployed via GitHub Pages at https://changjulian17.github.io/expected-return/
