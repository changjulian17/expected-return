"""
Damodaran Expected Returns — Three-methodology expected return estimation.

Uses Aswath Damodaran's published data and valuation frameworks to compute:
1. Cost of Equity (CAPM + CRP)
2. Total Return Decomposition (yields + growth)
3. FCFF DCF → Implied IRR

All data cached locally under ~/.cache/expected_returns/
"""

__version__ = "0.1.0"

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
