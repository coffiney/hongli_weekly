"""Factor 4/5: cycle positioning s_cyc — spec §4.6.

s_cyc = 1 - percentile(current EPS in own 10y history).
  EPS at cycle trough (low percentile) -> s_cyc high (contrarian attractive).
"""
from __future__ import annotations

import numpy as np

from ..types import FLAG_EPS, CycResult


def compute_cycle(eps_history: list) -> CycResult:
    """eps_history: annual EPS ascending by year (may span 10y)."""
    r = CycResult()
    vals = [e for e in eps_history if e is not None and np.isfinite(e)]
    if len(vals) < 3:
        r.flags.append(FLAG_EPS)
        return r
    cur = vals[-1]
    pct = float((np.array(vals) < cur).sum() + 0.5 * (np.array(vals) == cur).sum()) / len(vals)
    r.eps_pct = pct
    r.s_cyc = float(1.0 - pct)
    return r
