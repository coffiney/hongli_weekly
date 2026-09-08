"""Low-volatility filter column — inspired by 七维检验 (低波: 年化波动≤25%).

Computes annualized volatility from the LAST ~52 weeks of DAILY closes:
  take the trailing 52-week window of daily closes -> weekly samples
  (every 5th close) -> std of weekly log returns -> × sqrt(52).
Weekly sampling (not daily σ×sqrt(252)) dampens autocorrelation and matches
the convention the filter was designed for.
"""
from __future__ import annotations

import math

WEEKS = 52


def compute_ann_vol(price_hist_close: list[float], weeks: int = WEEKS) -> float | None:
    """Annualized volatility from weekly-sampled closes. Returns ratio (0.25=25%)."""
    if not price_hist_close:
        return None
    # trailing 52-week window of daily closes
    window = price_hist_close[-(weeks * 5 + 1):]
    if len(window) < weeks * 5 * 0.6:  # insufficient history (suspensions / new listing)
        return None
    sampled = window[::5]  # every 5th trading day ≈ weekly; ~53 samples
    rets = []
    for a, b in zip(sampled, sampled[1:]):
        if a > 0 and b > 0:
            rets.append(math.log(b / a))
    if len(rets) < weeks * 0.6:  # need >= ~31 weekly returns
        return None
    mu = sum(rets) / len(rets)
    var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
    weekly_sd = math.sqrt(var)
    return weekly_sd * math.sqrt(52)
