"""Factor 1/5: sustainable dividend yield (sus_yield) — spec §4.2.

sus_yield = mid_eps × payout_sus / price, fused with ttm_yield:
  w_ttm by sector type (cyclical 0.30 / defensive 0.60 / default 0.45).
mid_eps = median of last 5 annual EPS (robust mid-cycle).
payout_sus = clamp(median payout of last 5y, ideal_payout band soft-clamped).
"""
from __future__ import annotations

import numpy as np

from ..types import FLAG_DIV, FLAG_EPS, FLAG_KLINE, SusYieldResult


def _median(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v) and v > 0]
    return float(np.median(vals)) if vals else None


def compute_sus_yield(eps_annual: list, dps_by_year: dict, price: float | None,
                      sector_type: str, cfg: dict) -> SusYieldResult:
    """eps_annual: list of annual basic EPS (most recent last, annual reports only).
    dps_by_year: {year:int -> total dps_pretax for that year}.
    price: latest close. All data must be real; missing -> flag.
    """
    r = SusYieldResult()
    val = cfg["valuation"]
    eps_years = cfg["window"]["eps_years"]

    if price is None or not np.isfinite(price) or price <= 0:
        r.flags.append(FLAG_KLINE)
        return r

    mid_eps = _median(eps_annual[-eps_years:])
    r.mid_eps = mid_eps
    if mid_eps is None:
        r.flags.append(FLAG_EPS)

    # TTM yield: latest full-year dps (annualized) / price
    years = sorted(int(y) for y in dps_by_year)
    ttm_yield = None
    payout_sus = None
    if not years:
        r.flags.append(FLAG_DIV)
    else:
        last_y = years[-1]
        dps = dps_by_year[last_y]
        # payout: last year payout ratio needs EPS of same year; use the last
        # annual EPS aligned by year order
        eps_last = eps_annual[-1] if eps_annual else None
        if dps and dps > 0:
            ttm_yield = dps / price
            if eps_last and eps_last > 0:
                payout_raw = dps / eps_last
                lo, hi = val["ideal_payout"]
                payout_sus = min(max(payout_raw, lo), hi)
        r.ttm_yield = ttm_yield
        r.payout_sus = payout_sus

    if ttm_yield is None:
        r.flags.append(FLAG_DIV)

    sus = None
    if mid_eps and payout_sus:
        sus = mid_eps * payout_sus / price

    # fuse with ttm by sector type
    w_ttm = val["w_ttm_defensive"] if sector_type in ("defensive", "infra", "telecom", "utility") \
        else val["w_ttm_cyclical"] if sector_type in ("cyclical",) \
        else val.get("w_ttm_default", 0.45)
    parts = []
    if sus is not None:
        parts.append((1 - w_ttm, sus))
    if ttm_yield is not None:
        parts.append((w_ttm, ttm_yield))
    if parts:
        r.sus_yield = sum(w * v for w, v in parts) / sum(w for w, _ in parts)
    return r
