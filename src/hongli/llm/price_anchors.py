"""Objective pricing anchors — spec §4.8 (computed by fixed algorithm, not LLM).

Anchors per stock (all from real data; None when data insufficient):
  A target_yield:  dps_sus / target_yield_floor  (dividend-discount at floor yield)
  G gordon:        dps_ttm×(1+g) / (r - g),  g=clamped ROE×payout, r=y10+risk_premium
  B range_pos:     52w low + range_pos × (52w high - 52w low)
  C yield_floor:   dps_ttm / yield_floor
  D composite_solve: bisection on price so composite(score)=d_target_composite
Sell/buy zone: [max(A,G,C), ×1.0]; candidates only from matrix candidate cells.
"""
from __future__ import annotations

import numpy as np


def anchor_target_yield(dps_sus: float | None, floor: float) -> float | None:
    if dps_sus is None or dps_sus <= 0:
        return None
    return dps_sus / floor


def anchor_gordon(dps_ttm: float | None, roe: float | None, payout: float | None,
                  y10: float | None, cfg: dict) -> float | None:
    pa = cfg["llm"]["price_anchors"]
    if dps_ttm is None or dps_ttm <= 0 or roe is None or y10 is None:
        return None
    payout = payout if (payout is not None and 0 < payout < 1) else 0.5
    g = min(max(roe * payout, 0.0), pa["g_cap"])
    r = y10 / 100.0 + pa["risk_premium"]
    if r - g < pa["g_floor_spread"]:
        g = r - pa["g_floor_spread"]
    return dps_ttm * (1 + g) / (r - g)


def anchor_range(price_hist_close: list, pos: float) -> float | None:
    c = [p for p in price_hist_close if p and np.isfinite(p) and p > 0]
    if len(c) < 60:
        return None
    lo, hi = float(np.min(c[-250:])), float(np.max(c[-250:]))
    if hi <= lo:
        return None
    return lo + pos * (hi - lo)


def anchor_yield_floor(dps_ttm: float | None, floor: float) -> float | None:
    if dps_ttm is None or dps_ttm <= 0:
        return None
    return dps_ttm / floor


def anchor_composite_solve(price_now: float | None, composite_fn, cfg: dict) -> float | None:
    """Bisection: find price where composite(price)=target. composite_fn(price)->float.

    Higher price -> lower yield -> lower composite (monotone decreasing).
    Range widened to [0.05x, 8x] so deep-discount/deep-premium names still bracket."""
    target = cfg["llm"]["price_anchors"]["d_target_composite"]
    if price_now is None or price_now <= 0:
        return None
    try:
        lo_p, hi_p = price_now * 0.05, price_now * 8.0
        f_lo = composite_fn(lo_p)   # cheap price -> high composite
        f_hi = composite_fn(hi_p)   # expensive -> low composite
        if f_lo is None or f_hi is None or f_lo < target or f_hi > target:
            # target not bracketed -> no solution in range
            if f_lo is not None and f_lo < target:
                return lo_p  # even at -95% composite below target: extremely overvalued
            return None
        for _ in range(48):
            mid = (lo_p + hi_p) / 2
            f = composite_fn(mid)
            if f is None:
                return None
            if f >= target:
                lo_p = mid
            else:
                hi_p = mid
        return (lo_p + hi_p) / 2
    except Exception:  # noqa: BLE001
        return None


def compute_anchors(stock_result, dps_ttm, dps_sus, roe, payout, y10, price_hist_close,
                    composite_fn, cfg) -> dict:
    """Returns {anchor_name: price} with only computable anchors."""
    pa = cfg["llm"]["price_anchors"]
    out = {}
    a = anchor_target_yield(dps_sus, pa["target_yield_floor"])
    if a:
        out["A_target_yield"] = a
    g = anchor_gordon(dps_ttm, roe, payout, y10, cfg)
    if g:
        out["G_gordon"] = g
    b = anchor_range(price_hist_close, pa["range_pos"])
    if b:
        out["B_range_pos"] = b
    c = anchor_yield_floor(dps_ttm, pa["yield_floor"])
    if c:
        out["C_yield_floor"] = c
    d = anchor_composite_solve(stock_result.price, composite_fn, cfg)
    if d:
        out["D_composite_solve"] = d
    return out


def price_zone(anchors: dict, price_now: float | None) -> tuple | None:
    """(buy_below, sell_above) from objective anchors.
    buy = min of anchors (consensus cheap), sell = max(A,G,C) capped at 1.6×buy."""
    vals_a = [v for k, v in anchors.items()
              if k in ("A_target_yield", "G_gordon", "C_yield_floor")]
    if not vals_a:
        return None
    buy = min(vals_a + ([price_now] if price_now else []))
    sell = max(vals_a)
    if sell < buy:
        return None
    sell = min(sell, buy * 1.6)
    return (round(buy, 2), round(sell, 2))
