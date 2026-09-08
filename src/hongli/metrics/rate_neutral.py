"""Factor 2/5: rate-neutral valuation s_val — spec §4.3 (two-layer mechanism).

Layer 1 (rank): spread_hist = adj_yield_hist - y10_hist sampled monthly.
  rank_spread = (rank of latest spread within own 5y history) in [0,1].
Layer 2 (z): Z-score de-trending of spread history; if |z10| (current y10 vs
  its own 10y history) at extreme, de-trended layer dominates.

Blend: theta = clamp((|z10| - theta_box) / blend_span_span, 0, 1)
  actually: |z10|<=theta_box -> pure rank; >=theta_box*2 -> pure z; linear between.
Final s_val = theta_blend(rank_layer, z_layer), then EMA smoothing with previous
  s_val via blend_span weight.
"""
from __future__ import annotations

import numpy as np

from ..types import FLAG_Y10, SValResult


def _monthly_last(series: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Take last observation per YYYY-MM."""
    out = {}
    for d, v in series:
        if v is None or not np.isfinite(v):
            continue
        out[d[:7]] = (d, v)  # later dates overwrite
    return [out[k] for k in sorted(out)]


def _align(spread_dates: dict, y10_series: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Monthly spread paired with y10 at same month-end."""
    y10_m = {d[:7]: v for d, v in _monthly_last(y10_series)}
    out = []
    for d, spread in _monthly_last(spread_dates.items().__iter__().__length_hint__() and list(spread_dates.items()) or []):
        pass
    return out


def rank_spread_v2(adj_yield_hist: list[tuple[str, float]], y10_hist: list[tuple[str, float]],
                   price_hist: list[tuple[str, float]], dps_ttm: float | None,
                   prev_s_val: float | None, cfg: dict,
                   cur_yield: float | None = None) -> SValResult:
    """adj_yield_hist: [(date, dps_ttm/price)] monthly closing yields, ~5y.
    y10_hist: [(date, y10 yield %)] full history for z-layer.
    price_hist: [(date, close)] to rebuild historical yields.
    cur_yield: optional override for the current yield (used by D-anchor price
    solve — evaluates spread percentile at a hypothetical price).
    """
    r = SValResult()
    val = cfg["valuation"]
    r.theta = float(val["theta_box"])

    if dps_ttm is None or dps_ttm <= 0:
        r.flags.append("MISSING_DIV")
        return r

    # rebuild historical adjusted yield series (monthly last)
    px = _monthly_last(price_hist)
    if len(px) < val.get("min_spread_points", 36):
        r.flags.append("MISSING_KLINE")
    yields_m = [(d, dps_ttm / p) for d, p in px if p and p > 0]

    y10_m = _monthly_last(y10_hist)
    if not y10_m:
        r.flags.append(FLAG_Y10)
        return r

    # pair spread = yield - y10/100 (y10 in % -> ratio)
    y10_lookup = {d[:7]: v / 100.0 for d, v in y10_m}
    spreads = []
    for d, y in yields_m:
        m = d[:7]
        if m in y10_lookup:
            spreads.append((d, y - y10_lookup[m]))
    min_pts = val.get("min_spread_points", 36)
    if len(spreads) < min_pts:
        # insufficient history: rank layer unavailable -> flag
        r.flags.append("INSUFFICIENT_SPREAD_HISTORY")
        if len(spreads) < 12:
            return r

    sv = np.array([s for _, s in spreads], dtype=float)
    if cur_yield is not None:
        # hypothetical current yield (D-anchor solve): override the last spread
        m = sv  # keep history; current spread replaced
        cur_override = cur_yield - y10_lookup.get(spreads[-1][0][:7], y10_m[-1][1] / 100.0)
        cur = cur_override
    else:
        cur = sv[-1]

    # Layer 1: percentile rank of current spread in own history (higher=cheaper)
    rank_layer = float((sv < cur).sum() + 0.5 * (sv == cur).sum()) / len(sv)

    # Layer 2: z-score de-trending
    mu, sd = float(np.mean(sv)), float(np.std(sv))
    z_layer = 1.0 / (1.0 + np.exp(-(cur - mu) / sd)) if sd > 1e-9 else 0.5

    # current rate position: z of y10 vs its own 10y history
    y10v = np.array([v for _, v in y10_m], dtype=float)
    if len(y10v) >= 60:
        zm, zs = float(np.mean(y10v)), float(np.std(y10v))
        z10 = abs((y10v[-1] - zm) / zs) if zs > 1e-9 else 0.0
    else:
        z10 = 0.0
    theta_box = float(val["theta_box"])
    # blend weight: 0 -> pure rank, 1 -> pure z (rate at extreme)
    w_z = min(max((z10 - theta_box) / max(theta_box, 1e-9) / 2.0, 0.0), 1.0) \
        if z10 > theta_box else 0.0
    s = (1 - w_z) * rank_layer + w_z * z_layer

    # smoothing with previous s_val
    span = float(val["blend_span"])
    if prev_s_val is not None and np.isfinite(prev_s_val):
        s = span * prev_s_val + (1 - span) * s

    r.s_val = float(min(max(s, 0.0), 1.0))
    r.rank_layer = rank_layer
    r.z_layer = float(z_layer)
    r.theta = z10
    return r
