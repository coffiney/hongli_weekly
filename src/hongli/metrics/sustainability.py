"""Factor 3/5: dividend sustainability s_sus — spec §4.5.

s_sus = 0.35*payout_score + 0.30*fcf_score + 0.15*cfo_score + 0.20*trend_score
- payout_score: 1.0 if payout in ideal band; linear decay outside [0,1.2]x band.
- fcf_score: FCF coverage of dividends, full at cov_full=1.5, zero at 0.5.
- cfo_score: 现金流质量 (经营现金流/净利润), >=1.5 满分 <=0.4 零分 (七维检验口径,
  akshare 东财源; 缺数据时按权重重归一化并打 MISSING_CFO 标).
- trend_score: dividend continuity — years of consecutive non-cutting payouts,
  full at cont_full_years=15.
"""
from __future__ import annotations

import numpy as np

from ..types import FLAG_DIV, FLAG_EPS, SusScoreResult


def compute_sustainability(dps_by_year: dict, eps_by_year: dict,
                           fcf_by_year: dict | None, cfg: dict,
                           ocf_np_ratio=None) -> SusScoreResult:
    """dps_by_year: {year->dps}, eps_by_year: {year->eps}, fcf_by_year optional
    {year->fcf (经营现金流-资本开支 proxy: unavailable -> skipped by weight renorm)}.
    ocf_np_ratio: median 3y OCF/net-profit (akshare cash_flow table; None ->
    cfo component skipped by weight renorm, FLAG_CFO raised upstream)."""
    r = SusScoreResult()
    sus = cfg["sustainability"]
    years = sorted(int(y) for y in dps_by_year)
    if not years:
        r.flags.append(FLAG_DIV)
        return r

    # --- payout score (latest year with both dps & eps) ---
    payout_now = None
    payout_score = None
    for y in reversed(years):
        eps = eps_by_year.get(y)
        dps = dps_by_year.get(y)
        if eps and eps > 0 and dps is not None:
            payout_now = dps / eps
            break
    if payout_now is None or not np.isfinite(payout_now):
        r.flags.append(FLAG_EPS)
    else:
        lo, hi = sus.get("payout_band", cfg["valuation"]["ideal_payout"])
        if payout_now < 0:
            payout_score = 0.0
        elif lo <= payout_now <= hi:
            payout_score = 1.0
        elif payout_now < lo:
            payout_score = max(0.0, payout_now / lo)
        else:  # above hi: linear decay to 0 at 2x hi
            payout_score = max(0.0, 1.0 - (payout_now - hi) / hi)
    r.payout_now = payout_now

    # --- FCF coverage score ---
    fcf_score = None
    fcf_cov = None
    if fcf_by_year:
        covs = []
        for y in years[-3:]:
            fcf = fcf_by_year.get(y)
            dps = dps_by_year.get(y)
            if fcf is not None and dps and dps > 0:
                covs.append(fcf / dps)  # simplified coverage proxy
        if covs:
            fcf_cov = float(np.median(covs))
            full, zero = sus["fcf_full"], sus["fcf_zero"]
            fcf_score = min(max((fcf_cov - zero) / (full - zero), 0.0), 1.0)

    # --- cash-flow quality score (七维检验: OCF/NP, akshare source) ---
    cfo_s = None
    if ocf_np_ratio is not None and np.isfinite(ocf_np_ratio):
        full, zero = sus.get("cfo_full", 1.5), sus.get("cfo_zero", 0.4)
        cfo_s = float(min(max((ocf_np_ratio - zero) / (full - zero), 0.0), 1.0))
        r.fcf_cov = ocf_np_ratio  # surface as cash-flow coverage debug value

    # --- trend / continuity score ---
    cont_years = 0
    for y in reversed(years):
        if dps_by_year.get(y) and dps_by_year[y] > 0:
            cont_years += 1
        else:
            break
    r.cont_years = cont_years
    trend_score = min(cont_years / sus["cont_full_years"], 1.0)

    # weighted merge; missing components renormalized
    parts = []
    if payout_score is not None:
        parts.append((sus["payout_weights"]["payout"], payout_score))
    if fcf_score is not None:
        parts.append((sus["payout_weights"]["fcf"], fcf_score))
    if cfo_s is not None:
        parts.append((sus["payout_weights"].get("cfo", 0.15), cfo_s))
    parts.append((sus["payout_weights"]["trend"], trend_score))
    wsum = sum(w for w, _ in parts)
    r.s_sus = float(sum(w * v for w, v in parts) / wsum)
    r.fcf_cov = fcf_cov
    return r
