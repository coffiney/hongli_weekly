"""Shared dataclasses and MISSING_DATA flag protocol (spec §3.3.4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Missing-data flags — spec iron rule: 宁缺毋造 (never fabricate)
FLAG_EPS = "MISSING_FIN_EPS"
FLAG_DIV = "MISSING_DIV"
FLAG_KLINE = "MISSING_KLINE"
FLAG_Y10 = "MISSING_Y10"
FLAG_SUS = "MISSING_SUSTAINABILITY"
FLAG_Q = "MISSING_QUALITY"

ALL_FLAGS = [FLAG_EPS, FLAG_DIV, FLAG_KLINE, FLAG_Y10, FLAG_SUS, FLAG_Q]


@dataclass
class SusYieldResult:
    """sustainable dividend yield (sus_yield) + supporting debug info."""
    sus_yield: Optional[float] = None      # ratio, e.g. 0.051
    ttm_yield: Optional[float] = None
    mid_eps: Optional[float] = None        # mid-cycle EPS
    payout_sus: Optional[float] = None     # sustainable payout ratio
    flags: list = field(default_factory=list)


@dataclass
class SValResult:
    """rate-neutral valuation score s_val in [0,1] (1 = cheapest) + debug."""
    s_val: Optional[float] = None
    rank_layer: Optional[float] = None     # rank-difference layer [0,1]
    z_layer: Optional[float] = None        # z-score de-trend layer [0,1]
    theta: float = 0.0                     # effective |z10| used for blend
    flags: list = field(default_factory=list)


@dataclass
class SusScoreResult:
    """sustainability score s_sus in [0,1] + debug."""
    s_sus: Optional[float] = None
    payout_now: Optional[float] = None
    cont_years: Optional[int] = None
    fcf_cov: Optional[float] = None
    flags: list = field(default_factory=list)


@dataclass
class CycResult:
    """cycle positioning score s_cyc in [0,1] (1 = cycle trough -> attractive)."""
    s_cyc: Optional[float] = None
    eps_pct: Optional[float] = None        # EPS percentile in own history
    flags: list = field(default_factory=list)


@dataclass
class QualResult:
    """quality score s_q in [0,1] + debug."""
    s_q: Optional[float] = None
    roe_med: Optional[float] = None
    roe_std: Optional[float] = None
    cont_years: Optional[int] = None
    flags: list = field(default_factory=list)


@dataclass
class StockResult:
    """final per-stock result row."""
    code: str = ""
    name: str = ""
    sector: str = ""
    price: Optional[float] = None
    sus_yield: Optional[float] = None
    ttm_yield: Optional[float] = None
    s_val: Optional[float] = None
    s_sus: Optional[float] = None
    s_cyc: Optional[float] = None
    s_q: Optional[float] = None
    composite: Optional[float] = None      # 0-100, higher = more attractive
    tier: str = ""
    matrix_cell: str = ""                  # low_A..high_C / no_data
    q_rel: Optional[float] = None
    flags: list = field(default_factory=list)
    anchors: dict = field(default_factory=dict)   # pricing anchors (objective)
    zone: tuple | None = None                     # (buy_below, sell_above)
    ann_vol: Optional[float] = None               # 52w annualized vol of weekly returns (ratio)
    notes: str = ""


def no_rating(flags: list) -> bool:
    """>=2 missing factors -> no rating (gray card), spec §3.3.4."""
    core = [f for f in flags if f in (FLAG_EPS, FLAG_DIV, FLAG_KLINE, FLAG_Y10)]
    return len(core) >= 2
