"""Composite scoring, tier rating, matrix mapping — spec §4.4, §4.8.

- composite = Σ w_factor(type) × s_factor ∈ [0,1] → ×100 (higher=more attractive)
- tier from composite via thresholds.
- matrix: v-axis = s_val; q-axis = q_rel = (0.5·s_sus + 0.5·s_q) / sector median.
  9 cells low/mid/high × A/B/C; low_C = value trap; no rating if gray.
"""
from __future__ import annotations

from ..types import FLAG_EPS, StockResult, no_rating

TIER_ORDER = ["低估", "偏低", "合理", "偏高", "高估", "极高估"]


def composite_score(s_val, s_sus, s_cyc, s_q, sector: str, sector_type: str, cfg: dict):
    w = cfg["weights"].get(sector_type, cfg["weights"]["watch"])
    parts, flags = [], []
    m = {"val": s_val, "sus": s_sus, "cyc": s_cyc, "q": s_q}
    for k, v in m.items():
        if v is None:
            flags.append(k)
        else:
            parts.append((w[k], v))
    if not parts:
        return None, flags
    # renormalize over present factors, but scale down when factors missing
    wsum = sum(w for w, _ in parts)
    total_w = sum(w.values())
    score = sum(w * v for w, v in parts) / wsum * (wsum / total_w)
    return float(score * 100.0), flags


def tier_of(composite: float | None, cfg: dict) -> str:
    if composite is None:
        return "无评级"
    t = cfg["tiers"]
    c = composite
    if c >= t["低估"]:
        return "低估"
    if c >= t["偏低"]:
        return "偏低"
    if c >= t["合理"]:
        return "合理"
    if c >= t["偏高"]:
        return "偏高"
    if c >= t["高估"]:
        return "高估"
    return "极高估"


def q_relative(s_sus, s_q, sector: str, sector_medians: dict, universe_median: float | None,
               cfg: dict) -> float | None:
    if s_sus is None or s_q is None:
        return None
    q = cfg["matrix"]["q_weight_sus"] * s_sus + cfg["matrix"]["q_weight_q"] * s_q
    med = sector_medians.get(sector)
    if med is None or med <= 1e-9:
        med = universe_median
    if med is None or med <= 1e-9:
        return None
    return float(q / med)


def matrix_cell(s_val, q_rel, cfg: dict) -> str:
    if s_val is None or q_rel is None:
        return "no_data"
    vb = cfg["matrix"]["v_band"]
    qb = cfg["matrix"]["q_band"]
    v = "low" if s_val < vb["low"] else "high" if s_val > vb["high"] else "mid"
    g = "A" if q_rel > qb["A"] else "C" if q_rel < qb["C"] else "B"
    return f"{v}_{g}"


def finalize(res: StockResult, sector_medians: dict, universe_median: float | None,
             cfg: dict) -> StockResult:
    """Compute composite/tier/matrix on a partially-filled result."""
    ctype = cfg["sector_type_map"].get(res.sector, "watch")
    comp, _flags = composite_score(res.s_val, res.s_sus, res.s_cyc, res.s_q,
                                   res.sector, ctype, cfg)
    res.composite = comp
    res.q_rel = q_relative(res.s_sus, res.s_q, res.sector, sector_medians,
                           universe_median, cfg)
    res.tier = tier_of(comp, cfg)
    if no_rating(res.flags):
        res.matrix_cell = "no_data"
        res.tier = "无评级"
        res.composite = None
    else:
        res.matrix_cell = matrix_cell(res.s_val, res.q_rel, cfg)
    return res


def sector_medians_of(rows: list[StockResult]) -> dict:
    """q_raw median per sector (uses s_sus/s_q raw, before relativization)."""
    out: dict[str, list[float]] = {}
    for r in rows:
        if r.s_sus is not None and r.s_q is not None:
            out.setdefault(r.sector, []).append(0.5 * r.s_sus + 0.5 * r.s_q)
    return {k: float(sorted(v)[len(v) // 2]) if len(v) % 2 else
            float((sorted(v)[len(v) // 2 - 1] + sorted(v)[len(v) // 2]) / 2) for k, v in out.items()}


def universe_median_of(rows: list[StockResult]) -> float | None:
    vals = [0.5 * r.s_sus + 0.5 * r.s_q for r in rows if r.s_sus is not None and r.s_q is not None]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
