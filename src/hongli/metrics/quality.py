"""Factor 5/5: quality s_q — spec §4.7.

s_q = 0.45*roe_score + 0.30*stability_score + 0.25*continuity_score
- roe_score: median 5y ROE, full at roe_full=12%; negative ROE -> 0.
- stability_score: 1 - normalized std of ROE over 5y.
- continuity_score: consecutive dividend years (same trend metric as sus).
"""
from __future__ import annotations

import numpy as np

from ..types import FLAG_DIV, FLAG_EPS, QualResult


def compute_quality(roe_history: list, dps_by_year: dict, cfg: dict,
                    eps_history: list | None = None) -> QualResult:
    """roe_history: annual weighted ROE (ratio, e.g. 0.11) ascending by year.
    eps_history: optional annual EPS — when ROE data is unavailable (SDK income
    table lacks equity), ROE/stability fall back to EPS level & EPS stability.
    Per 宁缺毋造: no fabrication; fallback uses only real EPS data and is flagged.
    """
    r = QualResult()
    q = cfg["quality"]
    roe_full = cfg["sustainability"]["roe_full"]

    vals = [v for v in roe_history if v is not None and np.isfinite(v)]
    eps_vals = [v for v in (eps_history or []) if v is not None and np.isfinite(v)]
    if len(vals) < 3 and len(eps_vals) >= 3:
        # ROE unavailable -> EPS-based fallback (real data only)
        r.flags.append("ROE_FALLBACK_EPS")
        e_arr = np.array(eps_vals[-5:], dtype=float)
        e_med = float(np.median(e_arr))
        r.roe_med = None
        roe_score = min(max(e_med / 2.0, 0.0), 1.0)  # 2.0 EPS ~ strong for dividend names
        cv = float(np.std(e_arr)) / abs(e_med) if abs(e_med) > 1e-9 else 1.0
        stability_score = max(0.0, 1.0 - min(cv, 1.0))
        parts0 = [(q["weights"]["roe"], roe_score), (q["weights"]["stability"], stability_score)]
    elif len(vals) >= 3:
        arr = np.array(vals[-5:], dtype=float)
        roe_med = float(np.median(arr))
        r.roe_med = roe_med
        roe_score = min(max(roe_med / roe_full, 0.0), 1.0)
        # stability: std/|median| ratio -> invert; cap at 1.0
        if abs(roe_med) > 1e-9:
            cv = float(np.std(arr)) / abs(roe_med)
            stability_score = max(0.0, 1.0 - min(cv, 1.0))
        else:
            stability_score = 0.0
        r.roe_std = float(np.std(arr))
        parts0 = [(q["weights"]["roe"], roe_score), (q["weights"]["stability"], stability_score)]
    else:
        r.flags.append(FLAG_EPS)
        parts0 = []
    # continuity
    years = sorted(int(y) for y in dps_by_year)
    cont_years = 0
    for y in reversed(years):
        if dps_by_year.get(y) and dps_by_year[y] > 0:
            cont_years += 1
        else:
            break
    r.cont_years = cont_years
    continuity_score = min(cont_years / cfg["sustainability"]["cont_full_years"], 1.0)

    parts = parts0 + [(q["weights"]["continuity"], continuity_score)]
    if not parts:
        r.flags.append(FLAG_DIV)
        return r
    wsum = sum(w for w, _ in parts)
    r.s_q = float(sum(w * v for w, v in parts) / wsum)
    return r
