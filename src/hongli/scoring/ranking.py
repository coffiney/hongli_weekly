"""Recommendation ranking — 推荐程度排序（供 HTML 候选表与飞书消息共用）.

设计原则（用户 2026-09-09 确认的口径）：
候选格（low_A / mid_A / low_B）内部不是按综合分排，而是按"最推荐关注"排：
1. 矩阵格优先级：low_A（低估·质优）> mid_A（合理·质优）> low_B（低估·质中）
   —— 质量优先于便宜：low_B 紧邻陷阱区（low_C），便宜但质地一般的票不该排头部。
2. 现金流警示降级：OCF/NP < 0.6（利润含金量差，价值陷阱信号）的票排到最后。
3. 低波优先：52 周年化波动 ≤ 25% 的票优先（红利策略受众偏好稳）。
4. 综合分降序做同位决胜（估值为辅）。
"""
from __future__ import annotations

# matrix cell -> recommendation priority (lower = more recommended)
_CELL_PRIORITY = {"low_A": 0, "mid_A": 1, "low_B": 2}


def recommend_key(row: dict) -> tuple:
    """Sort key for 'most recommended first' ordering of candidate rows.

    row: dict from rows_for_report() — needs matrix_cell, ocf_np, ann_vol,
    composite. All missing fields are tolerated (treated as neutral/worst).
    """
    cell = row.get("matrix_cell")
    cell_p = _CELL_PRIORITY.get(cell, 9)          # non-candidates last
    ocf = row.get("ocf_np")
    # cash-flow alarm: OCF/NP < 0.6 -> demoted (2), healthy -> 0;
    # unknown (None) sits between (1) — missing data neither rewarded nor top
    cfo_p = 0 if (ocf is not None and ocf >= 0.6) else (1 if ocf is None else 2)
    vol = row.get("ann_vol")
    lowvol_p = 0 if (vol is not None and vol <= 0.25) else 1
    comp = row.get("composite")
    # descending composite -> invert sign
    return (cell_p, cfo_p, lowvol_p, -(comp if comp is not None else -1.0))


def sort_candidates(rows: list[dict]) -> list[dict]:
    """Return candidate rows ordered by recommendation strength (best first)."""
    return sorted(rows, key=recommend_key)
