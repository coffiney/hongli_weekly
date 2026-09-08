"""Cash-flow quality metric — 七维检验维度补全 (小红书验证帖映射).

标准（七维检验）: 经营现金流/净利润 >= 1 合格, >= 1.5 优秀.
A股规则下净利润含大量应计项，OCF/NP 长期 <1 说明利润未兑现为真金白银，
红利可持续性存疑（价值陷阱信号之一）。

方法：取最近 3 个完整年报的 OCF/NETPROFIT 比率的**中位数**（单年噪声大，
中位数抗异常），OCF<=0 或 NP<=0 属于明确警示但不直接判 0，由比率自然表达。
"""
from __future__ import annotations

import numpy as np


def compute_ocf_np(ocf_by_year: dict, np_by_year: dict, lookback: int = 3):
    """ocf_by_year/np_by_year: {year:int -> amount(float)} annual reports.
    Returns median ratio over up to `lookback` latest years, or None if
    fewer than 2 usable paired years (宁缺毋造: 1年样本不可信)."""
    ratios = []
    for y in sorted(set(ocf_by_year) & set(np_by_year))[-lookback:]:
        ocf, np_ = ocf_by_year.get(y), np_by_year.get(y)
        if ocf is None or np_ is None:
            continue
        if not (np.isfinite(ocf) and np.isfinite(np_)) or abs(np_) < 1e-9:
            continue
        ratios.append(ocf / np_)
    if len(ratios) < 2:
        return None
    return float(np.median(ratios))


def cfo_score(ocf_np_ratio, cfg: dict):
    """Map OCF/NP ratio -> score in [0,1].
    >= cfo_full (1.5) -> 1.0; <= cfo_zero (0.4) -> 0.0; linear in between."""
    if ocf_np_ratio is None or not np.isfinite(ocf_np_ratio):
        return None
    sus = cfg["sustainability"]
    full = sus.get("cfo_full", 1.5)
    zero = sus.get("cfo_zero", 0.4)
    return float(min(max((ocf_np_ratio - zero) / (full - zero), 0.0), 1.0))


def cfo_pass(ocf_np_ratio, cfg: dict) -> bool | None:
    """七维检验口径: >= cfo_pass(1.0) 合格. None = 数据不足无法判断."""
    if ocf_np_ratio is None:
        return None
    return ocf_np_ratio >= cfg["sustainability"].get("cfo_pass", 1.0)
