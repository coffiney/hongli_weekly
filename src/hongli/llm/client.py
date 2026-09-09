"""LLM client — openai SDK compat mode; graceful no-key degradation.

Per user instruction: model from agent plan (glm-4.7 default, env override),
API key left blank (env HONGLI_LLM_API_KEY). When key absent, LLM section is
skipped and the report renders objective content only.
LLM only *interprets* rule-computed scores/anchors; it never computes them.
"""
from __future__ import annotations

import json
import os


def _normalize_base_url(url: str) -> str:
    """Ensure base_url ends with /v1 for openai-compat endpoints.

    Volcengine Ark: https://.../api/plan (docs) but SDK needs /api/plan/v1.
    Endpoints already ending in /v1 (deepseek) or /v4 (zhipu) pass through.
    """
    if not url:
        return url
    url = url.rstrip("/")
    if url.endswith(("/v1", "/v4", "/compatible-mode")):
        return url
    return url + "/v1"


def llm_available(cfg: dict) -> bool:
    key = os.environ.get(cfg["llm"].get("api_key_env", "HONGLI_LLM_API_KEY"), "")
    base_url = cfg["llm"].get("base_url") or os.environ.get("HONGLI_LLM_BASE_URL", "")
    return bool(key) and bool(base_url)


def build_prompt(rows: list[dict], cfg: dict) -> str:
    cells = set(cfg["matrix"]["llm_candidate_cells"])
    # rows already ordered by recommendation strength (scoring/ranking.py)
    cands = [r for r in rows if r.get("matrix_cell") in cells]
    top = cands[: cfg["llm"]["top_n_recommend"]]
    data = []
    for r in top:
        data.append({
            "code": r["code"], "name": r["name"], "sector": r["sector"],
            "price": r.get("price"), "matrix_cell": r["matrix_cell"],
            "tier": r.get("tier"), "composite": round(r["composite"], 1) if r.get("composite") else None,
            "sus_yield_pct": round(r["sus_yield"] * 100, 2) if r.get("sus_yield") else None,
            "s_val": round(r["s_val"], 3) if r.get("s_val") else None,
            "ann_vol_pct": round(r["ann_vol"] * 100) if r.get("ann_vol") else None,
            "ocf_np": round(r["ocf_np"], 2) if r.get("ocf_np") else None,
            "vol_tag": r.get("vol_tag") or "",
            "buy_below": r.get("zone", (None, None))[0],
            "sell_above": r.get("zone", (None, None))[1],
            "flags": r.get("flags", []),
        })
    return (
        "你是A股红利策略分析师。以下是由固定算法计算的估值×质量矩阵候选标的"
        "（仅列出，评分与价格锚均为算法结果，禁止重新计算或修改数字）。\n"
        "请用中文输出：1) 一段≤200字的市场总结；2) 对每个标的≤80字的点评，"
        "引用其买入价/卖出价并给出操作建议（买入/持有/观望）。\n"
        f"数据（JSON）：\n{json.dumps(data, ensure_ascii=False, indent=1)}"
    )


def run_llm(rows: list[dict], cfg: dict) -> dict | None:
    """Returns {'summary': str, 'picks': [...] } or None when unavailable."""
    if not llm_available(cfg):
        return None
    try:
        from openai import OpenAI
        base_url = _normalize_base_url(
            cfg["llm"].get("base_url") or os.environ.get("HONGLI_LLM_BASE_URL", ""))
        client = OpenAI(api_key=os.environ[cfg["llm"]["api_key_env"]], base_url=base_url)
        resp = client.chat.completions.create(
            model=cfg["llm"]["model"],
            temperature=cfg["llm"]["temperature"],
            messages=[{"role": "user", "content": build_prompt(rows, cfg)}],
        )
        text = resp.choices[0].message.content
        return {"summary": text}
    except Exception as e:  # noqa: BLE001 — LLM failure must never break the report
        print(f"[llm] failed (degraded to no-LLM): {e}")
        return None


def output_guard(llm_out: dict | None, rows: list[dict]) -> dict | None:
    """Validate LLM output: reject if it fabricates codes/numbers not in anchors."""
    if llm_out is None:
        return None
    known = {r["code"] for r in rows}
    text = llm_out.get("summary", "")
    # cheap guard: any 6-digit number that looks like a code must be known
    import re
    for m in re.findall(r"\b\d{6}\b", text):
        if m not in known:
            print(f"[llm-guard] unknown code {m} in LLM output -> dropping LLM block")
            return None
    return llm_out
