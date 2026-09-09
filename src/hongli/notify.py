"""Feishu webhook notifier — pushes weekly report digest after pipeline run.

Uses a custom bot webhook (open.feishu.cn/open-apis/bot/v2/hook/...).
Message: interactive-free simple text card via "text" plus an "post" rich text
block. Sends the local HTML file path; for a public URL, serve report/ dir and
fill cfg notify.public_base_url.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import requests


def _pick_watchlist(rows: list[dict], cfg: dict, top_n: int = 3) -> list[dict]:
    """值得关注 = 推荐程度排序（scoring/ranking.py，与 HTML 候选表同序）的 top_n.

    rows 已由 rows_for_report 按推荐程度排好：矩阵格 low_A>mid_A>low_B、
    现金流警示降级、低波优先、综合分决胜。这里只做兜底过滤。"""
    cells = set(cfg["matrix"]["llm_candidate_cells"])
    cands = [r for r in rows if r.get("matrix_cell") in cells]
    return cands[:top_n]


def _render_rows_md(items: list[dict], vol_threshold: float = 0.25) -> str:
    lines = []
    for r in items:
        zone = r.get("zone") or (None, None)
        buy = f"{zone[0]:.2f}" if zone[0] else "-"
        sell = f"{zone[1]:.2f}" if zone[1] else "-"
        sy = f"{r['sus_yield']*100:.2f}%" if r.get("sus_yield") else "-"
        vol = r.get("ann_vol")
        vol_s = f"{vol*100:.0f}%" if vol else "-"
        # 低波标记: 七维检验标准 年化波动<=25% 达标; >5.5%息率+低波=加速线
        vol_tag = ""
        if vol is not None:
            vol_tag = "｜低波✅" if vol <= vol_threshold else "｜波动⚠️"
        accel = ""
        if r.get("sus_yield") and vol is not None and vol <= vol_threshold \
                and r["sus_yield"] > 0.055:
            accel = "｜🔥息率>5.5%+低波(加速线)"
        # 现金流标记 (七维检验: OCF/NP>=1 合格, >=1.5 优秀)
        onp = r.get("ocf_np")
        cfo_tag = ""
        if onp is not None:
            if onp >= 1.5:
                cfo_tag = "｜现金流✅✅"
            elif onp >= 1.0:
                cfo_tag = "｜现金流✅"
            elif onp < 0.6:
                cfo_tag = "｜现金流❌"
            else:
                cfo_tag = "｜现金流⚠️"
        lines.append(
            f"**{r['name']}（{r['code']}）**｜{r['tier']}｜"
            f"现价 {r.get('price', '-')}｜股息率 {sy}｜"
            f"52w波动 {vol_s}{vol_tag}{accel}{cfo_tag}｜"
            f"买入参考 < {buy}｜卖出参考 > {sell}"
        )
    return "\n".join(lines)


def notify_run(rows: list[dict], html_path: Path, cfg: dict, run_id: str) -> bool:
    """Send Feishu notification. Returns True on HTTP 200 with code 0."""
    notify = cfg.get("notify") or {}
    webhook = notify.get("feishu_webhook") or ""
    if not webhook:
        print("[notify] feishu_webhook not configured, skip")
        return False
    asof = run_id.split("-")[0]
    asof_cn = f"{asof[:4]}年{int(asof[4:6])}月{int(asof[6:8])}日"
    items = _pick_watchlist(rows, cfg)
    top3 = " / ".join(f"{r['name']}（{r['code']}）" for r in items) or "无候选标的"
    title = f"【红利周报-{asof_cn}】{top3} 值得关注"

    public_url = (notify.get("public_base_url") or "").rstrip("/")
    if public_url:
        # published site serves the latest report as index.html (root link)
        link = public_url + "/"
        body = (
            f"{title}\n\n{_render_rows_md(items)}\n\n"
            f"完整报告（在线查看）：{link}\n"
            f"本地文件：{html_path}"
        )
    else:
        body = (
            f"{title}\n\n{_render_rows_md(items)}\n\n"
            f"完整报告（本地 HTML，见附件路径）：\n{html_path.resolve()}\n"
            f"（如需在线查看，配置 notify.public_base_url 后自动带链接）"
        )
    payload = {"msg_type": "text", "content": {"text": body}}
    try:
        resp = requests.post(webhook, json=payload, timeout=15)
        ok = resp.status_code == 200 and resp.json().get("code", resp.json().get("StatusCode", -1)) == 0
        print(f"[notify] feishu {'OK' if ok else 'FAILED'}: http={resp.status_code} resp={resp.text[:120]}")
        return ok
    except Exception as e:  # noqa: BLE001 — notify must never break pipeline
        print(f"[notify] feishu error (ignored): {e}")
        return False
