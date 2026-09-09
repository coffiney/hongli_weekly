"""Pipeline orchestrator — the ONLY entry that touches AmazingData.

Flow: lock -> AmazingSession(login) -> ingest -> logout(iron rule) ->
compute metrics from DuckDB -> score -> matrix -> anchors -> LLM(optional) ->
HTML report -> postcheck. All with run_state checkpoints.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, load_config, load_universe
from .db import connect, init_schema, record_run
from .ingest import ingest_all
from .llm.client import output_guard, run_llm
from .llm.price_anchors import compute_anchors, price_zone
from .metrics.cashflow import compute_ocf_np
from .metrics.cycle import compute_cycle
from .metrics.quality import compute_quality
from .metrics.rate_neutral import rank_spread_v2
from .metrics.sustainability import compute_sustainability
from .metrics.sus_yield import compute_sus_yield
from .report.html_report import postcheck, render_report
from .scoring.composite import (finalize, sector_medians_of, universe_median_of)
from .types import StockResult
from .xy.session import AmazingSession

LOCK = PROJECT_ROOT / "data" / ".lock"
STATE = PROJECT_ROOT / "run_state.json"


def acquire_lock() -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        raise RuntimeError("another run holds the lock (data/.lock). If stale, delete manually.")
    LOCK.write_text(str(datetime.now()), encoding="utf-8")


def release_lock() -> None:
    LOCK.unlink(missing_ok=True)


def save_state(stage: str, meta: dict) -> None:
    STATE.write_text(json.dumps({"stage": stage, "at": datetime.now().isoformat(), **meta},
                                ensure_ascii=False, indent=1), encoding="utf-8")


def _y10_series(con) -> list[tuple[str, float]]:
    df = con.execute("SELECT tdate, y10 FROM treasury ORDER BY tdate").fetchdf()
    if df is None or len(df) == 0:
        return []
    df["tdate"] = pd.to_datetime(df["tdate"]).dt.strftime("%Y-%m-%d")
    return list(df.itertuples(index=False, name=None))


def _prev_s_val(con, code: str):
    # single-run system: no prior run yet -> None (smoothing starts fresh)
    return None


def compute_all(con, universe: list[dict], cfg: dict, asof: str) -> list[StockResult]:
    """Pure local computation from DuckDB (no SDK). Returns final rows."""
    y10_hist = _y10_series(con)
    results: list[StockResult] = []
    asof_dt = pd.Timestamp(asof)

    for u in universe:
        code, name, sector = u["code"], u["name"], u["sector"]
        stype = cfg["sector_type_map"].get(sector, "watch")
        r = StockResult(code=code, name=name, sector=sector)

        # price + history
        k = con.execute(
            "SELECT kline_time, close FROM kline WHERE code=? AND kline_time<=? ORDER BY kline_time",
            [code, asof_dt.date()]).fetchdf()
        if len(k) == 0:
            r.flags.append("MISSING_KLINE")
            results.append(r)
            continue
        k["kline_time"] = pd.to_datetime(k["kline_time"]).dt.strftime("%Y-%m-%d")
        price_hist = list(k.itertuples(index=False, name=None))
        r.price = float(k["close"].iloc[-1])
        price_hist_close = [p for _, p in price_hist]

        # dividends by year (ex-date calendar year).
        # TTM rule: the current calendar year is INCOMPLETE (e.g. CCB pays
        # interim in Jul + final in Jan — both land in different years), so
        # exclude the current year from dps_by_year to avoid understating yield.
        dv = con.execute(
            "SELECT ann_date, ex_date, dps_pretax FROM dividend WHERE code=? AND ex_date<=?",
            [code, asof_dt.date()]).fetchdf()
        dps_by_year: dict[int, float] = {}
        if len(dv):
            dv["year"] = pd.to_datetime(dv["ex_date"]).dt.year
            cur_year = asof_dt.year
            dv = dv[dv["year"] < cur_year]  # only complete calendar years
            if len(dv):
                g = dv.groupby("year")["dps_pretax"].sum()
                dps_by_year = {int(y): float(v) for y, v in g.items()}
        if not dps_by_year:
            r.flags.append("MISSING_DIV")

        # official weighted ROE from 业绩快报 (profit_express) — primary source.
        # ROE_WEIGHTED is in percent (e.g. 13.44) -> convert to ratio.
        roe_official: dict[int, float] = {}
        px = con.execute(
            "SELECT report_period, roe_weighted, parent_equity FROM profit_express "
            "WHERE code=? ORDER BY report_period",
            [code]).fetchdf()
        if len(px):
            for _, row in px.iterrows():
                rp = str(row["report_period"])
                if not rp.endswith("1231"):  # annual reports only for factor input
                    continue
                rw = row["roe_weighted"]
                if rw is not None and np.isfinite(rw):
                    roe_official[int(rp[:4])] = float(rw) / 100.0

        # annual financials (consolidated annual reports only; ROE derived)
        inc = con.execute(
            "SELECT report_period, eps, parent_net_profit, parent_equity FROM income "
            "WHERE code=? AND statement_type='1' ORDER BY report_period",
            [code]).fetchdf()
        eps_by_year: dict[int, float] = {}
        roe_by_year: dict[int, float] = {}
        eps_annual: list[float] = []
        roe_annual: list[float] = []
        if len(inc):
            for _, row in inc.iterrows():
                rp = str(row["report_period"])
                if not rp.endswith("1231"):  # annual reports only
                    continue
                y = int(rp[:4])
                if row["eps"] is not None and np.isfinite(row["eps"]):
                    eps_by_year[y] = float(row["eps"])
                    eps_annual.append(float(row["eps"]))
                np_, eq = row["parent_net_profit"], row["parent_equity"]
                if np_ is not None and eq is not None and np.isfinite(np_) and np.isfinite(eq) and eq > 0:
                    roe_by_year[y] = float(np_) / float(eq)
        # merge: official weighted ROE wins; income-derived fills gaps
        roe_merged = dict(roe_by_year)
        roe_merged.update(roe_official)
        roe_annual = [roe_merged[y] for y in sorted(roe_merged)]
        if not eps_annual:
            r.flags.append("MISSING_FIN_EPS")

        # --- factor 1: sus yield ---
        f1 = compute_sus_yield(eps_annual, dps_by_year, r.price, stype, cfg)
        r.sus_yield = f1.sus_yield
        r.ttm_yield = f1.ttm_yield
        r.flags.extend(f1.flags)
        dps_ttm = dps_by_year[sorted(dps_by_year)[-1]] if dps_by_year else None

        # --- factor 2: rate-neutral s_val ---
        f2 = rank_spread_v2(price_hist, y10_hist, price_hist, dps_ttm,
                            _prev_s_val(con, code), cfg)
        r.s_val = f2.s_val
        r.flags.extend(f2.flags)

        # --- factor 3: sustainability (+ cash-flow quality from akshare table) ---
        # akshare batch API provides OCF only (no NETPROFIT); NP falls back to
        # the income table (parent_net_profit) already ingested from the SDK.
        cf = con.execute(
            "SELECT report_period, ocf, net_profit FROM cash_flow "
            "WHERE code=? AND report_period LIKE '%1231' ORDER BY report_period",
            [code]).fetchdf()
        ocf_by_year: dict[int, float] = {}
        np_by_year: dict[int, float] = {}
        if len(cf):
            for _, row in cf.iterrows():
                y = int(str(row["report_period"])[:4])
                if row["ocf"] is not None and np.isfinite(row["ocf"]):
                    ocf_by_year[y] = float(row["ocf"])
                if row["net_profit"] is not None and np.isfinite(row["net_profit"]):
                    np_by_year[y] = float(row["net_profit"])
        if len(inc):
            for _, row in inc.iterrows():
                rp = str(row["report_period"])
                if not rp.endswith("1231"):
                    continue
                y = int(rp[:4])
                if y not in np_by_year and row["parent_net_profit"] is not None \
                        and np.isfinite(row["parent_net_profit"]):
                    np_by_year[y] = float(row["parent_net_profit"])
        ocf_np_ratio = compute_ocf_np(ocf_by_year, np_by_year)
        r.ocf_np = ocf_np_ratio
        if ocf_np_ratio is None:
            r.flags.append("MISSING_CFO")
        f3 = compute_sustainability(dps_by_year, eps_by_year, None, cfg,
                                    ocf_np_ratio=ocf_np_ratio)
        r.s_sus = f3.s_sus
        r.flags.extend(f3.flags)

        # --- factor 4: cycle ---
        f4 = compute_cycle(eps_annual)
        r.s_cyc = f4.s_cyc
        r.flags.extend(f4.flags)

        # --- factor 5: quality (ROE; falls back to EPS stability when equity missing) ---
        f5 = compute_quality(roe_annual, dps_by_year, cfg, eps_history=eps_annual)
        r.s_q = f5.s_q
        r.flags.extend(f5.flags)

        # --- low-vol filter column (52w annualized vol of weekly returns) ---
        from .metrics.lowvol import compute_ann_vol
        r.ann_vol = compute_ann_vol(price_hist_close)
        if r.ann_vol is None:
            r.flags.append("MISSING_VOL")

        results.append(r)

    # sector-relative quality
    smed = sector_medians_of(results)
    umed = universe_median_of(results)
    fin = [finalize(r, smed, umed, cfg) for r in results]

    # --- pricing anchors (objective, from real data) ---
    y10_now = y10_hist[-1][1] if y10_hist else None
    for r in fin:
        if r.matrix_cell == "no_data":
            continue
        k = con.execute(
            "SELECT close FROM kline WHERE code=? AND kline_time<=? ORDER BY kline_time",
            [r.code, pd.Timestamp(asof).date()]).fetchdf()
        price_hist_close = list(k["close"]) if len(k) else []
        dv = con.execute(
            "SELECT ann_date, ex_date, dps_pretax FROM dividend WHERE code=? AND ex_date<=?",
            [r.code, pd.Timestamp(asof).date()]).fetchdf()
        dps_by_year = {}
        if len(dv):
            dv["year"] = pd.to_datetime(dv["ex_date"]).dt.year
            dv = dv[dv["year"] < pd.Timestamp(asof).year]  # complete years only (TTM rule)
            if len(dv):
                for y, v in dv.groupby("year")["dps_pretax"].sum().items():
                    dps_by_year[int(y)] = float(v)
        dps_ttm = dps_by_year[sorted(dps_by_year)[-1]] if dps_by_year else None
        inc = con.execute(
            "SELECT report_period, eps, parent_net_profit, parent_equity FROM income "
            "WHERE code=? AND statement_type='1' ORDER BY report_period",
            [r.code]).fetchdf()
        # official weighted ROE (percent -> ratio) from profit_express, latest annual
        roe_last, payout_last = None, None
        px = con.execute(
            "SELECT report_period, roe_weighted FROM profit_express "
            "WHERE code=? AND report_period LIKE '%1231' ORDER BY report_period",
            [r.code]).fetchdf()
        if len(px):
            for _, prow in px.iloc[::-1].iterrows():
                rw = prow["roe_weighted"]
                if rw is not None and np.isfinite(rw):
                    roe_last = float(rw) / 100.0
                    break
        if len(inc):
            for _, row in inc.iloc[::-1].iterrows():
                rp = str(row["report_period"])
                if not rp.endswith("1231"):
                    continue
                if roe_last is None:
                    np_, eq = row["parent_net_profit"], row["parent_equity"]
                    if np_ is not None and eq is not None \
                            and np.isfinite(np_) and np.isfinite(eq) and eq > 0:
                        roe_last = float(np_) / float(eq)
                if payout_last is None and row["eps"] and row["eps"] > 0 and int(rp[:4]) in dps_by_year:
                    payout_last = dps_by_year[int(rp[:4])] / float(row["eps"])
                if payout_last is not None:
                    break
        # dps_sus ≈ mid_eps × payout_sus (proxy from factor outputs)
        dps_sus = None
        stype = cfg["sector_type_map"].get(r.sector, "watch")
        w_ttm = cfg["valuation"].get("w_ttm_default", 0.45)
        if r.sus_yield and r.price:
            dps_sus = r.sus_yield * r.price  # fused; acceptable as algorithm anchor input

        def composite_fn(price_try: float) -> float | None:
            """Composite at hypothetical price: s_val recomputed with current-month
            yield at price_try (spread percentile shifts), other factors held."""
            from .scoring.composite import composite_score
            f1x = compute_sus_yield(eps_annual, dps_by_year, price_try, stype, cfg)
            if f1x.sus_yield is None:
                return None
            f2x = rank_spread_v2(price_hist, y10_hist, price_hist, dps_ttm,
                                 None, cfg, cur_yield=f1x.sus_yield)
            comp, _ = composite_score(f2x.s_val, r.s_sus, r.s_cyc, r.s_q, r.sector, stype, cfg)
            return comp

        anchors = compute_anchors(r, dps_ttm, dps_sus, roe_last, payout_last, y10_now,
                                  price_hist_close, composite_fn, cfg)
        r.anchors = anchors
        r.zone = price_zone(anchors, r.price)
    return fin


def rows_for_report(results: list[StockResult]) -> list[dict]:
    out = []
    for r in results:
        flags_unique = sorted(set(r.flags))
        # low-vol tag (七维检验: 年化波动<=25% 达标; 息率>5.5%+低波=加速线)
        vol = getattr(r, "ann_vol", None)
        vol_tag_parts = []
        if vol is not None:
            vol_tag_parts.append("低波✅" if vol <= 0.25 else "波动⚠️")
            if r.sus_yield and vol <= 0.25 and r.sus_yield > 0.055:
                vol_tag_parts.append("🔥加速线")
        # cash-flow tag (七维检验: OCF/NP>=1 合格, >=1.5 优秀)
        onp = getattr(r, "ocf_np", None)
        if onp is not None:
            if onp >= 1.5:
                vol_tag_parts.append("现金流✅✅")
            elif onp >= 1.0:
                vol_tag_parts.append("现金流✅")
            elif onp < 0.6:
                vol_tag_parts.append("现金流❌")
            else:
                vol_tag_parts.append("现金流⚠️")
        out.append({
            "code": r.code, "name": r.name, "sector": r.sector,
            "price": r.price, "sus_yield": r.sus_yield, "ttm_yield": r.ttm_yield,
            "s_val": r.s_val, "s_sus": r.s_sus, "s_cyc": r.s_cyc, "s_q": r.s_q,
            "q_rel": r.q_rel, "composite": r.composite, "tier": r.tier,
            "matrix_cell": r.matrix_cell, "flags": flags_unique,
            "zone": getattr(r, "zone", None),
            "ann_vol": vol,
            "vol_tag": " ".join(vol_tag_parts),
            "ocf_np": onp,
            "anchor_str": ", ".join(sorted(r.anchors.keys())) if r.anchors else "",
        })
    # order by recommendation strength (matrix cell > cashflow > lowvol > composite)
    from .scoring.ranking import sort_candidates
    out = sort_candidates(out)
    return out


def run_pipeline(ingest: bool = True, llm: bool = True, asof: str | None = None) -> Path:
    cfg = load_config()
    universe = load_universe()
    asof = asof or date.today().strftime("%Y-%m-%d")
    run_id = f"{asof.replace('-', '')}-{uuid.uuid4().hex[:6]}"
    acquire_lock()
    try:
        con = connect(str(PROJECT_ROOT / cfg["db"]["path"]))
        init_schema(con)
        stats = {}
        if ingest:
            print("[pipeline] stage 1/4: ingest (SDK online)")
            save_state("ingest", {"run_id": run_id})
            with AmazingSession(
                account_file=cfg["amazingdata"]["account_file"],
                host_primary=cfg["amazingdata"]["host_primary"],
                host_backup=cfg["amazingdata"]["host_backup"],
                port=cfg["amazingdata"]["port"],
            ) as ses:
                stats = ingest_all(con, ses, universe, cfg)
            # session exited here => logout done (iron rule)
            print(f"[pipeline] ingest stats: {json.dumps(stats, ensure_ascii=False)}")
        else:
            print("[pipeline] stage 1/4: ingest skipped (offline compute)")

        print("[pipeline] stage 2/4: compute metrics")
        save_state("compute", {"run_id": run_id})
        results = compute_all(con, universe, cfg, asof)

        print("[pipeline] stage 3/4: report + llm")
        rows = rows_for_report(results)
        llm_out = run_llm(rows, cfg) if llm else None
        llm_out = output_guard(llm_out, rows)
        llm_text = llm_out["summary"] if llm_out else None

        html_path = render_report(rows, asof, run_id, cfg, llm_text)
        problems = postcheck(html_path, rows)
        record_run(con, run_id, asof, {"stats": stats, "problems": problems,
                                       "n_rows": len(rows)})
        con.close()
        print(f"[pipeline] stage 4/4: done -> {html_path}")
        if problems:
            print(f"[pipeline] POSTCHECK PROBLEMS: {problems}")
        save_state("done", {"run_id": run_id, "html": str(html_path), "problems": problems})
        # feishu notification (never breaks the run)
        try:
            from .notify import notify_run
            notify_run(rows, html_path, cfg, run_id)
        except Exception as e:  # noqa: BLE001
            print(f"[notify] unexpected error (ignored): {e}")
        return html_path
    finally:
        release_lock()


if __name__ == "__main__":
    argv = sys.argv[1:]
    kw = {"ingest": "--no-ingest" not in argv, "llm": "--no-llm" not in argv}
    if "--asof" in argv:
        kw["asof"] = argv[argv.index("--asof") + 1]
    run_pipeline(**kw)
