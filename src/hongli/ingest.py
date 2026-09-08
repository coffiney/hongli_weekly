"""Ingest layer: fetch from AmazingData -> normalize -> DuckDB."""
from __future__ import annotations

import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import db
from .xy.fetch import (fetch_cash_flow_akshare, fetch_dividend, fetch_income,
                       fetch_kline, fetch_profit_express, fetch_treasury)


def normalize_kline(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for code, df in raw.items():
        if df is None or len(df) == 0:
            continue
        d = df.copy()
        # kline_time may be datetime or string
        tcol = "kline_time" if "kline_time" in d.columns else ("time" if "time" in d.columns else None)
        if tcol is None or "close" not in d.columns:
            continue
        d[tcol] = pd.to_datetime(d[tcol]).dt.date
        out = pd.DataFrame({
            "code": code,
            "kline_time": d[tcol],
            "open": pd.to_numeric(d.get("open"), errors="coerce"),
            "high": pd.to_numeric(d.get("high"), errors="coerce"),
            "low": pd.to_numeric(d.get("low"), errors="coerce"),
            "close": pd.to_numeric(d["close"], errors="coerce"),
            "volume": pd.to_numeric(d.get("volume"), errors="coerce"),
            "amount": pd.to_numeric(d.get("amount"), errors="coerce"),
        }).dropna(subset=["close"])
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=["code", "kline_time", "open", "high", "low", "close", "volume", "amount"])
    return pd.concat(rows, ignore_index=True).drop_duplicates(subset=["code", "kline_time"])


_INCOME_COLS = {
    "MARKET_CODE": "code", "ANN_DATE": "ann_date", "REPORTING_PERIOD": "report_period",
    "STATEMENT_TYPE": "statement_type", "BASIC_EPS": "eps",
    "NET_PRO_EXCL_MIN_INT_INC": "parent_net_profit",
    "C_PARENT_COMP": "parent_equity", "TOT_OPERA_REV": "revenue",
    "NET_PRO_AFTER_DED_NR_GL": "net_profit_deducted",
}
# note: SDK income table has NO ROE column (manual p52 was inaccurate);
# ROE is computed downstream as parent_net_profit / parent_equity.


def normalize_income(raw) -> pd.DataFrame:
    """raw may be DataFrame or dict[code]->DataFrame (real SDK returns dict)."""
    if isinstance(raw, dict):
        parts = [v for v in raw.values() if v is not None and len(v) > 0]
        raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=list(_INCOME_COLS.values()))
    d = raw.copy()
    missing = [c for c in _INCOME_COLS if c not in d.columns]
    for m in missing:  # keep schema complete; values NULL -> flags handled downstream
        d[m] = None
    d = d.rename(columns=_INCOME_COLS)
    d["code"] = d["code"].astype(str).str.split(".").str[0].str.zfill(6)
    d["ann_date"] = pd.to_datetime(d["ann_date"], errors="coerce").dt.date
    d["report_period"] = d["report_period"].astype(str)
    d["statement_type"] = d["statement_type"].astype(str)
    num = ["eps", "parent_net_profit", "parent_equity", "revenue", "net_profit_deducted"]
    for c in num:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    # keep only consolidated (STATEMENT_TYPE==1) annual reports
    d = d[d["statement_type"] == "1"]
    d["year"] = d["report_period"].str[:4]
    return d[list(_INCOME_COLS.values())].drop_duplicates(subset=["code", "report_period"])


_DIV_COLS = {
    "MARKET_CODE": "code", "ANN_DATE": "ann_date", "DATE_EX": "ex_date",
    "DVD_PER_SHARE_PRE_TAX_CASH": "dps_pretax", "REPORT_PERIOD": "report_period",
}


def normalize_dividend(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=list(_DIV_COLS.values()))
    d = raw.copy()
    for m in [c for c in _DIV_COLS if c not in d.columns]:
        d[m] = None
    d = d.rename(columns=_DIV_COLS)
    d["code"] = d["code"].astype(str).str.split(".").str[0].str.zfill(6)
    d["ann_date"] = pd.to_datetime(d["ann_date"], errors="coerce").dt.date
    d["ex_date"] = pd.to_datetime(d["ex_date"], errors="coerce").dt.date
    d["report_period"] = d["report_period"].astype(str)
    d["dps_pretax"] = pd.to_numeric(d["dps_pretax"], errors="coerce")
    # drop non-cash/zero rows; drop pre-ex duplicates keeping latest
    d = d.dropna(subset=["dps_pretax"])
    d = d[d["dps_pretax"] > 0]
    # announced-but-not-yet-ex dividend rows have no ex_date: keep only ex-dated rows
    # (undated ones are not yet realized -> excluded per 宁缺毋造; future runs pick them up)
    d = d.dropna(subset=["ex_date"])
    d = d.sort_values(["code", "ann_date", "ex_date"]).drop_duplicates(
        subset=["code", "ann_date", "ex_date", "report_period"])
    return d[list(_DIV_COLS.values())]


_PX_COLS = {
    "MARKET_CODE": "code", "ANN_DATE": "ann_date", "REPORTING_PERIOD": "report_period",
    "ROE_WEIGHTED": "roe_weighted", "TOT_SHARE_EQU_EXCL_MIN_INT": "parent_equity",
    "EPS_BASIC": "eps", "NET_ASSET_PS": "net_asset_ps",
}


def normalize_profit_express(raw) -> pd.DataFrame:
    """业绩快报: official weighted ROE + parent equity (replaces EPS fallback).
    raw may be DataFrame or dict[code]->DataFrame. Keeps ALL report periods
    (annual + interim); ROE annual selection happens in pipeline."""
    if isinstance(raw, dict):
        parts = [v for v in raw.values() if v is not None and len(v) > 0]
        raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=list(_PX_COLS.values()))
    d = raw.copy()
    for m in [c for c in _PX_COLS if c not in d.columns]:
        d[m] = None
    d = d.rename(columns=_PX_COLS)
    d["code"] = d["code"].astype(str).str.split(".").str[0].str.zfill(6)
    d["ann_date"] = pd.to_datetime(d["ann_date"], errors="coerce").dt.date
    d["report_period"] = d["report_period"].astype(str)
    for c in ["roe_weighted", "parent_equity", "eps", "net_asset_ps"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    # dedupe: one row per code+period (latest announcement wins)
    d = d.sort_values(["code", "report_period", "ann_date"]).drop_duplicates(
        subset=["code", "report_period"], keep="last")
    return d[list(_PX_COLS.values())]


def normalize_treasury(raw) -> pd.DataFrame:
    """raw: dict[term]->df with YIELD column (real SDK), or DataFrame fallback."""
    if isinstance(raw, pd.DataFrame):
        raw = {"y10": raw}
    if not isinstance(raw, dict) or not raw:
        return pd.DataFrame(columns=["tdate", "y10"])
    y10 = raw.get("y10")
    if y10 is None or len(y10) == 0:
        return pd.DataFrame(columns=["tdate", "y10"])
    d = y10.copy()
    if isinstance(d, pd.Series):
        d = d.to_frame(name="y10")
    if "YIELD" in d.columns:
        d = d.rename(columns={"YIELD": "y10"})
    else:
        d = d.rename(columns={d.columns[-1]: "y10"})
    d = d.reset_index()
    d.columns = ["tdate", "y10"] + list(d.columns[2:])
    d["tdate"] = pd.to_datetime(d["tdate"], errors="coerce").dt.date
    d["y10"] = pd.to_numeric(d["y10"], errors="coerce")
    return d[["tdate", "y10"]].dropna().drop_duplicates(subset=["tdate"])


_CFO_COLS = {"code": "code", "report_period": "report_period",
             "ocf": "ocf", "net_profit": "net_profit", "capex": "capex"}


def normalize_cash_flow(raw) -> pd.DataFrame:
    """akshare cash-flow rows -> DuckDB schema. Keeps annual report periods
    (YYYY1231). raw may be DataFrame or dict[code]->DataFrame."""
    if isinstance(raw, dict):
        parts = [v for v in raw.values() if v is not None and len(v) > 0]
        raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    cols = list(_CFO_COLS.values())
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=cols)
    d = raw.copy()
    for m in [c for c in cols if c not in d.columns]:
        d[m] = None
    d = d.rename(columns=_CFO_COLS)
    d["code"] = d["code"].astype(str).str.split(".").str[0].str.zfill(6)
    d["report_period"] = d["report_period"].astype(str)
    d = d[d["report_period"].str.endswith("1231")]  # annual reports only
    for c in ["ocf", "net_profit", "capex"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    # OCF is the factor's core input; rows without it are dead weight
    d = d.dropna(subset=["ocf"])
    return d[cols].drop_duplicates(subset=["code", "report_period"]).sort_values(
        ["code", "report_period"])


def ingest_all(con, ses, universe: list[dict], cfg: dict) -> dict:
    """Full backfill/refresh. Returns stats dict. Reads once, writes locally."""
    codes = [u["code"] for u in universe]
    w = cfg["window"]
    stats = {}

    # --- kline ---
    kline_raw = fetch_kline(ses, codes, years=w["kline_years"])
    kdf = normalize_kline(kline_raw)
    stats["kline_rows"] = db.upsert_dataframe(con, "kline", kdf, ["code", "kline_time"])
    stats["kline_codes"] = kdf["code"].nunique() if len(kdf) else 0

    # --- income ---
    inc_raw = fetch_income(ses, codes, from_year=w["fin_from"])
    idf = normalize_income(inc_raw)
    stats["income_rows"] = db.upsert_dataframe(con, "income", idf,
                                               ["code", "report_period", "statement_type"])
    stats["income_codes"] = idf["code"].nunique() if len(idf) else 0

    # --- dividend ---
    div_raw = fetch_dividend(ses, codes, from_year=w["div_from"])
    ddf = normalize_dividend(div_raw)
    stats["div_rows"] = db.upsert_dataframe(con, "dividend", ddf, ["code", "ann_date", "ex_date", "report_period"])
    stats["div_codes"] = ddf["code"].nunique() if len(ddf) else 0

    # --- profit express (official weighted ROE / parent equity) ---
    px_raw = fetch_profit_express(ses, codes, from_year=w["fin_from"])
    pxdf = normalize_profit_express(px_raw)
    stats["px_rows"] = db.upsert_dataframe(con, "profit_express", pxdf,
                                           ["code", "report_period"])
    stats["px_codes"] = pxdf["code"].nunique() if len(pxdf) else 0

    # --- cash flow (akshare; AmazingData cash-flow endpoint fails server-side) ---
    try:
        cfo_raw = fetch_cash_flow_akshare(codes, from_year=w["fin_from"])
    except Exception as e:  # noqa: BLE001 — akshare outage must not kill the run
        print(f"[fetch] cash_flow akshare unavailable: {e}")
        cfo_raw = pd.DataFrame()
    cfdf = normalize_cash_flow(cfo_raw)
    stats["cf_rows"] = db.upsert_dataframe(con, "cash_flow", cfdf, ["code", "report_period"])
    stats["cf_codes"] = cfdf["code"].nunique() if len(cfdf) else 0

    # --- treasury ---
    tre_raw = fetch_treasury(ses, years=w["y10_hist_years"])
    tdf = normalize_treasury(tre_raw)
    stats["treasury_rows"] = db.upsert_dataframe(con, "treasury", tdf, ["tdate"])

    # missing codes
    have = set(kdf["code"].unique()) if len(kdf) else set()
    stats["missing_kline"] = sorted(set(codes) - have)
    return stats
