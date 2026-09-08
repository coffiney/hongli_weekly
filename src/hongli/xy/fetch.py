"""Fetchers over AmazingData. All return plain pandas DataFrames/dicts.

Field names verified against AmazingData开发手册:
- query_kline: code, kline_time, open, high, low, close, volume, amount
- get_income: MARKET_CODE, ANN_DATE, BASIC_EPS, DILUTED_EPS, TOT_OPERA_REV,
              NET_PRO_AFTER_DED_NR_GL, ROE_WEIGHTED, C_PARENT_COMP, ...
- get_dividend: DVD_PER_SHARE_PRE_TAX_CASH, DATE_EX, DATE_DVD_RECORD,
                ANN_DATE, REPORT_PERIOD, DIV_PROGRESS
- get_treasury_yield(terms) -> dict[term] -> DataFrame(YIELD, index=date)
- get_stock_basic(code_list) -> DataFrame(MARKET_CODE, LIST_DATE, ...)
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .session import AmazingSession

# SDK caches downloads to disk; default 'D://AmazingData_local_data//' fails on
# machines without D:. Route all local caching into the project data dir.
LOCAL_CACHE = str((Path(__file__).resolve().parents[3] / "data" / "ad_local").resolve()) + "/"


def _today() -> date:
    return date.today()


def to_sdk_code(code6: str) -> str:
    """6-digit code -> SDK format like 601398.SH / 000651.SZ."""
    code6 = str(code6).zfill(6)
    if code6.startswith(("6", "9", "5")):
        return f"{code6}.SH"
    if code6.startswith(("4", "8")):
        return f"{code6}.BJ"
    return f"{code6}.SZ"


def from_sdk_code(code: str) -> str:
    return str(code).split(".")[0].zfill(6)


def fetch_kline(ses: AmazingSession, codes: list[str], years: int = 6) -> dict[str, pd.DataFrame]:
    """Daily K-line for codes, last `years` years. dict[code6]->df."""
    end = _today()
    begin = end - timedelta(days=int(years * 365.25))
    out: dict[str, pd.DataFrame] = {}
    sdk_codes = [to_sdk_code(c) for c in codes]
    # SDK expects int dates (yyyymmdd) — it compares against its int calendar
    begin_i, end_i = int(begin.strftime("%Y%m%d")), int(end.strftime("%Y%m%d"))
    for i in range(0, len(sdk_codes), 50):
        batch = sdk_codes[i:i + 50]
        for attempt in range(3):
            try:
                res = ses.market_data.query_kline(
                    code_list=batch,
                    begin_date=begin_i,
                    end_date=end_i,
                    period=ses.ad.constant.Period.day.value,
                )
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    print(f"[fetch] kline batch {i} failed: {e}")
                    res = {}
                else:
                    time.sleep(3)
        if isinstance(res, dict):
            for code, df in res.items():
                if df is not None and len(df) > 0:
                    out[from_sdk_code(code)] = df
    return out


def fetch_income(ses: AmazingSession, codes: list[str], from_year: str = "2015") -> pd.DataFrame:
    """Annual income statements. Columns include BASIC_EPS, ROE_WEIGHTED, C_PARENT_COMP."""
    dfs = []
    sdk_codes = [to_sdk_code(c) for c in codes]
    for attempt in range(3):
        try:
            df = ses.info_data.get_income(
                code_list=sdk_codes,
                local_path=LOCAL_CACHE,
                is_local=False,
                begin_date=f"{from_year}0101",
                end_date=_today().strftime("%Y%m%d"),
            )
            if df is not None and len(df) > 0:
                dfs.append(df)
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                print(f"[fetch] income failed: {e}")
            else:
                time.sleep(3)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    return df


def fetch_profit_express(ses: AmazingSession, codes: list[str], from_year: str = "2015") -> pd.DataFrame:
    """业绩快报 (manual §3.5.5.4). Carries ROE_WEIGHTED (official weighted ROE)
    and TOT_SHARE_EQU_EXCL_MIN_INT (parent equity) — the only working source of
    ROE on this server (balance_sheet/cash_flow return 查询失败 server-side).
    Returns DataFrame or dict[code]->DataFrame depending on SDK; normalize upstream."""
    sdk_codes = [to_sdk_code(c) for c in codes]
    for attempt in range(3):
        try:
            df = ses.info_data.get_profit_express(
                code_list=sdk_codes,
                local_path=LOCAL_CACHE,
                is_local=False,
                begin_date=f"{from_year}0101",
                end_date=_today().strftime("%Y%m%d"),
            )
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                print(f"[fetch] profit_express failed: {e}")
            else:
                time.sleep(3)
    return pd.DataFrame()


def fetch_dividend(ses: AmazingSession, codes: list[str], from_year: str = "2015") -> pd.DataFrame:
    """Dividend records (pre-tax per share, ex-date)."""
    sdk_codes = [to_sdk_code(c) for c in codes]
    for attempt in range(3):
        try:
            df = ses.info_data.get_dividend(
                code_list=sdk_codes,
                local_path=LOCAL_CACHE,
                is_local=False,
                begin_date=f"{from_year}0101",
                end_date=_today().strftime("%Y%m%d"),
            )
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                print(f"[fetch] dividend failed: {e}")
            else:
                time.sleep(3)
    return pd.DataFrame()


def fetch_treasury(ses: AmazingSession, years: int = 10) -> dict[str, pd.DataFrame]:
    """Treasury yield curves, ~`years` of history. dict[term]->df(YIELD)."""
    end = _today()
    begin = end - timedelta(days=int(years * 365.25) + 30)
    terms = ["m3", "m6", "y1", "y3", "y5", "y7", "y10"]
    try:
        return ses.info_data.get_treasury_yield(
            terms,
            local_path=LOCAL_CACHE,
            begin_date=begin.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
    except Exception as e:  # noqa: BLE001 — signature fallback
        try:
            return ses.info_data.get_treasury_yield(
                ["y10"], local_path=LOCAL_CACHE,
                begin_date=begin.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"))
        except Exception as e2:  # noqa: BLE001
            print(f"[fetch] treasury failed: {e} / {e2}")
            return {}


def fetch_cash_flow_akshare(codes: list[str], from_year: str = "2015") -> pd.DataFrame:
    """Cash-flow statements from akshare (东方财富) — fills the gap left by the
    AmazingData server whose cash-flow endpoint fails server-side.

    Primary: ak.stock_xjll_em(date=YYYY1231) — one request per annual report
    date, covers the whole market (~5200 rows). We fetch from_year..last year.
    Fallback for missing codes: ak.stock_cash_flow_sheet_by_report_em(symbol)
    (per-stock, full history incl. latest interim).

    Returns DataFrame: code, report_period(YYYY1231), ocf, net_profit, capex.
    Never raises — returns whatever it managed to fetch (宁缺毋造 upstream flags).
    """
    import akshare as ak

    code_set = {str(c).zfill(6) for c in codes}

    def _em_code(c: str) -> str:
        return ("SH" if c.startswith(("6", "9", "5")) else
                "BJ" if c.startswith(("4", "8")) else "SZ") + c

    this_year = _today().year
    years = list(range(int(from_year), this_year))  # complete annual reports only
    rows: list[pd.DataFrame] = []
    have: set[str] = set()
    for y in years:
        try:
            df = ak.stock_xjll_em(date=f"{y}1231")
        except Exception as e:  # noqa: BLE001
            print(f"[fetch] cash_flow batch {y} failed: {e}")
            continue
        if df is None or len(df) == 0:
            continue
        d = df.rename(columns={"股票代码": "code",
                               "经营性现金流-现金流量净额": "ocf",
                               "净现金流-净现金流": "ncf"})
        d["code"] = d["code"].astype(str).str.zfill(6)
        d = d[d["code"].isin(code_set)]
        if "ocf" not in d.columns:
            continue
        d = d[d["ocf"].notna()]
        if len(d):
            have.update(d["code"])
            rows.append(pd.DataFrame({
                "code": d["code"], "report_period": f"{y}1231",
                "ocf": pd.to_numeric(d["ocf"], errors="coerce"),
            }))

    # fallback: per-stock full history for codes the batch API missed
    missing = sorted(code_set - have)
    if missing:
        print(f"[fetch] cash_flow batch missed {len(missing)} codes -> per-stock fallback")
        for c in missing[:40]:  # hard cap to bound runtime
            try:
                df = ak.stock_cash_flow_sheet_by_report_em(symbol=_em_code(c))
            except Exception as e:  # noqa: BLE001
                print(f"[fetch] cash_flow per-stock {c} failed: {e}")
                continue
            if df is None or len(df) == 0 or "NETCASH_OPERATE" not in df.columns:
                continue
            d = df[df["REPORT_DATE"].astype(str).str.contains("12-31")].copy()
            d = d[d["NETCASH_OPERATE"].notna()]
            if not len(d):
                continue
            d["report_period"] = pd.to_datetime(d["REPORT_DATE"]).dt.strftime("%Y1231")
            d["code"] = c
            d = d[d["report_period"].str[:4].astype(int) >= int(from_year)]
            rows.append(pd.DataFrame({
                "code": d["code"], "report_period": d["report_period"],
                "ocf": pd.to_numeric(d["NETCASH_OPERATE"], errors="coerce"),
                "net_profit": (pd.to_numeric(d.get("NETPROFIT"), errors="coerce")
                               if "NETPROFIT" in df.columns else None),
                "capex": (pd.to_numeric(d.get("CONSTRUCT_LONG_ASSET"), errors="coerce")
                          if "CONSTRUCT_LONG_ASSET" in df.columns else None),
            }))
            time.sleep(0.5)  # be gentle with the public endpoint

    if not rows:
        return pd.DataFrame(columns=["code", "report_period", "ocf", "net_profit", "capex"])
    out = pd.concat(rows, ignore_index=True)
    for col in ("net_profit", "capex"):
        if col not in out.columns:
            out[col] = None
    return out.drop_duplicates(subset=["code", "report_period"])


def fetch_stock_basic(ses: AmazingSession, codes: list[str]) -> pd.DataFrame:
    try:
        return ses.info_data.get_stock_basic(code_list=[to_sdk_code(c) for c in codes])
    except Exception as e:  # noqa: BLE001
        print(f"[fetch] stock_basic failed: {e}")
        return pd.DataFrame()


def probe_schema(ses: AmazingSession, codes: list[str], out_path: str) -> None:
    """One-shot schema probe: dump real column names to JSON for the ingest layer."""
    info: dict = {}
    codes = [from_sdk_code(c) if "." in str(c) else str(c).zfill(6) for c in codes[:3]]
    try:
        inc = fetch_income(ses, codes[:20], from_year="2024")
        info["income_columns"] = list(inc.columns) if inc is not None and len(inc) else []
    except Exception as e:  # noqa: BLE001
        info["income_error"] = str(e)
    try:
        dv = fetch_dividend(ses, codes[:20], from_year="2024")
        info["dividend_columns"] = list(dv.columns) if dv is not None and len(dv) else []
    except Exception as e:  # noqa: BLE001
        info["dividend_error"] = str(e)
    try:
        tr = fetch_treasury(ses, years=1)
        info["treasury_keys"] = list(tr.keys())
        k0 = next(iter(tr.values())) if tr else None
        info["treasury_columns"] = list(k0.columns) if k0 is not None else []
    except Exception as e:  # noqa: BLE001
        info["treasury_error"] = str(e)
    try:
        kl = fetch_kline(ses, codes[:3], years=1)
        first = next(iter(kl.values())) if kl else None
        info["kline_columns"] = list(first.columns) if first is not None else []
    except Exception as e:  # noqa: BLE001
        info["kline_error"] = str(e)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[probe] schema written to {out_path}")
