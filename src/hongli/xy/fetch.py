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
