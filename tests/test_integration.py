"""Integration test: end-to-end pipeline on synthetic DuckDB data (no SDK).

Proves: ingest normalize -> DB -> compute -> matrix -> HTML -> postcheck all work.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hongli.config import load_config  # noqa: E402
from hongli.db import connect, init_schema  # noqa: E402
from hongli.ingest import (normalize_dividend, normalize_income,  # noqa: E402
                           normalize_kline, normalize_treasury)
from hongli.pipeline import compute_all, rows_for_report  # noqa: E402
from hongli.report.html_report import postcheck, render_report  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def db(tmp_path_factory, cfg):
    tmp = tmp_path_factory.mktemp("db")
    con = connect(str(tmp / "test.duckdb"))
    init_schema(con)

    # --- synthetic universe data for 3 stocks ---
    codes = ["601398", "600900", "000651"]
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", "2026-09-04")
    kdf = pd.concat([
        pd.DataFrame({
            "code": c,
            "kline_time": [d.date() for d in dates],
            "open": 10.0, "high": 10.5, "low": 9.5,
            "close": 10.0 + i * 0.01 + rng.normal(0, 0.05, len(dates)),
            "volume": 1e6, "amount": 1e7,
        }) for i, c in enumerate(codes)])
    con.execute("INSERT INTO kline SELECT * FROM kdf")

    inc_rows = []
    for c in codes:
        for y in range(2015, 2025):  # annual reports 2015..2024 (report_period ends 1231)
            inc_rows.append((c, pd.Timestamp(f"{y+1}-03-30").date(), f"{y}1231", "1",
                             0.9 + 0.02 * (y % 5), 450.0, 4000.0, 5000.0, 430.0))
    idf = pd.DataFrame(inc_rows, columns=["code", "ann_date", "report_period", "statement_type",
                                          "eps", "parent_net_profit", "parent_equity",
                                          "revenue", "net_profit_deducted"])
    idf["report_period"] = idf["report_period"].astype(str)
    con.execute("INSERT INTO income SELECT * FROM idf")

    div_rows = []
    for c in codes:
        for y in range(2015, 2026):
            div_rows.append((c, pd.Timestamp(f"{y+1}-06-15").date(),
                             pd.Timestamp(f"{y+1}-06-20").date(), 0.55 + 0.01 * (y % 3), f"{y}1231"))
    ddf = pd.DataFrame(div_rows, columns=["code", "ann_date", "ex_date", "dps_pretax", "report_period"])
    con.execute("INSERT INTO dividend SELECT * FROM ddf")

    ty = pd.DataFrame({
        "tdate": [d.date() for d in pd.bdate_range("2016-01-01", "2026-09-04")],
        "y10": 3.0,
    })
    con.execute("INSERT INTO treasury SELECT * FROM ty")
    yield con


UNIVERSE = [
    {"code": "601398", "name": "工商银行", "sector": "bank"},
    {"code": "600900", "name": "长江电力", "sector": "power"},
    {"code": "000651", "name": "格力电器", "sector": "consumer"},
]


def test_normalizers():
    k = normalize_kline({"601398": pd.DataFrame({
        "kline_time": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "close": [10.0, 10.1], "open": [10, 10], "high": [10.2, 10.3],
        "low": [9.9, 10], "volume": [1, 1], "amount": [1, 1]})})
    assert len(k) == 2 and set(k.columns) >= {"code", "kline_time", "close"}

    inc = normalize_income(pd.DataFrame({
        "MARKET_CODE": ["601398.SH"], "ANN_DATE": ["2024-03-30"],
        "REPORTING_PERIOD": ["20231231"], "STATEMENT_TYPE": ["1"],
        "BASIC_EPS": [1.0], "NET_PRO_EXCL_MIN_INT_INC": [90.0],
        "C_PARENT_COMP": [1000], "TOT_OPERA_REV": [1000],
        "NET_PRO_AFTER_DED_NR_GL": [880]}))
    assert len(inc) == 1 and inc.iloc[0]["eps"] == 1.0
    assert inc.iloc[0]["statement_type"] == "1"

    dv = normalize_dividend(pd.DataFrame({
        "MARKET_CODE": ["601398.SH"], "ANN_DATE": ["2024-06-01"],
        "DATE_EX": ["2024-06-05"], "DVD_PER_SHARE_PRE_TAX_CASH": [0.3],
        "REPORT_PERIOD": ["20231231"]}))
    assert len(dv) == 1 and dv.iloc[0]["dps_pretax"] == 0.3

    tr = normalize_treasury({"y10": pd.DataFrame({"YIELD": [2.5, 2.6]},
                                                index=pd.to_datetime(["2024-01-01", "2024-01-02"]))})
    assert len(tr) == 2


def test_full_pipeline_and_report(cfg, db, tmp_path, monkeypatch):
    con = db
    results = compute_all(con, UNIVERSE, cfg, asof="2026-09-04")
    assert len(results) == 3
    for r in results:
        assert r.price is not None
        # synthetic data is complete: core factors should compute
        assert r.s_sus is not None
        assert r.matrix_cell in {"low_A", "low_B", "low_C", "mid_A", "mid_B",
                                 "mid_C", "high_A", "high_B", "high_C", "no_data"}
    rows = rows_for_report(results)
    monkeypatch.chdir(tmp_path)  # report out_dir relative
    html = render_report(rows, "2026-09-04", "test-run", cfg, llm_text=None)
    assert html.exists()
    problems = postcheck(html, rows)
    assert problems == [], f"postcheck problems: {problems}"
    txt = html.read_text(encoding="utf-8")
    assert "双维矩阵" in txt and "601398" in txt
