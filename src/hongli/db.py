"""DuckDB data layer — single writer, read-only queries (tech plan §1.3).

Schema (v2.3):
- kline(code DATE close volume amount)
- income(code, ann_date, report_period, eps, roe, parent_equity, revenue)
- dividend(code, ann_date, ex_date, dps_pretax, report_period)
- treasury(date, y10)
- runs(run_id, asof, meta JSON)
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS kline (
    code VARCHAR, kline_time DATE, open DOUBLE, high DOUBLE, low DOUBLE,
    close DOUBLE, volume DOUBLE, amount DOUBLE,
    PRIMARY KEY (code, kline_time)
);
CREATE TABLE IF NOT EXISTS income (
    code VARCHAR, ann_date DATE, report_period VARCHAR, statement_type VARCHAR,
    eps DOUBLE, parent_net_profit DOUBLE, parent_equity DOUBLE, revenue DOUBLE,
    net_profit_deducted DOUBLE,
    PRIMARY KEY (code, report_period, statement_type)
);
CREATE TABLE IF NOT EXISTS dividend (
    code VARCHAR, ann_date DATE, ex_date DATE, dps_pretax DOUBLE,
    report_period VARCHAR
);CREATE TABLE IF NOT EXISTS profit_express (
    code VARCHAR, ann_date DATE, report_period VARCHAR,
    roe_weighted DOUBLE, parent_equity DOUBLE, eps DOUBLE, net_asset_ps DOUBLE
);
CREATE TABLE IF NOT EXISTS treasury (
    tdate DATE, y10 DOUBLE,
    PRIMARY KEY (tdate)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR, as_of DATE, created_at TIMESTAMP, meta JSON,
    PRIMARY KEY (run_id)
);
"""


def connect(db_path: str, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(db_path, read_only=read_only)


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA_SQL)


def upsert_dataframe(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame,
                     key_cols: list[str]) -> int:
    """Idempotent upsert: delete matching keys, then insert. Works with or
    without PK/UNIQUE constraints. Returns net row delta."""
    if df is None or len(df) == 0:
        return 0
    cols = list(df.columns)
    con.register("_tmp_upsert", df)
    n_before = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    con.execute(f"DELETE FROM {table} WHERE ({', '.join(key_cols)}) IN "
                f"(SELECT {', '.join(key_cols)} FROM _tmp_upsert)")
    con.execute(f"INSERT INTO {table} ({', '.join(cols)}) "
                f"SELECT {', '.join(cols)} FROM _tmp_upsert")
    n_after = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    con.unregister("_tmp_upsert")
    return int(n_after - n_before)


def record_run(con: duckdb.DuckDBPyConnection, run_id: str, asof: str, meta: dict) -> None:
    con.execute(
        "INSERT OR REPLACE INTO runs VALUES (?, ?, now(), ?)",
        [run_id, asof, json.dumps(meta, ensure_ascii=False)],
    )