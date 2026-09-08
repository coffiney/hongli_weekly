"""Unit tests: cash-flow quality metric (OCF/NP, 七维检验口径). Synthetic only."""
import os
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hongli.config import load_config  # noqa: E402
from hongli.ingest import normalize_cash_flow  # noqa: E402
from hongli.metrics.cashflow import compute_ocf_np  # noqa: E402
from hongli.metrics.sustainability import compute_sustainability  # noqa: E402
import pandas as pd  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_ocf_np_median_of_3():
    ocf = {2022: 150.0, 2023: 120.0, 2024: 180.0}
    np_ = {2022: 100.0, 2023: 100.0, 2024: 100.0}
    # ratios 1.5, 1.2, 1.8 -> median 1.5
    assert compute_ocf_np(ocf, np_) == pytest.approx(1.5)


def test_ocf_np_needs_two_years():
    assert compute_ocf_np({2024: 150.0}, {2024: 100.0}) is None
    assert compute_ocf_np({}, {}) is None


def test_ocf_np_skips_bad_pairs():
    ocf = {2023: 120.0, 2024: 90.0}
    np_ = {2023: 0.0, 2024: 100.0}  # 2023 np=0 unusable -> only 1 valid pair -> None
    assert compute_ocf_np(ocf, np_) is None


def test_cfo_pass_thresholds(cfg):
    from hongli.metrics.cashflow import cfo_pass
    assert cfo_pass(1.5, cfg) is True     # 优秀
    assert cfo_pass(1.0, cfg) is True     # 合格线
    assert cfo_pass(0.99, cfg) is False
    assert cfo_pass(None, cfg) is None    # 数据不足


def test_sustainability_with_cfo(cfg):
    dps = {2022: 0.6, 2023: 0.6, 2024: 0.6}
    eps = {2022: 1.0, 2023: 1.0, 2024: 1.0}
    r_no = compute_sustainability(dps, eps, None, cfg, ocf_np_ratio=None)
    r_yes = compute_sustainability(dps, eps, None, cfg, ocf_np_ratio=1.6)
    assert r_yes.s_sus is not None and r_no.s_sus is not None
    # strong cash flow must lift sustainability vs missing-cfo baseline
    assert r_yes.s_sus > r_no.s_sus


def test_sustainability_weak_cfo_penalizes(cfg):
    dps = {2022: 0.6, 2023: 0.6, 2024: 0.6}
    eps = {2022: 1.0, 2023: 1.0, 2024: 1.0}
    r_bad = compute_sustainability(dps, eps, None, cfg, ocf_np_ratio=0.4)
    r_good = compute_sustainability(dps, eps, None, cfg, ocf_np_ratio=1.5)
    assert r_good.s_sus > r_bad.s_sus


def test_normalize_cash_flow():
    df = pd.DataFrame({
        "code": ["601398.SH", "601398", "000651.SZ"],
        "report_period": ["20241231", "20240630", "20241231"],
        "ocf": [100.0, 50.0, None],
        "net_profit": [90.0, 40.0, 30.0],
        "capex": [10.0, 5.0, None],
    })
    out = normalize_cash_flow(df)
    # interim row dropped; ocf=None row dropped; code normalized to 6-digit
    assert len(out) == 1
    assert out.iloc[0]["code"] == "601398"
    assert out.iloc[0]["report_period"] == "20241231"


def test_normalize_cash_flow_empty():
    out = normalize_cash_flow(pd.DataFrame())
    assert len(out) == 0
    assert set(out.columns) == {"code", "report_period", "ocf", "net_profit", "capex"}
