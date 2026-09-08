"""Unit tests: 5 factors + scoring + anchors. All synthetic data, no SDK."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hongli.config import load_config  # noqa: E402
from hongli.llm.price_anchors import (anchor_gordon, anchor_range,  # noqa: E402
                                      anchor_target_yield, anchor_yield_floor,
                                      price_zone)
from hongli.metrics.cycle import compute_cycle  # noqa: E402
from hongli.metrics.quality import compute_quality  # noqa: E402
from hongli.metrics.rate_neutral import rank_spread_v2  # noqa: E402
from hongli.metrics.sustainability import compute_sustainability  # noqa: E402
from hongli.metrics.sus_yield import compute_sus_yield  # noqa: E402
from hongli.scoring.composite import (composite_score, matrix_cell,  # noqa: E402
                                      q_relative, tier_of)
from hongli.types import StockResult, no_rating  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    # minimal test config (subset of base.yaml, inline for isolation)
    return load_config()


# ---------- factor 1: sus_yield ----------
def test_sus_yield_basic(cfg):
    eps = [1.0, 1.0, 1.1, 1.0, 1.0]
    dps = {2024: 0.6, 2023: 0.6, 2022: 0.55}
    r = compute_sus_yield(eps, dps, price=10.0, sector_type="bank", cfg=cfg)
    assert r.sus_yield is not None
    assert 0.04 < r.sus_yield < 0.08
    assert not r.flags


def test_sus_yield_missing_div(cfg):
    r = compute_sus_yield([1.0, 1.0], {}, price=10.0, sector_type="bank", cfg=cfg)
    assert r.sus_yield is None
    assert "MISSING_DIV" in r.flags


def test_sus_yield_missing_price(cfg):
    r = compute_sus_yield([1.0], {2024: 0.5}, price=None, sector_type="bank", cfg=cfg)
    assert "MISSING_KLINE" in r.flags


# ---------- factor 2: rate-neutral ----------
def _mk_y10(n_months=120, base=2.8):
    out = []
    y, m = 2016, 1
    for i in range(n_months):
        out.append((f"{y}-{m:02d}-28", base + 0.5 * np.sin(i / 12)))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def test_rank_spread_extreme_cheap(cfg):
    # price constant, dps constant -> yield flat; y10 falling -> spread rising -> cheap
    price_hist = [(f"202{i%6}-{(i%12)+1:02d}-28", 10.0) for i in range(72)]
    y10 = _mk_y10()
    r = rank_spread_v2(price_hist, y10, price_hist, dps_ttm=0.6,
                       prev_s_val=None, cfg=cfg)
    assert r.s_val is not None
    assert 0.0 <= r.s_val <= 1.0
    assert r.rank_layer is not None


def test_rank_spread_insufficient_history(cfg):
    price_hist = [("2024-01-31", 10.0), ("2024-02-28", 10.0)]
    r = rank_spread_v2(price_hist, _mk_y10(), price_hist, dps_ttm=0.6,
                       prev_s_val=None, cfg=cfg)
    assert r.s_val is None
    assert len(r.flags) > 0


def test_rank_spread_no_div(cfg):
    price_hist = [(f"202{i%6}-{(i%12)+1:02d}-28", 10.0) for i in range(72)]
    r = rank_spread_v2(price_hist, _mk_y10(), price_hist, dps_ttm=None,
                       prev_s_val=None, cfg=cfg)
    assert r.s_val is None


# ---------- factor 3: sustainability ----------
def test_sustainability_ideal_payout(cfg):
    dps = {y: 0.5 for y in range(2020, 2025)}
    eps = {y: 1.0 for y in range(2020, 2025)}
    r = compute_sustainability(dps, eps, None, cfg)
    assert r.s_sus is not None and r.s_sus > 0.5
    assert abs(r.payout_now - 0.5) < 1e-9
    assert r.cont_years == 5


def test_sustainability_excess_payout(cfg):
    dps = {2024: 1.5}
    eps = {2024: 1.0}
    r = compute_sustainability(dps, eps, None, cfg)
    assert r.s_sus < 0.6  # penalized


# ---------- factor 4: cycle ----------
def test_cycle_trough(cfg):
    r = compute_cycle([2.0, 2.0, 2.0, 2.0, 0.5])
    assert r.s_cyc > 0.8  # current EPS at trough -> high cycle score


def test_cycle_missing(cfg):
    r = compute_cycle([1.0])
    assert r.s_cyc is None and "MISSING_FIN_EPS" in r.flags


# ---------- factor 5: quality ----------
def test_quality_stable_high_roe(cfg):
    roe = [0.12, 0.12, 0.12, 0.12, 0.12]
    dps = {y: 0.5 for y in range(2020, 2025)}
    r = compute_quality(roe, dps, cfg)
    assert r.s_q > 0.7


def test_quality_negative_roe(cfg):
    roe = [-0.05, 0.02, -0.01, 0.03, -0.02]
    dps = {2024: 0.5}
    r = compute_quality(roe, dps, cfg)
    assert r.s_q < 0.4


# ---------- scoring & matrix ----------
def test_composite_renorm(cfg):
    s, flags = composite_score(0.8, 0.7, None, 0.6, "bank", "bank", cfg)
    assert s is not None and 0 <= s <= 100
    assert "cyc" in flags


def test_matrix_cells(cfg):
    assert matrix_cell(0.30, 1.2, cfg) == "low_A"
    assert matrix_cell(0.30, 0.7, cfg) == "low_C"
    assert matrix_cell(0.50, 1.0, cfg) == "mid_B"
    assert matrix_cell(0.80, 0.7, cfg) == "high_C"
    assert matrix_cell(None, 1.2, cfg) == "no_data"


def test_tier_order(cfg):
    assert tier_of(95, cfg) == "低估"
    assert tier_of(5, cfg) == "极高估"
    assert tier_of(None, cfg) == "无评级"


def test_no_rating_rule(cfg):
    assert no_rating(["MISSING_KLINE", "MISSING_DIV"]) is True
    assert no_rating(["MISSING_KLINE"]) is False
    assert no_rating(["INSUFFICIENT_SPREAD_HISTORY"]) is False


def test_q_relative_sector_median(cfg):
    # sector "x" has no median -> falls back to universe median (0.6) -> 0.8/0.6
    rows = [StockResult(code="a", s_sus=0.8, s_q=0.8), StockResult(code="b", s_sus=0.4, s_q=0.4)]
    from hongli.scoring.composite import sector_medians_of, universe_median_of
    med = sector_medians_of(rows)
    q = q_relative(0.8, 0.8, "x", med, universe_median_of(rows), cfg)
    assert q == pytest.approx(0.8 / 0.6)
    # exact sector median: 3 stocks with raw q 0.6/0.8/1.0 -> median 0.8 -> top stock = 1.0
    rows2 = [StockResult(code="p", sector="s", s_sus=0.5, s_q=0.7),
             StockResult(code="q", sector="s", s_sus=0.7, s_q=0.9),
             StockResult(code="r", sector="s", s_sus=0.9, s_q=1.1)]
    med2 = sector_medians_of(rows2)
    # middle stock raw q = 0.5·0.7+0.5·0.9 = 0.8 == median -> q_rel = 1.0 exactly
    q2 = q_relative(0.7, 0.9, "s", med2, universe_median_of(rows2), cfg)
    assert q2 == pytest.approx(1.0)


# ---------- anchors ----------
def test_anchors(cfg):
    a = anchor_target_yield(0.5, cfg["llm"]["price_anchors"]["target_yield_floor"])
    assert a == pytest.approx(0.5 / 0.045)
    g = anchor_gordon(0.5, 0.12, 0.5, 2.5, cfg)
    assert g and g > 0
    c = anchor_yield_floor(0.5, cfg["llm"]["price_anchors"]["yield_floor"])
    assert c == pytest.approx(0.5 / 0.045)
    hist = list(np.linspace(8, 12, 250))
    b = anchor_range(hist, cfg["llm"]["price_anchors"]["range_pos"])
    assert 8 < b < 12


def test_price_zone(cfg):
    z = price_zone({"A_target_yield": 10.0, "G_gordon": 12.0, "C_yield_floor": 11.0}, 9.0)
    assert z == (9.0, 12.0)
