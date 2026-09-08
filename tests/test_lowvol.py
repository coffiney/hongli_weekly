# -*- coding: utf-8 -*-
"""Tests for lowvol.py — 52w annualized volatility of weekly-sampled closes."""
import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hongli.metrics.lowvol import compute_ann_vol


def _synthetic(weekly_sd: float, n_weeks: int = 70, seed: int = 42) -> list[float]:
    """Daily closes where each week's close-to-close return ~ N(0, weekly_sd)."""
    rng = random.Random(seed)
    closes = [100.0]
    for _ in range(n_weeks):
        weekly_ret = rng.gauss(0.0, weekly_sd)
        for _ in range(5):
            closes.append(closes[-1] * math.exp(weekly_ret / 5))
        closes[-1] = closes[-2] * math.exp(weekly_ret) if len(closes) > 1 else closes[-1]
    return closes


def test_low_vol_series_returns_low_vol():
    # 1% weekly sd -> annualized ≈ sqrt(52) ≈ 7.2% -> well under 25%
    v = compute_ann_vol(_synthetic(0.01))
    assert v is not None
    assert 0.03 < v < 0.15


def test_high_vol_series_returns_high_vol():
    # 6% weekly sd -> annualized ≈ 43% -> above 25% threshold
    v = compute_ann_vol(_synthetic(0.06))
    assert v is not None
    assert v > 0.30


def test_insufficient_data_returns_none():
    assert compute_ann_vol([]) is None
    assert compute_ann_vol([100.0, 101.0, 99.0]) is None  # way too short
    assert compute_ann_vol([100.0] * 100) is None  # only 20 weeks of samples


def test_flat_series_zero_vol():
    prices = [100.0] * 300
    v = compute_ann_vol(prices)
    assert v is not None and v < 1e-9


def test_zero_price_guards():
    prices = [0.0] * 10 + [100.0] * 300
    v = compute_ann_vol(prices)
    assert v is not None and v < 1e-9  # zero-price pairs skipped


def test_threshold_semantics():
    """七维检验 low-vol rule: 年化波动<=25% 达标."""
    low = compute_ann_vol(_synthetic(0.02, seed=7))    # ~14% annualized
    high = compute_ann_vol(_synthetic(0.06, seed=8))   # ~43% annualized
    assert low is not None and low <= 0.25
    assert high is not None and high > 0.25
