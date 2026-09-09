"""Unit tests: recommendation ranking (scoring/ranking.py). Synthetic only."""
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hongli.scoring.ranking import recommend_key, sort_candidates  # noqa: E402


def _row(cell, comp=50.0, ocf=1.2, vol=0.20, **kw):
    return {"matrix_cell": cell, "composite": comp, "ocf_np": ocf,
            "ann_vol": vol, **kw}


def test_cell_priority_low_A_first():
    rows = [_row("low_B", comp=90), _row("mid_A", comp=50), _row("low_A", comp=40)]
    out = sort_candidates(rows)
    assert [r["matrix_cell"] for r in out] == ["low_A", "mid_A", "low_B"]


def test_cashflow_alarm_demoted_within_same_cell():
    good = _row("low_A", comp=50, ocf=1.5)
    alarm = _row("low_A", comp=80, ocf=0.4)  # higher composite but bad cashflow
    out = sort_candidates([alarm, good])
    assert out[0] is good


def test_lowvol_preferred_at_same_tier():
    calm = _row("low_A", comp=50, vol=0.20)
    wild = _row("low_A", comp=70, vol=0.35)
    out = sort_candidates([wild, calm])
    assert out[0] is calm


def test_composite_tiebreak():
    a = _row("low_A", comp=60)
    b = _row("low_A", comp=55)
    out = sort_candidates([b, a])
    assert out[0] is a


def test_missing_data_tolerated():
    # no ocf / no vol -> sorts between alarm and healthy-lowvol, never crashes
    r = {"matrix_cell": "low_A", "composite": 50, "ocf_np": None, "ann_vol": None}
    out = sort_candidates([r])
    assert len(out) == 1


def test_non_candidate_cells_last():
    stray = _row("high_A", comp=95)
    cand = _row("low_B", comp=30)
    out = sort_candidates([stray, cand])
    assert out[0] is cand


def test_feishu_pick_matches_html_order():
    """notify._pick_watchlist must take the first N of the pre-ordered rows."""
    from hongli.notify import _pick_watchlist
    cfg = {"matrix": {"llm_candidate_cells": ["low_A", "low_B", "mid_A"]}}
    rows = [_row("low_A", comp=40), _row("mid_A", comp=70),
            _row("low_B", comp=90), _row("low_A", comp=35)]
    ordered = sort_candidates(rows)  # real pipeline: rows_for_report sorts first
    picked = _pick_watchlist(ordered, cfg, top_n=3)
    assert [r["matrix_cell"] for r in picked] == ["low_A", "low_A", "mid_A"]
