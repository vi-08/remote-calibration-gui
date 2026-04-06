"""tests/test_filtering.py — unit tests for sigma clipping."""

import os, sys
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.filtering import sigma_clip_epoch, filter_report


def make_df(refsys_vals, mjd=60919, sttime=0):
    """Helper: build a minimal DataFrame for one epoch."""
    return pd.DataFrame({
        "SAT":    [f"G{i:02d}" for i in range(len(refsys_vals))],
        "CONST":  ["G"] * len(refsys_vals),
        "MJD":    [mjd] * len(refsys_vals),
        "STTIME": [sttime] * len(refsys_vals),
        "ELV":    [45.0] * len(refsys_vals),
        "AZ":     [180.0] * len(refsys_vals),
        "REFSYS": refsys_vals,
    })


def test_clip_removes_outlier():
    # 9 values near 1000, one at 3000 → should be clipped
    vals = [1000, 1010, 990, 1005, 995, 1002, 998, 1008, 3000]
    df  = make_df(vals)
    out = sigma_clip_epoch(df, sigma=2.0)
    assert 3000 not in out["REFSYS"].values
    assert len(out) == 8


def test_clip_keeps_small_group():
    # Only 2 points — too few to define sigma, must be kept unchanged
    df  = make_df([1000, 3000])
    out = sigma_clip_epoch(df, sigma=2.0)
    assert len(out) == 2


def test_clip_zero_std():
    # All identical values → std = 0 → all kept
    df  = make_df([500, 500, 500, 500])
    out = sigma_clip_epoch(df, sigma=2.0)
    assert len(out) == 4


def test_clip_multiple_epochs():
    # Two epochs; outlier only in second
    ep1 = make_df([1000, 1010, 990, 1005, 995], mjd=60919, sttime=0)
    ep2 = make_df([1000, 1010, 990, 5000, 995], mjd=60919, sttime=1300)
    df  = pd.concat([ep1, ep2], ignore_index=True)
    out = sigma_clip_epoch(df, sigma=2.0)
    ep2_out = out[out["STTIME"] == 1300]
    assert 5000 not in ep2_out["REFSYS"].values
    # epoch 1 untouched
    assert len(out[out["STTIME"] == 0]) == 5


def test_filter_report_counts():
    original = make_df([1000, 1010, 990, 1005, 995, 1002, 998, 1008, 3000])
    filtered = sigma_clip_epoch(original, sigma=2.0)
    report   = filter_report(original, filtered)
    assert report["total_points"]    == 9
    assert report["retained_points"] == 8
    assert report["removed_points"]  == 1
    assert report["retention_percentage"] == pytest.approx(100 * 8 / 9, rel=1e-3)


def test_clip_empty_df():
    df  = pd.DataFrame(columns=["SAT", "CONST", "MJD", "STTIME", "ELV", "AZ", "REFSYS"])
    out = sigma_clip_epoch(df)
    assert out.empty
