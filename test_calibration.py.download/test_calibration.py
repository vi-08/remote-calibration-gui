"""tests/test_calibration.py — unit tests for the calibration pipeline."""

import os, sys
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.calibration import run_calibration, epoch_mean, common_view_filter, mjd_to_date, sttime_to_hhmmss


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_records(refsys_map, const="G"):
    """
    refsys_map: { (mjd, sttime): [list of REFSYS values] }
    Returns a DataFrame with one row per satellite per epoch.
    """
    rows = []
    for (mjd, sttime), vals in refsys_map.items():
        for i, v in enumerate(vals):
            rows.append({
                "SAT": f"{const}{i+1:02d}", "CONST": const,
                "MJD": mjd, "STTIME": sttime,
                "ELV": 45.0, "AZ": 0.0, "REFSYS": v,
            })
    return pd.DataFrame(rows)


# ── mjd_to_date ───────────────────────────────────────────────────────────────

def test_mjd_to_date_known():
    # MJD 60919 = 1 September 2025
    assert mjd_to_date(60919) == "01-09-2025"


def test_sttime_to_hhmmss():
    assert sttime_to_hhmmss(130000) == "13:00:00"
    assert sttime_to_hhmmss(0)      == "00:00:00"
    assert sttime_to_hhmmss(235959) == "23:59:59"


# ── epoch_mean ────────────────────────────────────────────────────────────────

def test_epoch_mean_simple():
    df = make_records({(60919, 0): [1000, 1100, 90