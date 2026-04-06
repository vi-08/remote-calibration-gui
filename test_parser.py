"""tests/test_parser.py — unit tests for the CGGTTS / CSV parser."""

import io
import os
import tempfile
import pandas as pd
import pytest

# Make sure the package root is on the path when running from tests/
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.parser import parse_cggtts_file, parse_csv_file, load_files

# ── Minimal synthetic CGGTTS content ─────────────────────────────────────────
CGGTTS_CONTENT = """\
CGGTTS GENERIC DATA FORMAT VERSION = 2E
REV DATE = 2020-01-01
SAT CL  MJD  STTIME REFSV   REFSYS  ELV AZ  MDTR  MDIO MSIO SMDI ISGF DSG ISG TRKL CK
G01 FF 60919 000000      -95    +1172  532  130   815    -39    -17    3    4    6   40  780 00
G14 FF 60919 000000      +69    +1214  325 1839  1101    -15    +21    0    1   32   26  780 00
E01 FF 60919 000000      +20    +1100  400  200   900      0      0    0    0   10   10  780 00
G01 FF 60919 001300      -10    +1180  600  180   820    -20    -10    2    3    8   35  780 00
"""

CSV_CONTENT = """\
MJD,STTIME,REFSYS
60919,0,1172
60919,1300,1180
60920,0,1200
"""


# ── Tests ─────────────────────────────────────────────────────────────────────

def write_tmp(content: str, suffix: str = ".txt"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def test_parse_cggtts_gps_only():
    path = write_tmp(CGGTTS_CONTENT)
    try:
        df = parse_cggtts_file(path, ["GPS"], elev_min=10)
        # Only G lines should be parsed
        assert (df["CONST"] == "G").all()
        assert len(df) == 3  # G01 epoch0, G14 epoch0, G01 epoch1
    finally:
        os.unlink(path)


def test_parse_cggtts_multi_const():
    path = write_tmp(CGGTTS_CONTENT)
    try:
        df = parse_cggtts_file(path, ["GPS", "Galileo"], elev_min=10)
        assert set(df["CONST"]) == {"G", "E"}
        assert len(df) == 4
    finally:
        os.unlink(path)


def test_parse_cggtts_elev_filter():
    path = write_tmp(CGGTTS_CONTENT)
    try:
        # ELV=53.2 (532/10), 32.5, 40.0, 60.0 — all >= 10 so nothing filtered
        df = parse_cggtts_file(path, ["GPS"], elev_min=50)
        # Only G01 epoch0 (elv=53.2) and G01 epoch1 (elv=60.0) pass 50°
        assert len(df) == 2
    finally:
        os.unlink(path)


def test_parse_cggtts_refsys_conversion():
    path = write_tmp(CGGTTS_CONTENT)
    try:
        df = parse_cggtts_file(path, ["GPS"], elev_min=10)
        # Raw REFSYS for G01 epoch0 = +1172 (raw 0.1-ns units)
        row = df[(df["SAT"] == "G01") & (df["STTIME"] == 0)].iloc[0]
        assert row["REFSYS"] == pytest.approx(1172.0)
    finally:
        os.unlink(path)


def test_parse_csv_basic():
    path = write_tmp(CSV_CONTENT, suffix=".csv")
    try:
        df = parse_csv_file(path)
        assert len(df) == 3
        assert set(df.columns).issuperset({"MJD", "STTIME", "REFSYS"})
        assert df.iloc[0]["REFSYS"] == 1172.0
    finally:
        os.unlink(path)


def test_load_files_mixed():
    p_cggtts = write_tmp(CGGTTS_CONTENT, suffix=".txt")
    p_csv    = write_tmp(CSV_CONTENT,    suffix=".csv")
    try:
        df = load_files([p_cggtts, p_csv], ["GPS"], elev_min=10)
        # 3 GPS rows from CGGTTS + 3 rows from CSV = 6
        assert len(df) == 6
    finally:
        os.unlink(p_cggtts)
        os.unlink(p_csv)


def test_load_files_empty_on_wrong_const():
    path = write_tmp(CGGTTS_CONTENT)
    try:
        df = load_files([path], ["BeiDou"], elev_min=10)
        assert df.empty or (df["CONST"] == "C").all()
    finally:
        os.unlink(path)
