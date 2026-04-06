"""
NMC Remote Calibration System — Flask Backend
National Metrology Centre, Singapore

Endpoints:
  POST /api/calibrate   — parse CGGTTS/CSV files, compute time differences
  POST /api/mdev        — compute Modified Allan Deviation

CGGTTS column layout (fixed-width, ITU-R TF.1153):
  V2E: PRN  CL  MJD  STTIME  TRKL  ELV  AZTH  REFSV  SRSV  REFSYS  SRSYS  DSG  IOE  MDTR  SMDT  MDIO  SMDI  MSIO  SMSI  ISG  FR  HC  FRC  CK
  V1E: PRN  CL  MJD  STTIME  REFSV  SRSV  REFSYS  SRSYS  DSG  IOE  MDTR  SMDT  MDIO  SMDI  CK
  Each REFSYS value is in units of 0.1 ns  →  divide by 10 to get ns.
"""

from __future__ import annotations
import io, json, math, re, traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    import allantools
    HAS_ALLANTOOLS = True
except ImportError:
    HAS_ALLANTOOLS = False

app = Flask(__name__)
CORS(app)

class _NumpyEncoder(json.JSONEncoder):
    """Make numpy scalars and arrays JSON-serializable."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json_encoder = _NumpyEncoder

# ──────────────────────────────────────────────────────────────────────────────
# MJD helpers
# ──────────────────────────────────────────────────────────────────────────────

def mjd_to_date(mjd: float) -> str:
    """Convert MJD (float) to DD-MM-YYYY string."""
    jd = mjd + 2400000.5
    # Algorithm from Jean Meeus, Astronomical Algorithms
    jd = jd + 0.5
    Z = int(jd)
    F = jd - Z
    if Z < 2299161:
        A = Z
    else:
        alpha = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - alpha // 4
    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)
    day   = B - D - int(30.6001 * E)
    month = E - 1 if E < 14 else E - 13
    year  = C - 4716 if month > 2 else C - 4715
    return f"{day:02d}-{month:02d}-{year:04d}"

def sttime_to_hms(sttime_s: int) -> str:
    """Convert STTIME (seconds-of-day) to HH:MM:SS."""
    h = sttime_s // 3600
    m = (sttime_s % 3600) // 60
    s = sttime_s % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ──────────────────────────────────────────────────────────────────────────────
# CGGTTS parser
# ──────────────────────────────────────────────────────────────────────────────

# Constellation character → name
CONST_CHAR = {'G': 'GPS', 'E': 'Galileo', 'R': 'GLONASS', 'C': 'BeiDou', 'J': 'QZSS'}

# Column indices for each format variant, keyed by (version, n_tokens).
# V2E full row: PRN CL MJD STTIME TRKL ELV AZTH REFSV SRSV REFSYS SRSYS DSG IOE MDTR SMDT MDIO SMDI MSIO SMSI ISG FR HC FRC CK
#  index:        0   1  2    3     4    5    6    7     8     9     10    11  12   13   14   15   16   17   18  19  20 21  22  23
# V1E row:      PRN CL MJD STTIME REFSV SRSV REFSYS SRSYS DSG IOE MDTR SMDT MDIO SMDI CK
#  index:        0   1  2    3      4     5     6      7    8   9   10   11   12   13  14

_V2E_IDX = {'mjd': 2, 'sttime': 3, 'elv': 5, 'refsys': 9, 'min_tokens': 10}
_V1E_IDX = {'mjd': 2, 'sttime': 3, 'elv': None, 'refsys': 6, 'min_tokens': 7}


def _detect_version(lines: list[str]) -> str:
    """
    Return '2E' or '1E' by inspecting header lines.
    A v2E file has 'GENERIC DATA FORMAT VERSION = 2E' in its header,
    OR a data row with 24 whitespace-separated tokens (the v2E row width).
    """
    for line in lines:
        upper = line.upper()
        if 'VERSION' in upper and '2E' in upper:
            return '2E'
        if 'GENERIC DATA FORMAT' in upper and '2E' in upper:
            return '2E'
    # Fallback: inspect first data-looking line token count
    for line in lines:
        stripped = line.strip().rstrip('\r')
        if not stripped or stripped.startswith('#'):
            continue
        tokens = stripped.split()
        if len(tokens) >= 1 and re.match(r'^[A-Z]\d+$', tokens[0]):
            # A data line — count tokens to distinguish format
            if len(tokens) >= 20:
                return '2E'
            else:
                return '1E'
    return '1E'


def _find_data_start(lines: list[str]) -> int:
    """
    Locate the index of the first actual data line in the file.

    CGGTTS v2E files have this structure:
        ... header key=value lines ...
        CKSUM = xx
        <blank line>
        SAT CL  MJD  STTIME TRKL ELV AZTH ...   ← column header
                     hhmmss  s  .1dg ...          ← unit row
        C01 FF 60919 001400 ...                   ← first data line  ← we want this

    v1E files are similar but may have a different column header line.

    Strategy: find the line that starts with "SAT" and contains "MJD",
    then skip any immediately following non-data lines (unit rows, blank lines).
    """
    for i, line in enumerate(lines):
        stripped = line.strip().rstrip('\r')
        if stripped.upper().startswith('SAT') and 'MJD' in stripped.upper():
            # Found column header — skip unit/continuation rows after it
            j = i + 1
            while j < len(lines):
                candidate = lines[j].strip().rstrip('\r')
                # A real data line starts with a PRN token like G01, C14, E05, R07 …
                if re.match(r'^[A-Z]\d+\s', candidate):
                    return j
                j += 1
            return i + 1  # fallback: line right after SAT header
    # Last-resort: find first line matching a PRN pattern
    for i, line in enumerate(lines):
        if re.match(r'^\s*[A-Z]\d+\s', line):
            return i
    return 0


def parse_cggtts(content: str, allowed_constellations: list[str]) -> pd.DataFrame:
    """
    Parse a CGGTTS file (version 1E or 2E) and return a DataFrame with columns:
        PRN, CONST, MJD, STTIME, REFSYS_ns, ELV

    Handles:
      - CRLF and LF line endings
      - Multiple day-blocks concatenated in one file (each with its own header)
      - V2E 24-token rows (REFSYS at index 9)
      - V1E shorter rows (REFSYS at index 6)
      - REFSYS converted from 0.1 ns units → ns (divide by 10)
    """
    # Normalise line endings
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    lines = content.splitlines()

    version = _detect_version(lines)
    idx = _V2E_IDX if version == '2E' else _V1E_IDX

    records = []

    # We iterate the whole file; when we hit a new SAT header, we recalculate
    # the data start (handles multi-day concatenated files).
    in_data = False
    for raw in lines:
        line = raw.strip()

        # Detect start/restart of a data block (new header or new day block)
        if line.upper().startswith('SAT') and 'MJD' in line.upper():
            in_data = True
            continue

        # Skip unit rows, blank lines, comments, and non-data header lines
        if not in_data:
            continue
        if not line or line.startswith('#'):
            continue
        # Unit continuation row (starts with 'hhmmss' or all non-alphanumeric)
        if line.lower().startswith('hhmmss') or line.startswith('.'):
            continue
        # New file header block embedded in concatenated file
        if line.upper().startswith('CGGTTS'):
            in_data = False
            continue
        # Skip key=value header lines that may appear between blocks
        if '=' in line and not re.match(r'^[A-Z]\d+\s', line):
            continue

        tokens = line.split()
        if len(tokens) < idx['min_tokens']:
            continue

        try:
            prn_field = tokens[0]
            # PRN must match pattern like G01, C14, E05, R07
            if not re.match(r'^[A-Z]\d+$', prn_field):
                continue

            const_char = prn_field[0]
            const_name = CONST_CHAR.get(const_char)
            if const_name is None or const_name not in allowed_constellations:
                continue

            mjd    = int(tokens[idx['mjd']])
            sttime_raw = tokens[idx['sttime']]
            # STTIME is stored as HHMMSS integer in the file (e.g. 001400 = 01h 14m 00s)
            sttime_hhmmss = int(sttime_raw)
            hh = sttime_hhmmss // 10000
            mm = (sttime_hhmmss % 10000) // 100
            ss = sttime_hhmmss % 100
            sttime = hh * 3600 + mm * 60 + ss   # convert to seconds-of-day

            # Elevation
            if idx['elv'] is not None and len(tokens) > idx['elv']:
                elv_raw = int(tokens[idx['elv']])
                elv_deg = elv_raw / 10.0
            else:
                elv_deg = 45.0  # V1E default if not present

            if elv_deg < 10.0:
                continue

            refsys_raw = int(tokens[idx['refsys']])

            # Reject sentinel / invalid values (9999999 is the standard CGGTTS bad-value flag)
            if refsys_raw == 9999999 or abs(refsys_raw) > 9_000_000:
                continue

            records.append({
                'PRN':       prn_field,
                'CONST':     const_name,
                'MJD':       mjd,
                'STTIME':    sttime,
                'REFSYS_ns': refsys_raw / 10.0,  # 0.1 ns → ns
                'ELV':       elv_deg,
            })

        except (ValueError, IndexError):
            continue

    if not records:
        return pd.DataFrame(columns=['PRN', 'CONST', 'MJD', 'STTIME', 'REFSYS_ns', 'ELV'])
    return pd.DataFrame(records)


def parse_csv_refsys(content: str) -> pd.DataFrame:
    """
    Parse a customer CSV file.
    Expected columns (case-insensitive): MJD, STTIME, REFSYS
    REFSYS must already be in ns (not 0.1 ns).
    Also accepts 'REFSYS_ns' or 'TIME_DIFF'.
    """
    try:
        df = pd.read_csv(io.StringIO(content))
    except Exception:
        raise ValueError("Could not parse CSV file. Ensure it is comma-separated with headers.")

    df.columns = [c.strip().upper() for c in df.columns]

    # Flexible column mapping
    col_map = {}
    for col in df.columns:
        if col in ('MJD',):
            col_map['MJD'] = col
        elif col in ('STTIME', 'STTIME_S', 'TIME_S', 'EPOCH_S'):
            col_map['STTIME'] = col
        elif col in ('REFSYS', 'REFSYS_NS', 'TIME_DIFF', 'DIFF_NS'):
            col_map['REFSYS_ns'] = col

    missing = [k for k in ('MJD', 'STTIME', 'REFSYS_ns') if k not in col_map]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

    out = pd.DataFrame({
        'MJD':       pd.to_numeric(df[col_map['MJD']],      errors='coerce'),
        'STTIME':    pd.to_numeric(df[col_map['STTIME']],   errors='coerce'),
        'REFSYS_ns': pd.to_numeric(df[col_map['REFSYS_ns']], errors='coerce'),
    }).dropna()
    out['MJD']    = out['MJD'].astype(int)
    out['STTIME'] = out['STTIME'].astype(int)
    out['CONST']  = 'GPS'
    out['PRN']    = 'N/A'
    out['ELV']    = 45.0
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 2-sigma clipping
# ──────────────────────────────────────────────────────────────────────────────

def sigma_clip(series: pd.Series, sigma: float = 2.0) -> tuple[pd.Series, dict]:
    """Apply iterative 2σ clipping. Returns (mask_keep, report_dict)."""
    mask = pd.Series([True] * len(series), index=series.index)
    total = len(series)
    for _ in range(10):   # max 10 iterations (converges in ~3)
        subset = series[mask]
        if len(subset) < 3:
            break
        m, s = subset.mean(), subset.std(ddof=1)
        new_mask = mask & (series >= m - sigma * s) & (series <= m + sigma * s)
        if new_mask.equals(mask):
            break
        mask = new_mask

    retained = int(mask.sum())
    removed  = total - retained
    pct      = round(retained / total * 100, 1) if total else 0.0
    report   = {
        'total_points':       total,
        'retained_points':    retained,
        'removed_points':     removed,
        'retention_percentage': pct,
    }
    return mask, report


# ──────────────────────────────────────────────────────────────────────────────
# Core calibration engine
# ──────────────────────────────────────────────────────────────────────────────

def _epoch_key(mjd: int, sttime: int) -> tuple[int, int]:
    return (mjd, sttime)


def run_aiv(nmc_df: pd.DataFrame, cust_df: pd.DataFrame,
            sigma_filter: bool, sigma: float) -> tuple[list[dict], dict, dict]:
    """
    All-in-View: average all REFSYS values per (MJD, STTIME) epoch for each lab,
    then compute difference.
    Returns (epochs_list, filter_report_nmc, filter_report_cust).
    """
    def epoch_avg(df, do_clip, sigma_val):
        results = {}
        filter_totals = {'total_points': 0, 'retained_points': 0,
                         'removed_points': 0, 'retention_percentage': 0.0}
        for key, grp in df.groupby(['MJD', 'STTIME']):
            vals = grp['REFSYS_ns']
            filter_totals['total_points'] += len(vals)
            if do_clip and len(vals) >= 3:
                mask, _ = sigma_clip(vals, sigma_val)
                vals = vals[mask]
            filter_totals['retained_points'] += len(vals)
            if len(vals) == 0:
                continue
            results[key] = {
                'mean': vals.mean(),
                'n':    len(vals),
            }
        filter_totals['removed_points'] = (filter_totals['total_points'] -
                                            filter_totals['retained_points'])
        tot = filter_totals['total_points']
        filter_totals['retention_percentage'] = round(
            filter_totals['retained_points'] / tot * 100, 1) if tot else 0.0
        return results, filter_totals

    nmc_avg, fr_nmc  = epoch_avg(nmc_df,  sigma_filter, sigma)
    cust_avg, fr_cust = epoch_avg(cust_df, sigma_filter, sigma)

    common_keys = sorted(set(nmc_avg.keys()) & set(cust_avg.keys()))
    epochs = []
    for key in common_keys:
        mjd, sttime = key
        diff = cust_avg[key]['mean'] - nmc_avg[key]['mean']
        epochs.append({
            'MJD':      int(mjd),
            'STTIME':   int(sttime),
            'DATE':     mjd_to_date(mjd),
            'EPOCH':    sttime_to_hms(sttime),
            'NMC_ns':   round(float(nmc_avg[key]['mean']), 4),
            'CUST_ns':  round(float(cust_avg[key]['mean']), 4),
            'DIFF_ns':  round(float(diff), 4),
            'N_SATS':   int(max(nmc_avg[key]['n'], cust_avg[key]['n'])),
        })

    return epochs, fr_nmc, fr_cust


def run_cv(nmc_df: pd.DataFrame, cust_df: pd.DataFrame,
           sigma_filter: bool, sigma: float) -> tuple[list[dict], dict, dict]:
    """
    Common View: for each (MJD, STTIME, PRN) triplet visible in BOTH labs,
    compute per-satellite difference, then average over common satellites per epoch.
    """
    # Merge on MJD + STTIME + PRN
    merged = pd.merge(
        nmc_df[['MJD', 'STTIME', 'PRN', 'REFSYS_ns']].rename(columns={'REFSYS_ns': 'NMC_ns'}),
        cust_df[['MJD', 'STTIME', 'PRN', 'REFSYS_ns']].rename(columns={'REFSYS_ns': 'CUST_ns'}),
        on=['MJD', 'STTIME', 'PRN'],
        how='inner',
    )
    if merged.empty:
        raise ValueError(
            "Common View: no common satellites found between NMC and customer data. "
            "Check that file PRN labels (G01, E05…) are consistent and constellations match.")

    merged['DIFF_ns'] = merged['CUST_ns'] - merged['NMC_ns']

    filter_totals_nmc  = {'total_points': len(merged), 'retained_points': len(merged),
                           'removed_points': 0, 'retention_percentage': 100.0}
    filter_totals_cust = dict(filter_totals_nmc)

    epochs = []
    for key, grp in merged.groupby(['MJD', 'STTIME']):
        mjd, sttime = key
        diffs = grp['DIFF_ns']
        if sigma_filter and len(diffs) >= 3:
            mask, _ = sigma_clip(diffs)
            grp = grp[mask]
            diffs = grp['DIFF_ns']
        if len(diffs) == 0:
            continue
        mean_diff = diffs.mean()
        epochs.append({
            'MJD':     int(mjd),
            'STTIME':  int(sttime),
            'DATE':    mjd_to_date(mjd),
            'EPOCH':   sttime_to_hms(sttime),
            'NMC_ns':  round(float(grp['NMC_ns'].mean()), 4),
            'CUST_ns': round(float(grp['CUST_ns'].mean()), 4),
            'DIFF_ns': round(float(mean_diff), 4),
            'N_SATS':  int(len(grp)),
        })

    epochs.sort(key=lambda e: (e['MJD'], e['STTIME']))
    return epochs, filter_totals_nmc, filter_totals_cust


def build_summary(epochs: list[dict], mode: str, sigma_filter: bool) -> dict:
    if not epochs:
        return {}
    diffs   = [e['DIFF_ns'] for e in epochs]
    mjds    = [e['MJD']     for e in epochs]
    arr     = np.array(diffs)
    return {
        'mode':           mode,
        'sigma_filter':   sigma_filter,
        'n_epochs':       int(len(epochs)),
        'n_days':         int(len(set(mjds))),
        'mjd_start':      int(min(mjds)),
        'mjd_end':        int(max(mjds)),
        'date_start':     mjd_to_date(min(mjds)),
        'date_end':       mjd_to_date(max(mjds)),
        'mean_diff_ns':   float(arr.mean()),
        'std_diff_ns':    float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        'peak_diff_ns':   float(np.abs(arr).max()),
        'median_diff_ns': float(np.median(arr)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Flask routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    try:
        mode          = request.form.get('mode', 'AIV').upper()
        sigma_filter  = request.form.get('sigma_filter', 'true').lower() == 'true'
        sigma_val     = float(request.form.get('sigma', '2.0'))
        consts_raw    = request.form.get('constellations', '["GPS"]')
        try:
            allowed_consts = json.loads(consts_raw)
        except Exception:
            allowed_consts = ['GPS']

        nmc_files  = request.files.getlist('nmc_files[]')
        cust_files = request.files.getlist('cust_files[]')

        if not nmc_files or not cust_files:
            return jsonify({'error': 'Both NMC and customer files are required.'}), 400

        def load_files(file_list, label, allowed_consts):
            frames = []
            for f in file_list:
                raw = f.read().decode('utf-8', errors='replace')
                fname = f.filename.lower()
                # Detect file type: CSV vs CGGTTS
                if fname.endswith('.csv'):
                    df = parse_csv_refsys(raw)
                else:
                    df = parse_cggtts(raw, allowed_consts)
                if df.empty:
                    app.logger.warning(f"[{label}] No valid data from {f.filename}")
                    continue
                frames.append(df)
                app.logger.info(f"[{label}] {f.filename}: {len(df)} rows")
            if not frames:
                raise ValueError(
                    f"No valid {label} data found. Check that constellation selection matches "
                    f"file content (G=GPS, E=Galileo, R=GLONASS, C=BeiDou).")
            return pd.concat(frames, ignore_index=True)

        nmc_df  = load_files(nmc_files,  'NMC',      allowed_consts)
        cust_df = load_files(cust_files, 'Customer', allowed_consts)

        # Sort
        nmc_df  = nmc_df.sort_values(['MJD', 'STTIME']).reset_index(drop=True)
        cust_df = cust_df.sort_values(['MJD', 'STTIME']).reset_index(drop=True)

        if mode == 'CV':
            epochs, fr_nmc, fr_cust = run_cv(nmc_df, cust_df, sigma_filter, sigma_val)
        else:
            epochs, fr_nmc, fr_cust = run_aiv(nmc_df, cust_df, sigma_filter, sigma_val)

        if not epochs:
            return jsonify({'error':
                'No matched epochs found. Verify MJD ranges overlap between NMC and customer files.'}), 400

        summary = build_summary(epochs, mode, sigma_filter)
        return jsonify({
            'epochs':            epochs,
            'summary':           summary,
            'filter_report_nmc':  fr_nmc,
            'filter_report_cust': fr_cust,
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Internal error: {e}'}), 500


@app.route('/api/mdev', methods=['POST'])
def mdev_endpoint():
    try:
        body       = request.get_json(force=True)
        diff_ns    = body.get('diff_ns', [])
        epoch_sec  = float(body.get('epoch_sec', 780))

        if len(diff_ns) < 4:
            return jsonify({'error': 'Need at least 4 epochs to compute MDEV.'}), 400

        # Convert ns → s for frequency deviation (phase data)
        phase_s = np.array(diff_ns, dtype=float) * 1e-9

        if HAS_ALLANTOOLS:
            try:
                tau_out, mdev_out, _, _ = allantools.mdev(
                    phase_s,
                    rate=1.0 / epoch_sec,
                    data_type='phase',
                    taus='decade',
                )
                method = 'allantools.mdev (phase data, decade taus)'
            except Exception as ae:
                app.logger.warning(f"allantools failed: {ae}; falling back to manual MDEV")
                tau_out, mdev_out = _manual_mdev(phase_s, epoch_sec)
                method = 'manual MDEV (allantools fallback)'
        else:
            tau_out, mdev_out = _manual_mdev(phase_s, epoch_sec)
            method = 'manual MDEV (allantools not installed)'

        # Filter valid values
        valid = [(t, m) for t, m in zip(tau_out, mdev_out)
                 if np.isfinite(t) and np.isfinite(m) and m > 0]
        if not valid:
            return jsonify({'error': 'MDEV computation produced no valid points.'}), 400

        taus, mdevs = zip(*valid)

        # Build summary table at decade points
        summary_rows = []
        for t, m in zip(taus, mdevs):
            if t >= 900:
                label = ''
                if   t < 3600:   label = 'sub-hour'
                elif t < 86400:  label = 'sub-day'
                else:            label = 'multi-day'
                summary_rows.append({'tau': round(t, 1), 'mdev': m, 'label': label})

        return jsonify({
            'tau':     list(taus),
            'mdev':    list(mdevs),
            'method':  method,
            'summary': summary_rows,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'MDEV error: {e}'}), 500


def _manual_mdev(phase_s: np.ndarray, tau0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Manual Modified Allan Deviation computation (NIST algorithm).
    phase_s: phase data in seconds.
    tau0:    sample interval in seconds.
    Returns (tau_array, mdev_array).
    """
    N = len(phase_s)
    taus, mdevs = [], []
    m = 1
    while m <= N // 3:
        tau = m * tau0
        sums = 0.0
        count = 0
        for i in range(N - 3 * m + 1):
            inner = 0.0
            for j in range(m):
                inner += phase_s[i + 2*m + j] - 2 * phase_s[i + m + j] + phase_s[i + j]
            sums += inner ** 2
            count += 1
        if count > 0:
            mdev_val = math.sqrt(sums / (2.0 * count * m**2 * tau**2))
            taus.append(tau)
            mdevs.append(mdev_val)
        m = max(m + 1, int(m * 10 ** 0.25))   # log-spaced steps

    return np.array(taus), np.array(mdevs)


# ──────────────────────────────────────────────────────────────────────────────
# Static file serving (serve the HTML from same directory)
# ──────────────────────────────────────────────────────────────────────────────

from flask import send_from_directory

@app.route('/')
def index():
    here = Path(__file__).parent
    html_candidates = [
        'NMC_Remote_Calibration_System.html',
        'graphic user interface.html',
    ]
    for name in html_candidates:
        f = here / name
        if f.exists():
            return send_from_directory(str(here), name)
    return '<h2>NMC Remote Calibration System</h2><p>HTML file not found in this directory.</p>', 404


@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(str(Path(__file__).parent), filename)


if __name__ == '__main__':
    print("=" * 60)
    print("NMC Remote Calibration System — Backend Server")
    print("National Metrology Centre, Singapore")
    print("=" * 60)
    print(f"allantools available: {HAS_ALLANTOOLS}")
    print("Open http://localhost:5000 in your browser.")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
