"""
NMC Remote Calibration System — Flask Backend
National Metrology Centre, Singapore

Endpoints:
  POST /api/calibrate   — parse CGGTTS/CSV files, compute time differences
  POST /api/mdev        — compute Modified Allan Deviation

CGGTTS column layout (fixed-width, ITU-R TF.1153):
  PRN  CL  MJD    STTIME  TRKL  ELV  AZTH  REFSV     SRSV   REFGPS    SRGPS   DSG  IOE  MDTR  SMDT  MDIO  SMDI  CK
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

def _detect_version(header: str) -> str:
    """Return '1E' or '2E' based on header content."""
    if 'CGGTTS' in header and ('V2E' in header or 'version 2E' in header.lower()):
        return '2E'
    return '1E'


def parse_cggtts(content: str, allowed_constellations: list[str]) -> pd.DataFrame:
    """
    Parse a CGGTTS file (version 1E or 2E) and return a DataFrame with columns:
        PRN, CONST, MJD, STTIME, REFSYS_ns, ELV
    REFSYS is converted from 0.1 ns units → ns (divide by 10).
    """
    lines = content.splitlines()
    # Find the blank line that separates header from data
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Data starts after the blank line following the header
        if stripped == '' and i > 0:
            data_start = i + 1
            break

    header = '\n'.join(lines[:data_start])
    version = _detect_version(header)

    records = []
    for raw in lines[data_start:]:
        line = raw.strip()
        if not line or line.startswith('#'):
            continue

        tokens = line.split()
        if len(tokens) < 13:
            continue  # malformed

        try:
            # CGGTTS V2E: first token is PRN like "G01", "E05", etc.
            # CGGTTS V1E: first token may be numeric PRN (GPS only)
            prn_field = tokens[0]
            if re.match(r'^[A-Z]\d+$', prn_field):
                const_char = prn_field[0]
                const_name = CONST_CHAR.get(const_char, 'GPS')
            else:
                const_char = 'G'
                const_name = 'GPS'

            if const_name not in allowed_constellations:
                continue

            # V2E: PRN CL MJD STTIME TRKL ELV AZTH REFSV SRSV REFGPS SRGPS DSG IOE MDTR SMDT MDIO SMDI CK
            # V1E: PRN MJD STTIME ...  (no CL column)
            if version == '2E' and len(tokens) >= 17:
                mjd     = int(tokens[2])
                sttime  = int(tokens[3])
                elv     = int(tokens[5])   # 0.1 deg units
                refsys  = int(tokens[7])   # 0.1 ns → ns
            else:
                # Try both layouts gracefully
                mjd     = int(tokens[1])
                sttime  = int(tokens[2])
                elv     = int(tokens[4]) if len(tokens) > 4 else 100
                refsys  = int(tokens[6]) if len(tokens) > 6 else 0

            elv_deg = elv / 10.0
            if elv_deg < 10.0:          # elevation mask 10°
                continue
            if refsys == 9999999 or abs(refsys) > 9000000:   # sentinel / invalid
                continue

            records.append({
                'PRN':      prn_field,
                'CONST':    const_name,
                'MJD':      mjd,
                'STTIME':   sttime,
                'REFSYS_ns': refsys / 10.0,   # convert 0.1 ns → ns
                'ELV':      elv_deg,
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
            sigma_filter: bool, sigma: float) -> tuple[list[dict], dict, dict, dict]:
    """
    All-in-View: average all REFSYS values per (MJD, STTIME) epoch for each lab,
    then compute difference.
    Returns (epochs_list, summary, filter_report_nmc, filter_report_cust).
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
            'MJD':      mjd,
            'STTIME':   sttime,
            'DATE':     mjd_to_date(mjd),
            'EPOCH':    sttime_to_hms(sttime),
            'NMC_ns':   round(nmc_avg[key]['mean'], 4),
            'CUST_ns':  round(cust_avg[key]['mean'], 4),
            'DIFF_ns':  round(diff, 4),
            'N_SATS':   max(nmc_avg[key]['n'], cust_avg[key]['n']),
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
            'MJD':     mjd,
            'STTIME':  sttime,
            'DATE':    mjd_to_date(mjd),
            'EPOCH':   sttime_to_hms(sttime),
            'NMC_ns':  round(grp['NMC_ns'].mean(), 4),
            'CUST_ns': round(grp['CUST_ns'].mean(), 4),
            'DIFF_ns': round(mean_diff, 4),
            'N_SATS':  len(grp),
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
        'n_epochs':       len(epochs),
        'n_days':         len(set(mjds)),
        'mjd_start':      min(mjds),
        'mjd_end':        max(mjds),
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
