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
# ──────────────────────────────────────────────────────────────────────────────
# CGGTTS parser (ROBUST VERSION)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# CGGTTS parser (fixed)
# ──────────────────────────────────────────────────────────────────────────────

CONST_CHAR = {
    'G': 'GPS',
    'E': 'Galileo',
    'R': 'GLONASS',
    'C': 'BeiDou',
    'J': 'QZSS'
}


def _detect_version(header: str) -> str:
    header_lower = header.lower()
    if "2e" in header_lower:
        return "2E"
    return "1E"


def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def parse_cggtts(content: str, allowed_constellations: list[str]) -> pd.DataFrame:
    """
    Robust CGGTTS parser for V1E / V2E (NMC format safe).

    Output columns:
        PRN, CONST, MJD, STTIME, REFSYS_ns, ELV
    """

    lines = content.splitlines()

    # ── find start of data block ──────────────────────────────────────────────
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "" and i > 5:
            data_start = i + 1
            break

    header = "\n".join(lines[:data_start])
    version = _detect_version(header)

    # ── detect column indices from header (if present) ────────────────────────
    refsys_idx = None
    sat_idx = 0

    for line in lines:
        if line.strip().startswith("SAT CL"):
            cols = line.split()
            if "REFSYS" in cols:
                refsys_idx = cols.index("REFSYS")
            if "SAT" in cols:
                sat_idx = cols.index("SAT")
            break

    records = []

    for raw in lines[data_start:]:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        tokens = line.split()
        if len(tokens) < 8:
            continue

        prn = tokens[sat_idx] if sat_idx < len(tokens) else tokens[0]

        # ── constellation detection ───────────────────────────────────────────
        const_char = prn[0] if prn and prn[0].isalpha() else "G"
        const_name = CONST_CHAR.get(const_char, "GPS")

        if allowed_constellations and const_name not in allowed_constellations:
            continue

        try:
            mjd = _safe_int(tokens[2])
            sttime = _safe_int(tokens[3])

            if mjd is None or sttime is None:
                continue

            # ── ELV extraction ────────────────────────────────────────────────
            elv_raw = _safe_int(tokens[5]) if len(tokens) > 5 else None
            if elv_raw is None:
                continue

            elv_deg = elv_raw / 10.0
            if elv_deg < 10.0:
                continue

            # ── REFSYS extraction (header-based if possible) ──────────────────
            refsys_raw = None

            if refsys_idx is not None and refsys_idx < len(tokens):
                refsys_raw = _safe_int(tokens[refsys_idx])

            # fallback (for safety across variants)
            if refsys_raw is None:
                for t in tokens:
                    val = _safe_int(t, None)
                    if val is not None and abs(val) < 5_000_000:
                        refsys_raw = val
                        break

            if refsys_raw is None or abs(refsys_raw) > 9_000_000:
                continue

            records.append({
                "PRN": prn,
                "CONST": const_name,
                "MJD": mjd,
                "STTIME": sttime,
                "REFSYS_ns": refsys_raw / 10.0,  # 0.1 ns → ns
                "ELV": elv_deg,
            })

        except Exception:
            continue

    return pd.DataFrame(records)
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

    def epoch_avg(df, do_clip, sigma_val):
        results = {}
        filter_totals = {
            'total_points': 0,
            'retained_points': 0,
            'removed_points': 0,
            'retention_percentage': 0.0
        }

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

        tot = filter_totals['total_points']
        filter_totals['removed_points'] = tot - filter_totals['retained_points']
        filter_totals['retention_percentage'] = round(
            filter_totals['retained_points'] / tot * 100, 1
        ) if tot else 0.0

        return results, filter_totals

    nmc_avg, fr_nmc   = epoch_avg(nmc_df, sigma_filter, sigma)
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

    merged = pd.merge(
        nmc_df[['MJD', 'STTIME', 'PRN', 'REFSYS_ns']]
        .rename(columns={'REFSYS_ns': 'NMC_ns'}),
        cust_df[['MJD', 'STTIME', 'PRN', 'REFSYS_ns']]
        .rename(columns={'REFSYS_ns': 'CUST_ns'}),
        on=['MJD', 'STTIME', 'PRN'],
        how='inner',
    )

    if merged.empty:
        raise ValueError(
            "Common View: no common satellites found between NMC and customer data. "
            "Check PRN formatting and constellation consistency."
        )

    merged['DIFF_ns'] = merged['CUST_ns'] - merged['NMC_ns']

    filter_totals_nmc  = {
        'total_points': len(merged),
        'retained_points': len(merged),
        'removed_points': 0,
        'retention_percentage': 100.0
    }
    filter_totals_cust = dict(filter_totals_nmc)

    epochs = []

    for key, grp in merged.groupby(['MJD', 'STTIME']):
        mjd, sttime = key
        diffs = grp['DIFF_ns']

        if sigma_filter and len(diffs) >= 3:
            mask, _ = sigma_clip(diffs.reset_index(drop=True))

            # 🔧 NECESSARY FIX: ensure safe alignment
            grp = grp.reset_index(drop=True)[mask.values]
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

    diffs = [e['DIFF_ns'] for e in epochs]
    mjds  = [e['MJD'] for e in epochs]

    arr = np.array(diffs, dtype=float)
    arr = arr[~np.isnan(arr)]  # 🔧 NECESSARY FIX

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

# ──────────────────────────────────────────────────────────────────────────────
# Flask routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    try:
        mode         = request.form.get('mode', 'AIV').upper()
        sigma_filter = request.form.get('sigma_filter', 'true').lower() == 'true'
        sigma_val    = float(request.form.get('sigma', '2.0'))

        consts_raw = request.form.get('constellations', '["GPS"]')

        try:
            allowed_consts = json.loads(consts_raw)
            if not isinstance(allowed_consts, list):
                allowed_consts = ['GPS']
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

                # Detect file type
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
                    f"No valid {label} data found. Check constellation filter and file format."
                )

            return pd.concat(frames, ignore_index=True)

        nmc_df  = load_files(nmc_files,  'NMC', allowed_consts)
        cust_df = load_files(cust_files, 'Customer', allowed_consts)

        nmc_df  = nmc_df.sort_values(['MJD', 'STTIME']).reset_index(drop=True)
        cust_df = cust_df.sort_values(['MJD', 'STTIME']).reset_index(drop=True)

        if mode == 'CV':
            epochs, fr_nmc, fr_cust = run_cv(nmc_df, cust_df, sigma_filter, sigma_val)
        else:
            epochs, fr_nmc, fr_cust = run_aiv(nmc_df, cust_df, sigma_filter, sigma_val)

        if not epochs:
            return jsonify({
                'error': 'No matched epochs found. Verify MJD overlap between datasets.'
            }), 400

        summary = build_summary(epochs, mode, sigma_filter)

        return jsonify({
            'epochs': epochs,
            'summary': summary,
            'filter_report_nmc': fr_nmc,
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
        body      = request.get_json(force=True)
        diff_ns   = body.get('diff_ns', [])
        epoch_sec = float(body.get('epoch_sec', 780))

        if len(diff_ns) < 4:
            return jsonify({'error': 'Need at least 4 epochs to compute MDEV.'}), 400

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
                app.logger.warning(f"allantools failed: {ae}; fallback used")
                tau_out, mdev_out = _manual_mdev(phase_s, epoch_sec)
                method = 'manual MDEV fallback'
        else:
            tau_out, mdev_out = _manual_mdev(phase_s, epoch_sec)
            method = 'manual MDEV (no allantools)'

        tau_out = np.array(tau_out)
        mdev_out = np.array(mdev_out)

        valid = [
            (t, m)
            for t, m in zip(tau_out, mdev_out)
            if np.isfinite(t) and np.isfinite(m) and m > 0
        ]

        if not valid:
            return jsonify({'error': 'MDEV produced no valid points.'}), 400

        taus, mdevs = zip(*valid)

        summary_rows = []
        for t, m in zip(taus, mdevs):
            if t >= 900:
                if t < 3600:
                    label = 'sub-hour'
                elif t < 86400:
                    label = 'sub-day'
                else:
                    label = 'multi-day'

                summary_rows.append({
                    'tau': round(t, 1),
                    'mdev': m,
                    'label': label
                })

        return jsonify({
            'tau': list(taus),
            'mdev': list(mdevs),
            'method': method,
            'summary': summary_rows,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'MDEV error: {e}'}), 500

def _manual_mdev(phase_s: np.ndarray, tau0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Manual Modified Allan Deviation computation (NIST-style).
    """

    N = len(phase_s)
    taus, mdevs = [], []

    m = 1
    while m <= N // 3:
        tau = m * tau0
        sums = 0.0
        count = 0

        max_i = N - 3 * m + 1
        if max_i <= 0:
            break

        for i in range(max_i):
            inner = 0.0

            # safety: avoid index overflow
            for j in range(m):
                idx1 = i + j
                idx2 = i + m + j
                idx3 = i + 2*m + j

                if idx3 >= N:
                    continue

                inner += phase_s[idx3] - 2 * phase_s[idx2] + phase_s[idx1]

            sums += inner ** 2
            count += 1

        if count > 0:
            mdev_val = math.sqrt(sums / (2.0 * count * m**2 * tau**2))
            taus.append(tau)
            mdevs.append(mdev_val)

        m = max(m + 1, int(m * 10 ** 0.25))

    return np.array(taus), np.array(mdevs)

# ──────────────────────────────────────────────────────────────────────────────
# Static file serving (serve the HTML from same directory)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Static file serving (serve the HTML from same directory)
# ──────────────────────────────────────────────────────────────────────────────

from flask import send_file

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
            return send_file(f)

    return (
        '<h2>NMC Remote Calibration System</h2>'
        '<p>HTML file not found in this directory.</p>'
    ), 404


# ⚠️ SECURITY FIX: restrict static serving to safe file types only
@app.route('/<path:filename>')
def static_files(filename):
    allowed_ext = {'.html', '.css', '.js', '.png', '.jpg', '.jpeg', '.svg'}

    file_path = Path(__file__).parent / filename

    if file_path.suffix.lower() not in allowed_ext:
        return "Forbidden", 403

    if not file_path.exists():
        return "Not found", 404

    return send_file(file_path)


if __name__ == '__main__':
    print("=" * 60)
    print("NMC Remote Calibration System — Backend Server")
    print("National Metrology Centre, Singapore")
    print("=" * 60)
    print(f"allantools available: {HAS_ALLANTOOLS}")
    print("Open http://localhost:5000 in your browser.")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
