"""Shared helpers for Sphynx-13 EEG ISC v2 (ICA + FASTER cleaning)."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

EPOCH_SEC = 18.0
ISC_SFREQ = 25.0
# Anti-alias cutoff before downsampling to ISC_SFREQ (Nyquist = 12.5 Hz).
# Keep a small margin below Nyquist; content above this is removed, not folded.
ISC_ANTIALIAS_HZ = 12.0
NON_EEG_CHS = {"PPG", "GSR"}
OCCIPITAL_ROI = ["O1", "Oz", "O2"]
MIN_SUBJECTS_ISC = 2

DEFAULT_DATA_DIR = Path("All data/experiment_for_Sphynx_pilot/exp data")
V1_RESULTS_DIR = Path("All data/experiment_for_Sphynx_pilot/analysis_sphynx13_eeg_et")


def map_adaptation(visualisation_type: str) -> str:
    vt = str(visualisation_type).strip().lower()
    if vt in ("non_adapted", "fully_adapted"):
        return vt
    if "non_adapted" in vt:
        return "non_adapted"
    if "fully_adapted" in vt:
        return "fully_adapted"
    return "semi_adapted"


def find_subdir(folder: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        p = folder / name
        if p.is_dir():
            return p
    lower_map = {c.name.lower(): c for c in folder.iterdir() if c.is_dir()}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def find_vhdr(eeg_dir: Path) -> Path | None:
    vhdrs = sorted(eeg_dir.glob("*.vhdr"))
    return vhdrs[0] if vhdrs else None


def find_practice_csv(folder: Path) -> Path | None:
    found = sorted(folder.glob("*Practice Trials*.csv"))
    return found[0] if found else None


def discover_participants(data_dir: Path) -> list[dict]:
    rows = []
    for folder in sorted([p for p in data_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        practice = find_practice_csv(folder)
        eeg_dir = find_subdir(folder, ("eeg", "EEG"))
        vhdr = find_vhdr(eeg_dir) if eeg_dir else None
        rows.append(
            {
                "folder_name": folder.name,
                "folder": folder,
                "practice_csv": practice,
                "vhdr": vhdr,
            }
        )
    return rows


def load_psychopy_trials(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    trials = df[df["graph_complexity"].notna()].copy().reset_index(drop=True)
    trials["graph_nodes"] = trials["graph_complexity"].astype(int)
    trials["question_level"] = trials["question_complexity"].astype(int)
    trials["adaptation"] = trials["visualisation_type"].map(map_adaptation)
    trials["graph_image"] = trials["graph_image"].astype(str).str.strip()
    trials["trial_order"] = np.arange(len(trials))
    if "participant" in trials.columns and trials["participant"].notna().any():
        trials["participant_id"] = trials["participant"].ffill().bfill().astype(str)
    else:
        trials["participant_id"] = csv_path.stem
    return trials


def normalize_stim_code(desc: str) -> str | None:
    s = str(desc).strip().upper()
    m = re.search(r"S\s*(\d+)", s)
    if not m:
        return None
    return f"S{int(m.group(1))}"


def fisher_z(r: float) -> float:
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r))


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return np.nan
    if np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return np.nan
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    r, _ = stats.pearsonr(a[mask], b[mask])
    return float(r)


def resample_1d(x: np.ndarray, n_out: int) -> np.ndarray:
    """Linear interpolation to fixed length (no anti-alias). Prefer resample_1d_antialias."""
    if len(x) == n_out:
        return x.astype(float)
    if len(x) < 2:
        return np.full(n_out, np.nan)
    old_t = np.linspace(0.0, 1.0, len(x))
    new_t = np.linspace(0.0, 1.0, n_out)
    return np.interp(new_t, old_t, x.astype(float))


def lowpass_1d(x: np.ndarray, sfreq: float, cutoff_hz: float) -> np.ndarray:
    """Zero-phase Butterworth lowpass for anti-aliasing before downsampling."""
    from scipy.signal import butter, filtfilt

    x = np.asarray(x, dtype=float)
    if len(x) < 16 or not np.isfinite(sfreq) or sfreq <= 0:
        return x
    nyq = 0.5 * float(sfreq)
    if cutoff_hz >= nyq:
        return x
    mask = np.isfinite(x)
    if mask.sum() < 16:
        return x
    filled = x.copy()
    if not mask.all():
        idx = np.arange(len(x))
        filled[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
    wn = float(cutoff_hz) / nyq
    b, a = butter(4, wn, btype="low")
    filtered = filtfilt(b, a, filled)
    out = filtered
    out[~mask] = np.nan
    return out


def resample_1d_antialias(
    x: np.ndarray,
    n_out: int,
    sfreq_in: float,
    cutoff_hz: float = ISC_ANTIALIAS_HZ,
) -> np.ndarray:
    """
    Anti-alias lowpass then resample to n_out samples.

    Required when sfreq_in/2 > ISC Nyquist (ISC_SFREQ/2). Without this step,
    energy between ISC_SFREQ/2 and the EEG bandpass upper edge aliases into ISC.
    """
    x_lp = lowpass_1d(x, sfreq_in, cutoff_hz)
    return resample_1d(x_lp, n_out)


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    mask = np.isfinite(p)
    n = int(mask.sum())
    if n == 0:
        return out
    pv = p[mask]
    order = np.argsort(pv)
    ranked = pv[order]
    adj = np.empty_like(ranked)
    for i, val in enumerate(ranked):
        adj[i] = val * n / (i + 1)
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    adj = np.clip(adj, 0.0, 1.0)
    restored = np.empty_like(adj)
    restored[order] = adj
    out[mask] = restored
    return out


def apply_pvalue_corrections(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["p_fdr_factor"] = np.nan
    df["p_fdr_global"] = np.nan
    df["p_bonferroni_factor"] = np.nan
    df["p_bonferroni_global"] = np.nan

    eligible = df["p_raw"].notna()
    n_global = int(eligible.sum())
    if n_global:
        raw = df.loc[eligible, "p_raw"].to_numpy()
        df.loc[eligible, "p_fdr_global"] = benjamini_hochberg(raw)
        df.loc[eligible, "p_bonferroni_global"] = np.minimum(raw * n_global, 1.0)

    for factor, fam in df.groupby("condition_factor"):
        fam_idx = fam.index[eligible.loc[fam.index]]
        n_fam = len(fam_idx)
        if not n_fam:
            continue
        raw = df.loc[fam_idx, "p_raw"].to_numpy()
        df.loc[fam_idx, "p_fdr_factor"] = benjamini_hochberg(raw)
        df.loc[fam_idx, "p_bonferroni_factor"] = np.minimum(raw * n_fam, 1.0)

    df["sig_raw_0.05"] = df["p_raw"].lt(0.05)
    df["sig_fdr_factor"] = df["p_fdr_factor"].lt(0.05)
    df["sig_fdr_global"] = df["p_fdr_global"].lt(0.05)
    df["sig_bonferroni_factor"] = df["p_bonferroni_factor"].lt(0.05)
    df["sig_bonferroni_global"] = df["p_bonferroni_global"].lt(0.05)
    return df


def condition_keys() -> list[tuple[str, callable]]:
    return [
        ("adaptation", lambda r: r["adaptation"]),
        ("graph_nodes", lambda r: str(int(r["graph_nodes"]))),
        ("question_level", lambda r: str(int(r["question_level"]))),
        ("adapt_x_q", lambda r: f"{r['adaptation']}_q{int(r['question_level'])}"),
        ("adapt_x_nodes", lambda r: f"{r['adaptation']}_n{int(r['graph_nodes'])}"),
    ]


def average_timecourses(
    records: list[dict],
    signal_key: str,
    cond_fn,
) -> dict[str, dict[str, np.ndarray]]:
    buckets: dict[str, dict[str, list]] = {}
    for r in records:
        if not r.get("qc_pass", False):
            continue
        lab = cond_fn(r)
        pid = r["participant_id"]
        buckets.setdefault(lab, {}).setdefault(pid, []).append(r[signal_key])
    out: dict[str, dict[str, np.ndarray]] = {}
    for lab, by_p in buckets.items():
        out[lab] = {}
        for pid, arrs in by_p.items():
            stack = np.vstack(arrs)
            out[lab][pid] = np.nanmean(stack, axis=0)
    return out


def leave_one_out_isc(subj_ts: dict[str, np.ndarray]) -> tuple[float, dict[str, float], int]:
    pids = [p for p, ts in subj_ts.items() if np.isfinite(ts).sum() > 10]
    if len(pids) < MIN_SUBJECTS_ISC:
        return np.nan, {}, len(pids)
    rs = {}
    zs = []
    for p in pids:
        others = [subj_ts[q] for q in pids if q != p]
        mean_others = np.nanmean(np.vstack(others), axis=0)
        r = corr_safe(subj_ts[p], mean_others)
        rs[p] = r
        if np.isfinite(r):
            zs.append(fisher_z(r))
    if not zs:
        return np.nan, rs, len(pids)
    mean_z = float(np.mean(zs))
    mean_r = float(np.tanh(mean_z))
    return mean_r, rs, len(pids)


def contrast_pairs() -> dict[str, list[tuple[str, str]]]:
    return {
        "adaptation": [
            ("non_adapted", "fully_adapted"),
            ("non_adapted", "semi_adapted"),
            ("semi_adapted", "fully_adapted"),
        ],
        "graph_nodes": [("6", "12")],
        "question_level": [("1", "2")],
    }


def permutation_p_one_sided(null: np.ndarray, observed: float) -> float:
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(observed):
        return np.nan
    return float((np.sum(null >= observed) + 1) / (len(null) + 1))


def permutation_p_two_sided(null: np.ndarray, observed: float) -> float:
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(observed):
        return np.nan
    return float((np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1))


def circular_shift_isc_null(
    subj_ts: dict[str, np.ndarray],
    n_perm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Null distribution of mean LOO ISC after independent circular shifts per subject."""
    pids = [p for p, ts in subj_ts.items() if np.isfinite(ts).sum() > 10]
    if len(pids) < MIN_SUBJECTS_ISC:
        return np.array([])
    null_vals = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        shifted: dict[str, np.ndarray] = {}
        for p in pids:
            ts = np.asarray(subj_ts[p], dtype=float)
            n = len(ts)
            shift = int(rng.integers(0, n)) if n > 0 else 0
            shifted[p] = np.roll(ts, shift)
        mean_r, _, _ = leave_one_out_isc(shifted)
        null_vals[i] = mean_r
    return null_vals


def sign_flip_contrast_null(
    diff: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Null for paired mean Fisher-z difference via random sign flips."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return np.array([])
    null_vals = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(diff))
        null_vals[i] = float(np.mean(diff * signs))
    return null_vals

