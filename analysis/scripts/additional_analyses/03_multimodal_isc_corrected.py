#!/usr/bin/env python3
"""
FDR + permutation correction for ET/EDA/pupil ISC contrasts.

Addresses caveat: multimodal ISC claims (gaze Y, EDA) rest on v1 raw
Wilcoxon with no correction. Re-extracts per-subject LOO ISC from v1
ET/physio epochs, applies FDR-BH + sign-flip permutation.

  python 03_multimodal_isc_corrected.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V1_SCRIPT_DIR = ROOT
EEG_ISC_V2_DIR = ROOT / "All data/post_analysis/eeg_isc_v2"
OUT_DIR = HERE / "results"

sys.path.insert(0, str(V1_SCRIPT_DIR))
sys.path.insert(0, str(EEG_ISC_V2_DIR))

from pilot_eeg_et_isc_sphynx13 import (
    DEFAULT_DATA_DIR,
    discover_participants,
    load_psychopy_trials,
    load_eeg_epochs,
    load_et_epochs,
    condition_keys,
    average_timecourses,
    leave_one_out_isc,
)
from sphynx13_eeg_utils import (
    fisher_z,
    apply_pvalue_corrections,
    benjamini_hochberg,
    sign_flip_contrast_null,
    permutation_p_one_sided,
    permutation_p_two_sided,
)

N_PERM = 5000
SEED = 42
MIN_SUBJECTS = 2

CONTRAST_PAIRS = {
    "adaptation": [
        ("non_adapted", "fully_adapted"),
        ("non_adapted", "semi_adapted"),
        ("semi_adapted", "fully_adapted"),
    ],
    "graph_nodes": [("6", "12")],
    "question_level": [("1", "2")],
}

ET_SIGNALS = ["gx_isc", "gy_isc", "pupil_isc"]
PHYSIO_SIGNALS = ["eda_isc", "ppg_isc"]
MAIN_FACTORS = ["adaptation", "graph_nodes", "question_level"]


def load_all_epochs(data_dir: Path) -> tuple[list[dict], list[dict]]:
    """Re-run v1 epoch extraction for ET and physio (EDA/PPG from EEG aux)."""
    participants = discover_participants(data_dir)
    eeg_recs = []
    et_recs = []
    notes = []

    for p in participants:
        folder_name = p["folder_name"]
        if p["practice_csv"] is None:
            continue
        psy = load_psychopy_trials(p["practice_csv"])
        pid = str(psy["participant_id"].iloc[0])

        if p["vhdr"] is not None:
            recs, n = load_eeg_epochs(p["vhdr"], psy, folder_name)
            eeg_recs.extend(recs)
            notes.extend(n)

        if p["gaze_csv"] is not None:
            recs, n = load_et_epochs(
                p["gaze_csv"], p["fixations_csv"], folder_name, participant_id=pid
            )
            et_recs.extend(recs)
            notes.extend(n)

    return eeg_recs, et_recs


def compute_subject_loo(
    records: list[dict],
    signal_keys: list[str],
    modality_label: str,
) -> pd.DataFrame:
    rows = []
    for signal_key in signal_keys:
        for cond_name, cond_fn in condition_keys():
            if cond_name not in MAIN_FACTORS:
                continue
            by_cond = average_timecourses(records, signal_key, cond_name, cond_fn)
            for lab, subj_ts in by_cond.items():
                mean_r, per_subj, n = leave_one_out_isc(subj_ts)
                for pid, r in per_subj.items():
                    rows.append({
                        "modality": modality_label,
                        "signal": signal_key,
                        "condition_factor": cond_name,
                        "condition_level": lab,
                        "participant_id": pid,
                        "loo_r": r,
                        "loo_fisher_z": fisher_z(r) if np.isfinite(r) else np.nan,
                    })
    return pd.DataFrame(rows)


def run_contrasts_with_permutation(
    subj_df: pd.DataFrame,
    n_perm: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for (modality, signal, factor), g in subj_df.groupby(["modality", "signal", "condition_factor"]):
        if factor not in CONTRAST_PAIRS:
            continue
        for a, b in CONTRAST_PAIRS[factor]:
            ga = g[g["condition_level"] == a][["participant_id", "loo_fisher_z"]].rename(
                columns={"loo_fisher_z": "z_a"}
            )
            gb = g[g["condition_level"] == b][["participant_id", "loo_fisher_z"]].rename(
                columns={"loo_fisher_z": "z_b"}
            )
            m = ga.merge(gb, on="participant_id").dropna()
            if len(m) < 3:
                continue
            diff = (m["z_a"] - m["z_b"]).to_numpy()
            obs_diff = float(np.mean(diff))

            try:
                w_stat, w_p = stats.wilcoxon(m["z_a"], m["z_b"])
            except ValueError:
                w_stat, w_p = np.nan, np.nan

            null = sign_flip_contrast_null(diff, n_perm, rng)
            p_perm = permutation_p_two_sided(null, obs_diff)

            rows.append({
                "modality": modality,
                "signal": signal,
                "condition_factor": factor,
                "level_a": a,
                "level_b": b,
                "n_paired": len(m),
                "mean_r_a": float(np.tanh(m["z_a"].mean())),
                "mean_r_b": float(np.tanh(m["z_b"].mean())),
                "mean_z_diff": obs_diff,
                "wilcoxon_stat": float(w_stat),
                "p_wilcoxon": float(w_p),
                "p_perm": p_perm,
                "p_raw": p_perm,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return apply_pvalue_corrections(df)


def write_report(path: Path, subj_df: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    lines = ["MULTIMODAL ISC — CORRECTED STATS", "=" * 50, ""]
    lines.append(f"Signals: {', '.join(ET_SIGNALS + PHYSIO_SIGNALS)}")
    lines.append(f"Permutations: {N_PERM}")
    lines.append("")

    if subj_df.empty:
        lines.append("No subject LOO data extracted.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("ISC LEVELS BY ADAPTATION (mean LOO r)")
    for signal in ET_SIGNALS + PHYSIO_SIGNALS:
        sub = subj_df[subj_df["signal"] == signal]
        if sub.empty:
            continue
        lines.append(f"  {signal}:")
        for level in ["non_adapted", "semi_adapted", "fully_adapted"]:
            lev_sub = sub[(sub["condition_factor"] == "adaptation") & (sub["condition_level"] == level)]
            if lev_sub.empty:
                continue
            mean_r = float(np.tanh(lev_sub["loo_fisher_z"].mean()))
            lines.append(f"    {level:15s} r={mean_r:.3f} n={len(lev_sub)}")
    lines.append("")

    if contrasts.empty:
        lines.append("No contrasts computed.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("CONTRASTS — raw p_perm < .05")
    raw_sig = contrasts[contrasts["sig_raw_0.05"] == True]  # noqa: E712
    if raw_sig.empty:
        lines.append("  none")
    else:
        for _, r in raw_sig.sort_values("p_raw").iterrows():
            lines.append(
                f"  {r['signal']} {r['condition_factor']} {r['level_a']} vs {r['level_b']}: "
                f"r={r['mean_r_a']:.3f} vs {r['mean_r_b']:.3f} "
                f"p_perm={r['p_raw']:.4g} p_wilcoxon={r['p_wilcoxon']:.4g}"
            )
    lines.append("")
    lines.append("CONTRASTS — FDR within factor")
    fdr_sig = contrasts[contrasts["sig_fdr_factor"] == True]  # noqa: E712
    if fdr_sig.empty:
        lines.append("  none")
    else:
        for _, r in fdr_sig.sort_values("p_fdr_factor").iterrows():
            lines.append(
                f"  {r['signal']} {r['condition_factor']} {r['level_a']} vs {r['level_b']}: "
                f"p_fdr={r['p_fdr_factor']:.4g}"
            )
    lines.append("")

    lines.append("COMPARISON TO V1 RAW WILCOXON")
    lines.append("- V1 had no correction; check whether raw-significant v1 contrasts survive FDR/permutation")
    lines.append("- Gaze Y and EDA were the strongest v1 multimodal signals")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    data_dir = ROOT / DEFAULT_DATA_DIR
    print(f"Loading epochs from {data_dir}...")
    eeg_recs, et_recs = load_all_epochs(data_dir)
    print(f"  EEG epochs (for physio): {len(eeg_recs)}")
    print(f"  ET epochs: {len(et_recs)}")

    print("Computing per-subject LOO ISC for ET signals...")
    et_loo = compute_subject_loo(et_recs, ET_SIGNALS, "et")
    print("Computing per-subject LOO ISC for physio signals...")
    physio_loo = compute_subject_loo(eeg_recs, PHYSIO_SIGNALS, "physio")

    subj_df = pd.concat([et_loo, physio_loo], ignore_index=True)
    subj_path = OUT_DIR / "multimodal_subject_loo.csv"
    subj_df.to_csv(subj_path, index=False)
    print(f"Wrote {subj_path} ({len(subj_df)} rows)")

    print(f"Running contrasts + {args.n_perm} permutations...")
    contrasts = run_contrasts_with_permutation(subj_df, args.n_perm, rng)
    con_path = OUT_DIR / "multimodal_isc_contrasts.csv"
    contrasts.to_csv(con_path, index=False)

    report_path = OUT_DIR / "multimodal_isc_report.txt"
    write_report(report_path, subj_df, contrasts)

    print(f"Wrote {con_path}")
    print(f"Wrote {report_path}")
    if not contrasts.empty:
        print(
            f"Raw sig: {int(contrasts['sig_raw_0.05'].sum())} | "
            f"FDR-factor: {int(contrasts['sig_fdr_factor'].sum())}"
        )


if __name__ == "__main__":
    main()
