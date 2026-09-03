#!/usr/bin/env python3
"""
Sphynx-13 EEG LOO timecourse ISC v2 with multiple-comparison correction.

Reads ICA+FASTER-cleaned trial GFP/occ timecourses from 01_preprocess_eeg_ica_faster.py,
computes leave-one-out ISC by condition (same logic as v1 Analysis A, EEG only),
applies Wilcoxon paired contrasts on Fisher-z LOO values, and corrects p-values
(FDR-BH within factor + global, Bonferroni within factor).

Also compares v2 vs v1 ISC levels and contrasts.

  python 02_run_eeg_isc_corrected.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sphynx13_eeg_utils import (
    V1_RESULTS_DIR,
    apply_pvalue_corrections,
    average_timecourses,
    condition_keys,
    contrast_pairs,
    fisher_z,
    leave_one_out_isc,
)

HERE = Path(__file__).resolve().parent
FOR_REPORT = HERE.parents[1]
REPO_ROOT = FOR_REPORT.parents[2]
PRE_DIR = FOR_REPORT / "results" / "eeg_isc_v2_antialias" / "preprocessed"
OUT_DIR = FOR_REPORT / "results" / "eeg_isc_v2_antialias"

EEG_SIGNALS = ["gfp_isc", "occ_isc"]


def load_preprocessed_records(pre_dir: Path) -> list[dict]:
    index_path = pre_dir / "trial_index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing {index_path}; run 01_preprocess_eeg_ica_faster.py first")
    index_df = pd.read_csv(index_path)
    records = []
    for _, row in index_df.iterrows():
        gfp = np.load(pre_dir / row["gfp_npy"])
        occ = np.load(pre_dir / row["occ_npy"])
        records.append(
            {
                "folder_name": row["folder_name"],
                "participant_id": str(row["participant_id"]),
                "modality": "eeg",
                "trial_order": int(row["trial_order"]),
                "graph_nodes": int(row["graph_nodes"]),
                "question_level": int(row["question_level"]),
                "adaptation": str(row["adaptation"]),
                "graph_image": str(row["graph_image"]),
                "qc_pass": bool(row["qc_pass"]),
                "gfp_isc": gfp,
                "occ_isc": occ,
            }
        )
    return records


def run_isc_analysis(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    isc_rows = []
    subj_rows = []
    for signal_key in EEG_SIGNALS:
        for cond_name, cond_fn in condition_keys():
            by_cond = average_timecourses(records, signal_key, cond_fn)
            for lab, subj_ts in by_cond.items():
                mean_r, per_subj, n = leave_one_out_isc(subj_ts)
                isc_rows.append(
                    {
                        "analysis": "A_timecourse_ISC_v2",
                        "modality": "eeg",
                        "signal": signal_key,
                        "condition_factor": cond_name,
                        "condition_level": lab,
                        "n_subjects": n,
                        "isc_mean_r": mean_r,
                        "isc_mean_fisher_z": fisher_z(mean_r) if np.isfinite(mean_r) else np.nan,
                        "pipeline": "ica_faster",
                    }
                )
                for pid, r in per_subj.items():
                    subj_rows.append(
                        {
                            "modality": "eeg",
                            "signal": signal_key,
                            "condition_factor": cond_name,
                            "condition_level": lab,
                            "participant_id": pid,
                            "loo_r": r,
                            "loo_fisher_z": fisher_z(r) if np.isfinite(r) else np.nan,
                            "pipeline": "ica_faster",
                        }
                    )
    return pd.DataFrame(isc_rows), pd.DataFrame(subj_rows)


def contrast_subject_isc(subj_df: pd.DataFrame) -> pd.DataFrame:
    if subj_df.empty:
        return pd.DataFrame()
    out = []
    pairs = contrast_pairs()
    for (signal, factor), g in subj_df.groupby(["signal", "condition_factor"]):
        if factor not in pairs:
            continue
        for a, b in pairs[factor]:
            ga = g[g["condition_level"] == a][["participant_id", "loo_fisher_z"]].rename(
                columns={"loo_fisher_z": "z_a"}
            )
            gb = g[g["condition_level"] == b][["participant_id", "loo_fisher_z"]].rename(
                columns={"loo_fisher_z": "z_b"}
            )
            m = ga.merge(gb, on="participant_id").dropna()
            if len(m) < 3:
                continue
            try:
                stat, p = stats.wilcoxon(m["z_a"], m["z_b"])
            except ValueError:
                continue
            out.append(
                {
                    "analysis": "A_ISC_contrast_v2",
                    "modality": "eeg",
                    "signal": signal,
                    "condition_factor": factor,
                    "level_a": a,
                    "level_b": b,
                    "n_paired": len(m),
                    "mean_z_a": float(m["z_a"].mean()),
                    "mean_z_b": float(m["z_b"].mean()),
                    "mean_r_a": float(np.tanh(m["z_a"].mean())),
                    "mean_r_b": float(np.tanh(m["z_b"].mean())),
                    "wilcoxon_stat": float(stat),
                    "p_raw": float(p),
                    "pipeline": "ica_faster",
                }
            )
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return apply_pvalue_corrections(df)


def compare_to_v1(v2_isc: pd.DataFrame, v2_contrasts: pd.DataFrame, v1_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    v1_isc_path = v1_dir / "analysisA_isc_timecourse.csv"
    v1_con_path = v1_dir / "analysisA_isc_contrasts.csv"
    level_rows = []
    contrast_rows = []

    if v1_isc_path.is_file():
        v1 = pd.read_csv(v1_isc_path)
        v1_eeg = v1[(v1["modality"] == "eeg") & (v1["signal"].isin(EEG_SIGNALS))]
        v2_eeg = v2_isc.copy()
        merge_keys = ["signal", "condition_factor", "condition_level"]
        m = v2_eeg.merge(
            v1_eeg[merge_keys + ["isc_mean_r", "n_subjects"]],
            on=merge_keys,
            how="outer",
            suffixes=("_v2", "_v1"),
        )
        m["delta_r"] = m["isc_mean_r_v2"] - m["isc_mean_r_v1"]
        level_rows = m.to_dict("records")

    if v1_con_path.is_file():
        v1c = pd.read_csv(v1_con_path)
        v1c_eeg = v1c[(v1c["modality"] == "eeg") & (v1c["signal"].isin(EEG_SIGNALS))]
        keys = ["signal", "condition_factor", "level_a", "level_b"]
        m2 = v2_contrasts.merge(
            v1c_eeg[keys + ["mean_r_a", "mean_r_b", "p_raw", "significant_0.05"]],
            on=keys,
            how="outer",
            suffixes=("_v2", "_v1"),
        )
        m2["delta_r_a"] = m2["mean_r_a_v2"] - m2["mean_r_a_v1"]
        m2["delta_r_b"] = m2["mean_r_b_v2"] - m2["mean_r_b_v1"]
        m2["sig_v1_raw"] = m2.get("significant_0.05", False)
        m2["sig_v2_raw"] = m2["sig_raw_0.05"]
        m2["sig_v2_fdr_factor"] = m2["sig_fdr_factor"]
        contrast_rows = m2.to_dict("records")

    return pd.DataFrame(level_rows), pd.DataFrame(contrast_rows)


def write_report(
    path: Path,
    records: list[dict],
    isc_df: pd.DataFrame,
    contrasts: pd.DataFrame,
    level_cmp: pd.DataFrame,
    contrast_cmp: pd.DataFrame,
    qc_path: Path,
) -> None:
    lines = []
    lines.append("SPHYNX-13 EEG ISC v2 (ICA + FASTER + corrected stats)")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Trials loaded: {len(records)}")
    if qc_path.is_file():
        qc = pd.read_csv(qc_path)
        tot_in = int(qc["n_epochs_created"].sum())
        tot_kept = int(qc["n_trials_kept"].sum())
        lines.append(f"Preprocessing retention: {tot_kept}/{tot_in} ({tot_kept/max(tot_in,1):.0%})")
        lines.append(f"Participants with EEG: {len(qc)}")
    lines.append("")
    lines.append("ISC levels (main factors, GFP)")
    main = isc_df[
        (isc_df["signal"] == "gfp_isc")
        & (isc_df["condition_factor"].isin(["adaptation", "graph_nodes", "question_level"]))
    ].sort_values(["condition_factor", "condition_level"])
    for _, r in main.iterrows():
        lines.append(
            f"  {r['condition_factor']:15s} {str(r['condition_level']):20s} "
            f"ISC_r={r['isc_mean_r']:.3f} n={r['n_subjects']}"
        )
    lines.append("")
    lines.append("Contrasts — raw p<.05")
    raw_sig = contrasts[contrasts["sig_raw_0.05"] == True]  # noqa: E712
    if raw_sig.empty:
        lines.append("  none")
    else:
        for _, r in raw_sig.sort_values("p_raw").iterrows():
            lines.append(
                f"  {r['signal']} {r['condition_factor']} {r['level_a']} vs {r['level_b']}: "
                f"r={r['mean_r_a']:.3f} vs {r['mean_r_b']:.3f} p_raw={r['p_raw']:.4g}"
            )
    lines.append("")
    lines.append("Contrasts — FDR within factor (primary corrected claim)")
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
    lines.append("Contrasts — FDR global")
    fdr_g = contrasts[contrasts["sig_fdr_global"] == True]  # noqa: E712
    if fdr_g.empty:
        lines.append("  none")
    else:
        for _, r in fdr_g.sort_values("p_fdr_global").iterrows():
            lines.append(
                f"  {r['signal']} {r['condition_factor']} {r['level_a']} vs {r['level_b']}: "
                f"p_fdr_global={r['p_fdr_global']:.4g}"
            )
    lines.append("")
    lines.append("v1 vs v2 — GFP adaptation levels")
    if not level_cmp.empty:
        adapt = level_cmp[
            (level_cmp["signal"] == "gfp_isc") & (level_cmp["condition_factor"] == "adaptation")
        ]
        for _, r in adapt.sort_values("condition_level").iterrows():
            lines.append(
                f"  {r['condition_level']:15s} v1={r.get('isc_mean_r_v1', np.nan):.3f} "
                f"v2={r.get('isc_mean_r_v2', np.nan):.3f} delta={r.get('delta_r', np.nan):+.3f}"
            )
    lines.append("")
    lines.append("v1 vs v2 — adaptation contrasts (GFP)")
    if not contrast_cmp.empty:
        ac = contrast_cmp[
            (contrast_cmp["signal"] == "gfp_isc") & (contrast_cmp["condition_factor"] == "adaptation")
        ]
        for _, r in ac.iterrows():
            lines.append(
                f"  {r['level_a']} vs {r['level_b']}: "
                f"v1 p={r.get('p_raw_v1', np.nan):.4g} sig_v1={r.get('sig_v1_raw', False)} | "
                f"v2 p_raw={r.get('p_raw_v2', np.nan):.4g} p_fdr={r.get('p_fdr_factor', np.nan):.4g} "
                f"sig_fdr={r.get('sig_v2_fdr_factor', False)}"
            )
    lines.append("")
    lines.append("NOTES")
    lines.append("- v2: ICA + mne-faster (no EOG — eog_correlation excluded from component metrics).")
    lines.append("- v1: 400 µV peak-to-peak rejection, no ICA (~37% epochs dropped).")
    lines.append("- Primary corrected inference: FDR-BH within condition_factor.")
    lines.append("- LOO ISC Wilcoxon remains exploratory; correction addresses contrast multiplicity.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sphynx-13 EEG ISC v2 with corrected stats")
    parser.add_argument("--preprocessed-dir", type=Path, default=PRE_DIR)
    parser.add_argument("-o", "--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--v1-dir", type=Path, default=None)
    args = parser.parse_args()

    root = REPO_ROOT
    pre_dir = args.preprocessed_dir
    if not pre_dir.is_absolute():
        pre_dir = FOR_REPORT / "results" / "eeg_isc_v2_antialias" / pre_dir
    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = FOR_REPORT / "results" / "eeg_isc_v2_antialias" / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    v1_dir = args.v1_dir or (root / V1_RESULTS_DIR)
    if not v1_dir.is_absolute():
        v1_dir = root / v1_dir

    records = load_preprocessed_records(pre_dir)
    print(f"Loaded {len(records)} preprocessed trials")

    isc_df, subj_df = run_isc_analysis(records)
    contrasts = contrast_subject_isc(subj_df)
    level_cmp, contrast_cmp = compare_to_v1(isc_df, contrasts, v1_dir)

    isc_path = out_dir / "analysisA_isc_timecourse_v2.csv"
    subj_path = out_dir / "analysisA_subject_loo_v2.csv"
    con_path = out_dir / "analysisA_isc_contrasts_v2.csv"
    level_cmp_path = out_dir / "v1_v2_isc_levels.csv"
    con_cmp_path = out_dir / "v1_v2_isc_contrasts.csv"
    report_path = out_dir / "eeg_isc_v2_report.txt"

    isc_df.to_csv(isc_path, index=False)
    subj_df.to_csv(subj_path, index=False)
    contrasts.to_csv(con_path, index=False)
    level_cmp.to_csv(level_cmp_path, index=False)
    contrast_cmp.to_csv(con_cmp_path, index=False)
    write_report(
        report_path,
        records,
        isc_df,
        contrasts,
        level_cmp,
        contrast_cmp,
        pre_dir / "preprocessing_qc.csv",
    )

    print(f"Wrote {isc_path}")
    print(f"Wrote {con_path}")
    print(f"Wrote {report_path}")
    print(
        f"Raw sig: {int(contrasts['sig_raw_0.05'].sum())} | "
        f"FDR-factor: {int(contrasts['sig_fdr_factor'].sum())} | "
        f"FDR-global: {int(contrasts['sig_fdr_global'].sum())}"
    )


if __name__ == "__main__":
    main()
