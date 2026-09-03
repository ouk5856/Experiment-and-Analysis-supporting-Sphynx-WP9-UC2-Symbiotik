#!/usr/bin/env python3
"""
Permutation inference for Sphynx-13 EEG LOO timecourse ISC (v2 data).

Two null models:
  1. ISC level — circular time shift per subject (destroys cross-subject alignment,
     preserves each subject's autocorrelation). Tests whether mean LOO ISC > chance.
  2. Condition contrast — random sign flip on paired Fisher-z differences (exact
     paired permutation for mean difference). One-sided when level_a is expected higher.

Reads preprocessed trials from 01_preprocess_eeg_ica_faster.py.
Run after 02_run_eeg_isc_corrected.py (or standalone).

  python 03_permutation_isc.py
  python 03_permutation_isc.py --n-perm 10000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sphynx13_eeg_utils import (
    apply_pvalue_corrections,
    average_timecourses,
    benjamini_hochberg,
    circular_shift_isc_null,
    condition_keys,
    contrast_pairs,
    fisher_z,
    leave_one_out_isc,
    permutation_p_one_sided,
    permutation_p_two_sided,
    sign_flip_contrast_null,
)

HERE = Path(__file__).resolve().parent
FOR_REPORT = HERE.parents[1]
PRE_DIR = FOR_REPORT / "results" / "eeg_isc_v2_antialias" / "preprocessed"
OUT_DIR = FOR_REPORT / "results" / "eeg_isc_v2_antialias"

EEG_SIGNALS = ["gfp_isc", "occ_isc"]
MAIN_FACTORS = ["adaptation", "graph_nodes", "question_level"]
DEFAULT_N_PERM = 5000
SEED = 42

# Adaptation contrasts where level_a (non_adapted) is expected higher ISC
ONE_SIDED_HIGHER_A = {
    ("adaptation", "non_adapted", "fully_adapted"),
    ("adaptation", "non_adapted", "semi_adapted"),
}


def load_preprocessed_records(pre_dir: Path) -> list[dict]:
    index_path = pre_dir / "trial_index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing {index_path}; run 01_preprocess_eeg_ica_faster.py first")
    index_df = pd.read_csv(index_path)
    records = []
    for _, row in index_df.iterrows():
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
                "gfp_isc": np.load(pre_dir / row["gfp_npy"]),
                "occ_isc": np.load(pre_dir / row["occ_npy"]),
            }
        )
    return records


def run_level_permutation(
    records: list[dict],
    n_perm: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for signal_key in EEG_SIGNALS:
        for cond_name in MAIN_FACTORS:
            cond_fn = dict(condition_keys())[cond_name]
            by_cond = average_timecourses(records, signal_key, cond_fn)
            for lab, subj_ts in by_cond.items():
                obs_r, _, n = leave_one_out_isc(subj_ts)
                null = circular_shift_isc_null(subj_ts, n_perm, rng)
                p_one = permutation_p_one_sided(null, obs_r)
                null_mean = float(np.nanmean(null)) if len(null) else np.nan
                rows.append(
                    {
                        "test_type": "isc_level_circular_shift",
                        "signal": signal_key,
                        "condition_factor": cond_name,
                        "condition_level": lab,
                        "n_subjects": n,
                        "isc_mean_r_observed": obs_r,
                        "isc_mean_r_null": null_mean,
                        "n_perm": n_perm,
                        "p_perm_one_sided": p_one,
                        "sig_perm_0.05": bool(p_one < 0.05) if np.isfinite(p_one) else False,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_fdr_global"] = np.nan
    df["p_fdr_factor"] = np.nan
    eligible = df["p_perm_one_sided"].notna()
    n_global = int(eligible.sum())
    if n_global:
        raw = df.loc[eligible, "p_perm_one_sided"].to_numpy()
        df.loc[eligible, "p_fdr_global"] = benjamini_hochberg(raw)
    for factor, fam in df.groupby("condition_factor"):
        fam_idx = fam.index[eligible.loc[fam.index]]
        if len(fam_idx) == 0:
            continue
        raw = df.loc[fam_idx, "p_perm_one_sided"].to_numpy()
        df.loc[fam_idx, "p_fdr_factor"] = benjamini_hochberg(raw)
    df["sig_fdr_factor"] = df["p_fdr_factor"].lt(0.05)
    df["sig_fdr_global"] = df["p_fdr_global"].lt(0.05)
    return df


def run_contrast_permutation(
    subj_df: pd.DataFrame,
    n_perm: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    pairs = contrast_pairs()
    for signal_key in EEG_SIGNALS:
        gsig = subj_df[subj_df["signal"] == signal_key]
        for factor in MAIN_FACTORS:
            if factor not in pairs:
                continue
            g = gsig[gsig["condition_factor"] == factor]
            for level_a, level_b in pairs[factor]:
                ga = g[g["condition_level"] == level_a][["participant_id", "loo_fisher_z"]].rename(
                    columns={"loo_fisher_z": "z_a"}
                )
                gb = g[g["condition_level"] == level_b][["participant_id", "loo_fisher_z"]].rename(
                    columns={"loo_fisher_z": "z_b"}
                )
                m = ga.merge(gb, on="participant_id").dropna()
                if len(m) < 3:
                    continue
                diff = (m["z_a"] - m["z_b"]).to_numpy()
                obs_mean_diff = float(np.mean(diff))
                null = sign_flip_contrast_null(diff, n_perm, rng)
                key = (factor, level_a, level_b)
                if key in ONE_SIDED_HIGHER_A and signal_key == "gfp_isc":
                    p_perm = permutation_p_one_sided(null, obs_mean_diff)
                    tail = "a_gt_b"
                else:
                    p_perm = permutation_p_two_sided(null, obs_mean_diff)
                    tail = "two_sided"
                rows.append(
                    {
                        "test_type": "contrast_sign_flip",
                        "signal": signal_key,
                        "condition_factor": factor,
                        "level_a": level_a,
                        "level_b": level_b,
                        "n_paired": len(m),
                        "mean_z_diff_observed": obs_mean_diff,
                        "mean_z_diff_null": float(np.mean(null)) if len(null) else np.nan,
                        "mean_r_a": float(np.tanh(m["z_a"].mean())),
                        "mean_r_b": float(np.tanh(m["z_b"].mean())),
                        "n_perm": n_perm,
                        "tail": tail,
                        "p_raw": p_perm,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return apply_pvalue_corrections(df)


def build_subject_loo_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for signal_key in EEG_SIGNALS:
        for cond_name, cond_fn in condition_keys():
            if cond_name not in MAIN_FACTORS:
                continue
            by_cond = average_timecourses(records, signal_key, cond_fn)
            for lab, subj_ts in by_cond.items():
                _, per_subj, _ = leave_one_out_isc(subj_ts)
                for pid, r in per_subj.items():
                    rows.append(
                        {
                            "signal": signal_key,
                            "condition_factor": cond_name,
                            "condition_level": lab,
                            "participant_id": pid,
                            "loo_r": r,
                            "loo_fisher_z": fisher_z(r) if np.isfinite(r) else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def merge_with_wilcoxon(contrast_perm: pd.DataFrame, wilcoxon_path: Path) -> pd.DataFrame:
    if not wilcoxon_path.is_file() or contrast_perm.empty:
        return contrast_perm
    w = pd.read_csv(wilcoxon_path)
    w = w.rename(columns={"p_raw": "p_wilcoxon", "p_fdr_factor": "p_fdr_factor_wilcoxon"})
    keys = ["signal", "condition_factor", "level_a", "level_b"]
    return contrast_perm.merge(
        w[keys + ["p_wilcoxon", "p_fdr_factor_wilcoxon", "wilcoxon_stat"]],
        on=keys,
        how="left",
    )


def write_report(
    path: Path,
    level_df: pd.DataFrame,
    contrast_df: pd.DataFrame,
    n_perm: int,
) -> None:
    lines = []
    lines.append("SPHYNX-13 EEG ISC v2 — PERMUTATION INFERENCE")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Permutations per test: {n_perm}")
    lines.append("Level null: independent circular time shift per subject, then LOO ISC.")
    lines.append("Contrast null: random sign flip on paired Fisher-z differences.")
    lines.append("")
    lines.append("ISC LEVEL — circular shift (one-sided: observed > null), GFP adaptation")
    adapt = level_df[
        (level_df["signal"] == "gfp_isc") & (level_df["condition_factor"] == "adaptation")
    ].sort_values("condition_level")
    for _, r in adapt.iterrows():
        lines.append(
            f"  {r['condition_level']:15s} ISC_r={r['isc_mean_r_observed']:.3f} "
            f"null_mean={r['isc_mean_r_null']:.3f} p_perm={r['p_perm_one_sided']:.4g} "
            f"FDR_factor={r.get('p_fdr_factor', np.nan):.4g}"
        )
    lines.append("")
    lines.append("ISC LEVEL — significant after FDR within factor (any signal)")
    sig = level_df[level_df["sig_fdr_factor"] == True]  # noqa: E712
    if sig.empty:
        lines.append("  none")
    else:
        for _, r in sig.sort_values("p_fdr_factor").iterrows():
            lines.append(
                f"  {r['signal']} {r['condition_factor']}={r['condition_level']}: "
                f"p_perm={r['p_perm_one_sided']:.4g} p_fdr={r['p_fdr_factor']:.4g}"
            )
    lines.append("")
    lines.append("CONTRASTS — sign-flip permutation vs Wilcoxon (GFP)")
    gfp = contrast_df[contrast_df["signal"] == "gfp_isc"].sort_values("p_raw")
    for _, r in gfp.iterrows():
        lines.append(
            f"  {r['condition_factor']} {r['level_a']} vs {r['level_b']}: "
            f"r={r['mean_r_a']:.3f} vs {r['mean_r_b']:.3f} "
            f"p_perm={r['p_raw']:.4g} "
            f"p_wilcoxon={r.get('p_wilcoxon', np.nan):.4g} "
            f"FDR_perm={r.get('p_fdr_factor', np.nan):.4g}"
        )
    lines.append("")
    lines.append("CONTRASTS — significant permutation FDR within factor")
    csig = contrast_df[contrast_df["sig_fdr_factor"] == True]  # noqa: E712
    if csig.empty:
        lines.append("  none")
    else:
        for _, r in csig.sort_values("p_fdr_factor").iterrows():
            lines.append(
                f"  {r['signal']} {r['condition_factor']} {r['level_a']} vs {r['level_b']}: "
                f"p_perm={r['p_raw']:.4g}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Permutation ISC for EEG v2")
    parser.add_argument("--preprocessed-dir", type=Path, default=PRE_DIR)
    parser.add_argument("-o", "--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    pre_dir = args.preprocessed_dir if args.preprocessed_dir.is_absolute() else HERE / args.preprocessed_dir
    out_dir = args.output_dir if args.output_dir.is_absolute() else HERE / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    records = load_preprocessed_records(pre_dir)
    print(f"Loaded {len(records)} trials; running {args.n_perm} permutations per test")

    level_df = run_level_permutation(records, args.n_perm, rng)
    subj_df = build_subject_loo_table(records)
    contrast_df = run_contrast_permutation(subj_df, args.n_perm, rng)
    contrast_df = merge_with_wilcoxon(contrast_df, out_dir / "analysisA_isc_contrasts_v2.csv")

    level_path = out_dir / "permutation_isc_levels.csv"
    contrast_path = out_dir / "permutation_isc_contrasts.csv"
    report_path = out_dir / "permutation_isc_report.txt"

    level_df.to_csv(level_path, index=False)
    contrast_df.to_csv(contrast_path, index=False)
    write_report(report_path, level_df, contrast_df, args.n_perm)

    print(f"Wrote {level_path}")
    print(f"Wrote {contrast_path}")
    print(f"Wrote {report_path}")
    print(
        f"Level sig (perm FDR factor): {int(level_df['sig_fdr_factor'].sum())} | "
        f"Contrast sig (perm FDR factor): {int(contrast_df['sig_fdr_factor'].sum())}"
    )


if __name__ == "__main__":
    main()
