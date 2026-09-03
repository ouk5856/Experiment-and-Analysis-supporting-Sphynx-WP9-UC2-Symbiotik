#!/usr/bin/env python3
"""
Subject-level ISC vs behaviour correlation.

Addresses caveat: ISC-behaviour comparison was qualitative (condition-level).
This tests whether participants with higher EEG GFP ISC also have different
RT/accuracy, and whether per-person ISC *difference* predicts RT *difference*.

Canonical EEG LOO: anti-alias package results (for_report).

  python 01_isc_vs_behaviour.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "All data").is_dir():
            return p
    raise FileNotFoundError(f"Could not find repo root (All data/) from {start}")


REPO = find_repo_root(HERE)
FOR_REPORT = REPO / "All data" / "post_analysis" / "for_report"

ISC_PATH = (
    FOR_REPORT / "results" / "eeg_isc_v2_antialias" / "analysisA_subject_loo_v2.csv"
)
BEH_PATH = FOR_REPORT / "results" / "behaviour" / "pilot_trial_level_with_accuracy.csv"
OUT_DIR = FOR_REPORT / "results" / "multimodal"

N_BOOT = 5000
SEED = 42
SIGNALS = ["gfp_isc", "occ_isc"]


def bootstrap_ci(x: np.ndarray, y: np.ndarray, stat_fn, n_boot: int, rng, alpha: float = 0.05):
    observed = stat_fn(x, y)
    boots = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = stat_fn(x[idx], y[idx])
    lo = float(np.nanpercentile(boots, 100 * alpha / 2))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return observed, lo, hi


def spearman_r(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return np.nan
    return float(stats.spearmanr(x[mask], y[mask]).statistic)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    if not ISC_PATH.is_file():
        raise FileNotFoundError(f"Missing anti-alias LOO CSV: {ISC_PATH}")
    if not BEH_PATH.is_file():
        raise FileNotFoundError(f"Missing behaviour CSV: {BEH_PATH}")

    isc = pd.read_csv(ISC_PATH)
    beh = pd.read_csv(BEH_PATH)

    beh_subj = (
        beh.groupby(["participant", "adaptation"])
        .agg(
            mean_rt=("rt", "mean"),
            median_rt=("rt", "median"),
            accuracy=("is_correct", "mean"),
            n_trials=("is_correct", "count"),
        )
        .reset_index()
        .rename(columns={"participant": "participant_id"})
    )

    rows = []

    for signal in SIGNALS:
        isc_adapt = isc[
            (isc["signal"] == signal) & (isc["condition_factor"] == "adaptation")
        ][["participant_id", "condition_level", "loo_fisher_z"]].copy()

        for adapt_level in ["non_adapted", "semi_adapted", "fully_adapted"]:
            isc_sub = isc_adapt[isc_adapt["condition_level"] == adapt_level][
                ["participant_id", "loo_fisher_z"]
            ]
            beh_sub = beh_subj[beh_subj["adaptation"] == adapt_level][
                ["participant_id", "mean_rt", "accuracy"]
            ]
            m = isc_sub.merge(beh_sub, on="participant_id").dropna()
            if len(m) < 5:
                continue

            for outcome, col in [("mean_rt", "mean_rt"), ("accuracy", "accuracy")]:
                x = m["loo_fisher_z"].to_numpy()
                y = m[col].to_numpy()
                rho, lo, hi = bootstrap_ci(x, y, spearman_r, N_BOOT, rng)
                sp = stats.spearmanr(x, y)
                rows.append({
                    "test": "within_condition",
                    "signal": signal,
                    "condition": adapt_level,
                    "outcome": outcome,
                    "n": len(m),
                    "spearman_rho": float(sp.statistic),
                    "p_value": float(sp.pvalue),
                    "boot_ci_lo": lo,
                    "boot_ci_hi": hi,
                })

        # Paired difference: ISC(non) - ISC(full) vs RT(non) - RT(full)
        isc_wide = isc_adapt.pivot(
            index="participant_id", columns="condition_level", values="loo_fisher_z"
        )
        if "non_adapted" not in isc_wide.columns or "fully_adapted" not in isc_wide.columns:
            continue
        isc_wide["isc_diff"] = isc_wide["non_adapted"] - isc_wide["fully_adapted"]

        beh_wide = beh_subj.pivot(
            index="participant_id", columns="adaptation", values="mean_rt"
        )
        if "non_adapted" not in beh_wide.columns or "fully_adapted" not in beh_wide.columns:
            continue
        beh_wide["rt_diff"] = beh_wide["non_adapted"] - beh_wide["fully_adapted"]

        acc_wide = beh_subj.pivot(
            index="participant_id", columns="adaptation", values="accuracy"
        )
        acc_wide["acc_diff"] = acc_wide["non_adapted"] - acc_wide["fully_adapted"]

        for diff_col, diff_label, diff_df in [
            ("rt_diff", "delta_RT_non_minus_full", beh_wide),
            ("acc_diff", "delta_accuracy_non_minus_full", acc_wide),
        ]:
            m2 = isc_wide[["isc_diff"]].join(diff_df[[diff_col]], how="inner").dropna()
            if len(m2) < 5:
                continue
            x = m2["isc_diff"].to_numpy()
            y = m2[diff_col].to_numpy()
            rho, lo, hi = bootstrap_ci(x, y, spearman_r, N_BOOT, rng)
            sp = stats.spearmanr(x, y)
            rows.append({
                "test": "paired_difference",
                "signal": signal,
                "condition": "non_minus_full",
                "outcome": diff_label,
                "n": len(m2),
                "spearman_rho": float(sp.statistic),
                "p_value": float(sp.pvalue),
                "boot_ci_lo": lo,
                "boot_ci_hi": hi,
            })

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "isc_vs_behaviour.csv"
    df.to_csv(csv_path, index=False)

    report_path = OUT_DIR / "isc_vs_behaviour_report.txt"
    lines = ["SUBJECT-LEVEL ISC vs BEHAVIOUR", "=" * 50, ""]
    lines.append(
        "N participants: 13 (ISC from EEG anti-alias ICA+FASTER; "
        "for_report/results/eeg_isc_v2_antialias/)"
    )
    lines.append(f"Bootstrap CIs: {N_BOOT} resamples")
    lines.append(f"ISC source: {ISC_PATH}")
    lines.append("")
    for signal in SIGNALS:
        sub = df[df["signal"] == signal]
        lines.append(f"--- {signal} ---")
        within = sub[sub["test"] == "within_condition"]
        for _, r in within.iterrows():
            sig = "*" if r["p_value"] < 0.05 else ""
            lines.append(
                f"  {r['condition']:15s} vs {r['outcome']:10s}: "
                f"rho={r['spearman_rho']:+.3f} p={r['p_value']:.3f}{sig} "
                f"CI=[{r['boot_ci_lo']:+.3f}, {r['boot_ci_hi']:+.3f}]"
            )
        paired = sub[sub["test"] == "paired_difference"]
        for _, r in paired.iterrows():
            sig = "*" if r["p_value"] < 0.05 else ""
            lines.append(
                f"  PAIRED {r['outcome']}: "
                f"rho={r['spearman_rho']:+.3f} p={r['p_value']:.3f}{sig} "
                f"CI=[{r['boot_ci_lo']:+.3f}, {r['boot_ci_hi']:+.3f}]"
            )
        lines.append("")
    lines.append("Interpretation:")
    lines.append("- Positive rho for ISC vs RT = higher synchrony -> slower (more shared struggle)")
    lines.append("- Negative rho for ISC vs accuracy = higher synchrony -> less accurate")
    lines.append("- Paired diff: does ISC benefit predict RT benefit?")
    lines.append("- Exploratory battery: no FDR claim across these Spearmans.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path} ({len(df)} rows)")
    print(f"Wrote {report_path}")
    sig_count = int((df["p_value"] < 0.05).sum())
    print(f"Significant (p<.05): {sig_count}/{len(df)}")


if __name__ == "__main__":
    main()
