#!/usr/bin/env python3
"""
Cohen's d + bootstrap CIs for key contrasts across modalities.

Addresses caveat: no effect sizes reported anywhere.
EEG ISC subject LOO from anti-alias package results (for_report).

  python 04_effect_sizes.py
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
OUT_DIR = FOR_REPORT / "results" / "multimodal"

EEG_ISC_SUBJ = (
    FOR_REPORT / "results" / "eeg_isc_v2_antialias" / "analysisA_subject_loo_v2.csv"
)
MULTIMODAL_SUBJ = OUT_DIR / "multimodal_subject_loo.csv"
BEH_PATH = FOR_REPORT / "results" / "behaviour" / "pilot_trial_level_with_accuracy.csv"

N_BOOT = 5000
SEED = 42

CONTRAST_PAIRS = {
    "adaptation": [
        ("non_adapted", "fully_adapted"),
        ("non_adapted", "semi_adapted"),
    ],
    "graph_nodes": [("6", "12")],
    "question_level": [("1", "2")],
}


def paired_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return np.nan
    return float(np.mean(diff) / np.std(diff, ddof=1))


def bootstrap_mean_diff_ci(a, b, n_boot, rng, alpha=0.05):
    diff = a - b
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return np.nan, np.nan
    boots = np.empty(n_boot)
    n = len(diff)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = np.mean(diff[idx])
    lo = float(np.nanpercentile(boots, 100 * alpha / 2))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def isc_effect_sizes(subj_path: Path, label: str, rng) -> list[dict]:
    if not subj_path.is_file():
        return []
    subj = pd.read_csv(subj_path)
    rows = []
    for (signal, factor), g in subj.groupby(["signal", "condition_factor"]):
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
            za = m["z_a"].to_numpy()
            zb = m["z_b"].to_numpy()
            d = paired_cohens_d(za, zb)
            ci_lo, ci_hi = bootstrap_mean_diff_ci(za, zb, N_BOOT, rng)
            rows.append({
                "domain": label,
                "signal": signal,
                "factor": factor,
                "level_a": a,
                "level_b": b,
                "n": len(m),
                "mean_a": float(np.tanh(za.mean())),
                "mean_b": float(np.tanh(zb.mean())),
                "cohens_d": d,
                "mean_diff": float(np.mean(za - zb)),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            })
    return rows


def behaviour_effect_sizes(beh_path: Path, rng) -> list[dict]:
    if not beh_path.is_file():
        return []
    beh = pd.read_csv(beh_path)
    beh["rt_numeric"] = pd.to_numeric(beh["rt"], errors="coerce")
    beh["correct"] = beh["is_correct"].astype(int)
    rows = []

    for outcome, col in [("RT", "rt_numeric"), ("accuracy", "correct")]:
        subj = beh.groupby(["participant", "adaptation"])[col].mean().reset_index()
        subj = subj.rename(columns={"participant": "participant_id", col: "value"})
        for a, b in [("non_adapted", "fully_adapted"), ("non_adapted", "semi_adapted")]:
            va = subj[subj["adaptation"] == a][["participant_id", "value"]].rename(columns={"value": "va"})
            vb = subj[subj["adaptation"] == b][["participant_id", "value"]].rename(columns={"value": "vb"})
            m = va.merge(vb, on="participant_id").dropna()
            if len(m) < 3:
                continue
            xa = m["va"].to_numpy()
            xb = m["vb"].to_numpy()
            d = paired_cohens_d(xa, xb)
            ci_lo, ci_hi = bootstrap_mean_diff_ci(xa, xb, N_BOOT, rng)
            rows.append({
                "domain": "behaviour",
                "signal": outcome,
                "factor": "adaptation",
                "level_a": a,
                "level_b": b,
                "n": len(m),
                "mean_a": float(xa.mean()),
                "mean_b": float(xb.mean()),
                "cohens_d": d,
                "mean_diff": float(np.mean(xa - xb)),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            })

        # Q and nodes
        for factor_col, factor_name, levels in [
            ("question_level", "question_level", (1, 2)),
            ("graph_nodes", "graph_nodes", (6, 12)),
        ]:
            subj2 = beh.groupby(["participant", factor_col])[col].mean().reset_index()
            subj2 = subj2.rename(columns={"participant": "participant_id", col: "value"})
            a_val, b_val = levels
            va = subj2[subj2[factor_col] == a_val][["participant_id", "value"]].rename(columns={"value": "va"})
            vb = subj2[subj2[factor_col] == b_val][["participant_id", "value"]].rename(columns={"value": "vb"})
            m = va.merge(vb, on="participant_id").dropna()
            if len(m) < 3:
                continue
            xa = m["va"].to_numpy()
            xb = m["vb"].to_numpy()
            d = paired_cohens_d(xa, xb)
            ci_lo, ci_hi = bootstrap_mean_diff_ci(xa, xb, N_BOOT, rng)
            rows.append({
                "domain": "behaviour",
                "signal": outcome,
                "factor": factor_name,
                "level_a": str(a_val),
                "level_b": str(b_val),
                "n": len(m),
                "mean_a": float(xa.mean()),
                "mean_b": float(xb.mean()),
                "cohens_d": d,
                "mean_diff": float(np.mean(xa - xb)),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            })
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    if not EEG_ISC_SUBJ.is_file():
        raise FileNotFoundError(f"Missing anti-alias LOO CSV: {EEG_ISC_SUBJ}")
    if not MULTIMODAL_SUBJ.is_file():
        raise FileNotFoundError(f"Missing multimodal LOO CSV: {MULTIMODAL_SUBJ}")
    if not BEH_PATH.is_file():
        raise FileNotFoundError(f"Missing behaviour CSV: {BEH_PATH}")

    all_rows = []
    all_rows.extend(isc_effect_sizes(EEG_ISC_SUBJ, "eeg_isc_v2_antialias", rng))
    all_rows.extend(isc_effect_sizes(MULTIMODAL_SUBJ, "multimodal_isc", rng))
    all_rows.extend(behaviour_effect_sizes(BEH_PATH, rng))

    df = pd.DataFrame(all_rows)
    csv_path = OUT_DIR / "effect_sizes.csv"
    df.to_csv(csv_path, index=False)

    lines = ["EFFECT SIZES — ALL MODALITIES", "=" * 50, ""]
    lines.append(f"Cohen's d (paired), bootstrap {N_BOOT} CIs on mean difference")
    lines.append(f"EEG ISC LOO: {EEG_ISC_SUBJ} (anti-alias)")
    lines.append("")

    for domain in df["domain"].unique():
        lines.append(f"=== {domain} ===")
        sub = df[df["domain"] == domain]
        for _, r in sub.sort_values(["signal", "factor"]).iterrows():
            d_str = f"d={r['cohens_d']:+.2f}" if np.isfinite(r["cohens_d"]) else "d=n/a"
            lines.append(
                f"  {r['signal']:15s} {r['factor']:15s} {r['level_a']} vs {r['level_b']}: "
                f"{d_str}  mean_diff={r['mean_diff']:+.4f}  "
                f"CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
            )
        lines.append("")

    lines.append("Interpretation:")
    lines.append("  |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large")
    lines.append("  CI on mean difference (Fisher-z for ISC, seconds for RT, proportion for accuracy)")

    report_path = OUT_DIR / "effect_sizes_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path} ({len(df)} rows)")
    print(f"Wrote {report_path}")
    large = df[df["cohens_d"].abs() > 0.8]
    print(f"Large effects (|d|>0.8): {len(large)}")


if __name__ == "__main__":
    main()
