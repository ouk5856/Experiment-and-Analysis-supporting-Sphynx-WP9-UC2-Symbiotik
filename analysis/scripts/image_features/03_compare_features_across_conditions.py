#!/usr/bin/env python3
"""
Compare low-level and high-level image features across experiment conditions.

Factors (same as Sphynx-13):
  adaptation: non_adapted / semi_adapted / fully_adapted
  number_of_nodes: 6 vs 12
  question_level: 1 vs 2

This stage asks whether image features split by condition labels (stimulus check).
p-values are FDR-BH / Bonferroni corrected within each factor family.
It does NOT test whether those splits track human RT/accuracy/ET/ISC — that is script 04.

  python 03_compare_features_across_conditions.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
LOW_CSV = HERE / "results" / "low_level" / "low_level_features.csv"
HIGH_ROOT = HERE / "results" / "high_level"
OUT_DIR = HERE / "results" / "comparisons"

LOW_FEATURES = ["digital_ink", "data_utility", "effectiveness"]
SALIENCY_FEATURES = [
    "saliency_mean",
    "saliency_std",
    "saliency_max",
    "saliency_min",
    "saliency_entropy",
    "saliency_center_bias",
    "saliency_spread",
    "saliency_skewness",
    "saliency_kurtosis",
]
SCANPATH_FEATURES = [
    "fixation_count",
    "fixation_duration_mean",
    "fixation_duration_std",
    "scanpath_length",
    "scanpath_complexity",
    "mean_saccade_mag",
    "cognitive_load",
    "info_processing_idx",
]

ADAPT_ORDER = ["non_adapted", "semi_adapted", "fully_adapted"]
NEAR_CONSTANT_REL_RANGE = 1e-6


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def is_near_constant(x: np.ndarray) -> bool:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return True
    span = float(np.max(x) - np.min(x))
    scale = float(np.mean(np.abs(x))) + 1e-12
    return span / scale < NEAR_CONSTANT_REL_RANGE


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
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    restored = np.empty(n, dtype=float)
    restored[order] = adj
    out[mask] = restored
    return out


def add_multiple_comparisons(tests: pd.DataFrame) -> pd.DataFrame:
    df = tests.copy()
    df["near_constant"] = df["near_constant"].fillna(False).astype(bool)
    df["p_fdr_factor"] = np.nan
    df["p_fdr_global"] = np.nan
    df["p_bonferroni_global"] = np.nan
    df["p_bonferroni_factor"] = np.nan

    eligible = (~df["near_constant"]) & df["p_raw"].notna()
    n_global = int(eligible.sum())
    if n_global > 0:
        df.loc[eligible, "p_fdr_global"] = benjamini_hochberg(df.loc[eligible, "p_raw"].to_numpy())
        df.loc[eligible, "p_bonferroni_global"] = np.minimum(
            df.loc[eligible, "p_raw"].to_numpy() * n_global, 1.0
        )

    for factor, sub_idx in df.groupby("factor").groups.items():
        idx = pd.Index(sub_idx)
        fam = eligible.loc[idx]
        fam_idx = idx[fam.to_numpy()]
        n_fam = len(fam_idx)
        if n_fam == 0:
            continue
        raw = df.loc[fam_idx, "p_raw"].to_numpy()
        df.loc[fam_idx, "p_fdr_factor"] = benjamini_hochberg(raw)
        df.loc[fam_idx, "p_bonferroni_factor"] = np.minimum(raw * n_fam, 1.0)

    df["sig_raw_0.05"] = df["p_raw"].lt(0.05) & ~df["near_constant"]
    df["sig_fdr_factor"] = df["p_fdr_factor"].lt(0.05) & ~df["near_constant"]
    df["sig_fdr_global"] = df["p_fdr_global"].lt(0.05) & ~df["near_constant"]
    df["sig_bonferroni_factor"] = df["p_bonferroni_factor"].lt(0.05) & ~df["near_constant"]
    df["sig_bonferroni_global"] = df["p_bonferroni_global"].lt(0.05) & ~df["near_constant"]
    df["significant_0.05"] = df["sig_raw_0.05"]
    return df


def mannwhitney(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 1 or len(b) < 1:
        return np.nan
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def kruskal(groups: List[np.ndarray]) -> float:
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return np.nan
    try:
        return float(stats.kruskal(*groups).pvalue)
    except Exception:
        return np.nan


def mean_safe(x: np.ndarray) -> float:
    if x is None or len(x) == 0:
        return np.nan
    return float(np.mean(x))


def _row(
    source: str,
    feature: str,
    factor: str,
    comparison: str,
    n_a: float,
    n_b: float,
    mean_a: float,
    mean_b: float,
    mean_non: float,
    mean_semi: float,
    mean_full: float,
    p_raw: float,
    near_constant: bool,
) -> dict:
    return {
        "source": source,
        "feature": feature,
        "factor": factor,
        "comparison": comparison,
        "n_a": n_a,
        "n_b": n_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_non": mean_non,
        "mean_semi": mean_semi,
        "mean_full": mean_full,
        "p_raw": p_raw,
        "near_constant": bool(near_constant),
        "significant_0.05": bool(p_raw < 0.05) if pd.notna(p_raw) and not near_constant else False,
    }


def test_feature(df: pd.DataFrame, feature: str, source: str) -> List[dict]:
    rows = []
    x = _numeric(df[feature]) if feature in df.columns else pd.Series(dtype=float)
    if x.notna().sum() < 3:
        return rows
    feat_constant = is_near_constant(x.dropna().to_numpy())

    # adaptation (3-level omnibus + pairwise)
    if "adaptation" in df.columns:
        groups = []
        means = {}
        for level in ADAPT_ORDER:
            g = _numeric(df.loc[df["adaptation"] == level, feature]).dropna().to_numpy()
            groups.append(g)
            means[level] = mean_safe(g)
        p_omni = kruskal(groups)
        rows.append(
            _row(
                source,
                feature,
                "adaptation",
                "omnibus_non_semi_full",
                int(sum(len(g) for g in groups)),
                np.nan,
                np.nan,
                np.nan,
                means.get("non_adapted"),
                means.get("semi_adapted"),
                means.get("fully_adapted"),
                p_omni,
                feat_constant,
            )
        )
        pairs = [
            ("non_adapted", "fully_adapted"),
            ("non_adapted", "semi_adapted"),
            ("semi_adapted", "fully_adapted"),
        ]
        for a, b in pairs:
            xa = _numeric(df.loc[df["adaptation"] == a, feature]).dropna().to_numpy()
            xb = _numeric(df.loc[df["adaptation"] == b, feature]).dropna().to_numpy()
            p = mannwhitney(xa, xb)
            rows.append(
                _row(
                    source,
                    feature,
                    "adaptation",
                    f"{a}_vs_{b}",
                    int(len(xa)),
                    int(len(xb)),
                    mean_safe(xa),
                    mean_safe(xb),
                    means.get("non_adapted"),
                    means.get("semi_adapted"),
                    means.get("fully_adapted"),
                    p,
                    feat_constant or is_near_constant(np.concatenate([xa, xb])) if len(xa) and len(xb) else feat_constant,
                )
            )

    # graph size
    if "number_of_nodes" in df.columns:
        a = _numeric(df.loc[df["number_of_nodes"] == 6, feature]).dropna().to_numpy()
        b = _numeric(df.loc[df["number_of_nodes"] == 12, feature]).dropna().to_numpy()
        p = mannwhitney(a, b)
        rows.append(
            _row(
                source,
                feature,
                "graph_nodes",
                "6_vs_12",
                int(len(a)),
                int(len(b)),
                mean_safe(a),
                mean_safe(b),
                np.nan,
                np.nan,
                np.nan,
                p,
                feat_constant or is_near_constant(np.concatenate([a, b])) if len(a) and len(b) else feat_constant,
            )
        )

    # question level
    if "question_level" in df.columns:
        a = _numeric(df.loc[df["question_level"] == 1, feature]).dropna().to_numpy()
        b = _numeric(df.loc[df["question_level"] == 2, feature]).dropna().to_numpy()
        p = mannwhitney(a, b)
        rows.append(
            _row(
                source,
                feature,
                "question_level",
                "Q1_vs_Q2",
                int(len(a)),
                int(len(b)),
                mean_safe(a),
                mean_safe(b),
                np.nan,
                np.nan,
                np.nan,
                p,
                feat_constant or is_near_constant(np.concatenate([a, b])) if len(a) and len(b) else feat_constant,
            )
        )
    return rows


def load_high(model: str) -> Optional[pd.DataFrame]:
    path = HIGH_ROOT / model / f"{model}_features.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    return df if len(df) else None


def _fmt_p(p: float) -> str:
    if pd.isna(p):
        return "NA"
    return f"{p:.3g}"


def _mean_extra(r: pd.Series) -> str:
    if r["comparison"] == "omnibus_non_semi_full":
        return (
            f"  means non={r['mean_non']:.4g} semi={r['mean_semi']:.4g} "
            f"full={r['mean_full']:.4g}"
        )
    if pd.notna(r["mean_a"]) and pd.notna(r["mean_b"]):
        return f"  mean_a={r['mean_a']:.4g} mean_b={r['mean_b']:.4g}"
    return ""


def _write_sig_block(lines: List[str], sub: pd.DataFrame, p_col: str) -> None:
    if sub.empty:
        lines.append("  none")
        return
    for _, r in sub.sort_values(p_col).iterrows():
        lines.append(
            f"  {r['feature']:24s} {r['factor']:16s} {r['comparison']:40s} "
            f"p_raw={_fmt_p(r['p_raw'])}  {p_col}={_fmt_p(r[p_col])}{_mean_extra(r)}"
        )


def write_report(all_tests: pd.DataFrame, out_path: Path, missing_high: List[str]) -> None:
    lines = []
    lines.append("POST_ANALYSIS — FEATURES ACROSS CONDITIONS")
    lines.append("N stimuli = 48 catalog rows. Exploratory Mann–Whitney / Kruskal.")
    lines.append("This stage does NOT compare to human RT / accuracy / ISC (see 04).")
    lines.append("Question: do image features differentiate adaptation / graph size / question level?")
    lines.append("")
    lines.append("Correction:")
    lines.append("  FDR-BH and Bonferroni within each factor family (adaptation / graph_nodes /")
    lines.append("  question_level), and again across all eligible tests (global).")
    lines.append("  Near-constant features (e.g. max-normalized saliency_max ≈ 1) are excluded")
    lines.append("  from correction families and from significance counts.")
    lines.append("  Primary corrected claim = FDR-BH within factor (sig_fdr_factor).")
    lines.append("")

    if missing_high:
        lines.append("High-level models not yet available: " + ", ".join(missing_high))
        lines.append("Re-run this script after remote inference CSVs are copied here.")
        lines.append("")

    if all_tests.empty:
        lines.append("No tests could be run (missing feature CSVs).")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    n_const = int(all_tests["near_constant"].sum())
    n_elig = int((~all_tests["near_constant"] & all_tests["p_raw"].notna()).sum())
    lines.append(
        f"Tests run: {len(all_tests)}  |  eligible (not near-constant): {n_elig}  |  "
        f"near-constant excluded: {n_const}"
    )
    lines.append(
        f"Uncorrected p_raw<.05: {int(all_tests['sig_raw_0.05'].sum())}  |  "
        f"FDR-factor: {int(all_tests['sig_fdr_factor'].sum())}  |  "
        f"FDR-global: {int(all_tests['sig_fdr_global'].sum())}  |  "
        f"Bonferroni-factor: {int(all_tests['sig_bonferroni_factor'].sum())}  |  "
        f"Bonferroni-global: {int(all_tests['sig_bonferroni_global'].sum())}"
    )
    lines.append("")

    lines.append("══ UNCORRECTED (p_raw < .05, near-constant dropped) ══")
    lines.append("")
    for source, sub in all_tests.groupby("source"):
        lines.append(f"── {source} ──")
        _write_sig_block(lines, sub[sub["sig_raw_0.05"] == True], "p_raw")  # noqa: E712
        lines.append("")

    lines.append("══ FDR-BH WITHIN FACTOR (primary corrected) ══")
    lines.append("")
    for source, sub in all_tests.groupby("source"):
        lines.append(f"── {source} ──")
        _write_sig_block(lines, sub[sub["sig_fdr_factor"] == True], "p_fdr_factor")  # noqa: E712
        lines.append("")

    lines.append("══ FDR-BH GLOBAL ══")
    lines.append("")
    fdr_g = all_tests[all_tests["sig_fdr_global"] == True]  # noqa: E712
    if fdr_g.empty:
        lines.append("  none")
        lines.append("")
    else:
        for source, sub in fdr_g.groupby("source"):
            lines.append(f"── {source} ──")
            _write_sig_block(lines, sub, "p_fdr_global")
            lines.append("")

    lines.append("══ BONFERRONI WITHIN FACTOR ══")
    lines.append("")
    bon_f = all_tests[all_tests["sig_bonferroni_factor"] == True]  # noqa: E712
    if bon_f.empty:
        lines.append("  none")
        lines.append("")
    else:
        for source, sub in bon_f.groupby("source"):
            lines.append(f"── {source} ──")
            _write_sig_block(lines, sub, "p_bonferroni_factor")
            lines.append("")

    dropped = all_tests[all_tests["near_constant"] == True]  # noqa: E712
    if not dropped.empty:
        lines.append("══ NEAR-CONSTANT (excluded from correction; not claimed) ══")
        for _, r in dropped.drop_duplicates(["source", "feature"]).iterrows():
            lines.append(f"  {r['source']:12s} {r['feature']}")
        lines.append("")

    lines.append("INTERPRETATION HINT")
    lines.append("Uncorrected splits only show that some image statistics differ by condition")
    lines.append("label. That is a stimulus check, not evidence that the feature tracks human")
    lines.append("benefit. Compare features to RT / accuracy / ET / ISC in script 04.")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare image features across conditions")
    parser.add_argument("--low-csv", type=Path, default=LOW_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tests: List[dict] = []
    missing_high: List[str] = []

    if args.low_csv.is_file():
        low = pd.read_csv(args.low_csv)
        for feat in LOW_FEATURES:
            tests.extend(test_feature(low, feat, source="low_level"))
        print(f"Loaded low-level: {args.low_csv} ({len(low)} rows)")
    else:
        print(f"Missing low-level CSV: {args.low_csv}")

    for model in ("visalformer", "sum", "clipgaze"):
        hdf = load_high(model)
        if hdf is None:
            missing_high.append(model)
            print(f"Skipping {model}: no features CSV yet")
            continue
        print(f"Loaded {model}: {len(hdf)} ok rows")
        feats = list(SALIENCY_FEATURES)
        if model == "clipgaze":
            feats = feats + SCANPATH_FEATURES
        for feat in feats:
            tests.extend(test_feature(hdf, feat, source=model))

    tests_df = pd.DataFrame(tests)
    tests_path = args.out_dir / "condition_tests.csv"
    report_path = args.out_dir / "condition_comparison_report.txt"
    if not tests_df.empty:
        tests_df = add_multiple_comparisons(tests_df)
        tests_df.to_csv(tests_path, index=False)
        print(f"Wrote {tests_path}")
        print(
            f"  raw p<.05={int(tests_df['sig_raw_0.05'].sum())}  "
            f"FDR-factor={int(tests_df['sig_fdr_factor'].sum())}  "
            f"FDR-global={int(tests_df['sig_fdr_global'].sum())}  "
            f"Bonferroni-factor={int(tests_df['sig_bonferroni_factor'].sum())}"
        )
    write_report(tests_df, report_path, missing_high)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
