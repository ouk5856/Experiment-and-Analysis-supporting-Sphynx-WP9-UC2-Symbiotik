#!/usr/bin/env python3
"""
Compare image features to Sphynx-13 human outcomes.

Script 03 only asks whether features split by condition labels. That can be
true just because the images look different. This script asks whether those
feature differences track the human effects already seen in behaviour / ET / ISC.

  python 04_compare_features_to_human.py

Reads (read-only) from experiment_for_Sphynx_pilot/analysis_* ; writes only
under post_analysis/results/human_compare/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
LOW_CSV = HERE / "results" / "low_level" / "low_level_features.csv"
HIGH_ROOT = HERE / "results" / "high_level"
OUT_DIR = HERE / "results" / "human_compare"
PILOT = HERE.parent / "experiment_for_Sphynx_pilot"
TRIAL_CSV = PILOT / "analysis_sphynx13" / "pilot_trial_level_with_accuracy.csv"
ET_CSV = PILOT / "analysis_sphynx13_eeg_et" / "analysisB_features_by_trial.csv"
ISC_CSV = PILOT / "analysis_sphynx13_eeg_et" / "analysisA_isc_timecourse.csv"

LOW_FEATURES = ["digital_ink", "mark_density", "data_utility", "effectiveness"]
SALIENCY_FEATURES = [
    "saliency_mean",
    "saliency_std",
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
SKIP_FEATURES = {"saliency_max", "saliency_min"}
HUMAN_BEHAV = ["rt_median", "accuracy", "response_rate"]
HUMAN_ET = ["et_n_fixations", "et_mean_fix_dur", "et_mean_saccade_mag"]
ADAPT_ORDER = ["non_adapted", "semi_adapted", "fully_adapted"]
NEAR_CONSTANT_REL_RANGE = 1e-6

CLIPGAZE_TO_ET = [
    ("fixation_count", "et_n_fixations"),
    ("fixation_duration_mean", "et_mean_fix_dur"),
    ("mean_saccade_mag", "et_mean_saccade_mag"),
    ("scanpath_length", "et_sum_fix_dur"),
]


def norm_image(path) -> str:
    return str(path).strip().lower()


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


def spearman(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int]:
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < 5:
        return np.nan, np.nan, n
    xa, xb = a[mask], b[mask]
    if is_near_constant(xa) or is_near_constant(xb):
        return np.nan, np.nan, n
    try:
        r, p = stats.spearmanr(xa, xb)
        return float(r), float(p), n
    except Exception:
        return np.nan, np.nan, n


def ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Least-squares residuals; rows with non-finite y or X are NaN."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    resid = np.full(y.shape, np.nan, dtype=float)
    if ok.sum() < X.shape[1] + 2:
        return resid
    Xok = np.column_stack([np.ones(int(ok.sum())), X[ok]])
    beta, *_ = np.linalg.lstsq(Xok, y[ok], rcond=None)
    resid[ok] = y[ok] - Xok @ beta
    return resid


def design_matrix(df: pd.DataFrame) -> np.ndarray:
    adapt = pd.get_dummies(df["adaptation"], prefix="adapt", drop_first=True).to_numpy(dtype=float)
    nodes = (df["number_of_nodes"].astype(float).to_numpy() == 12.0).astype(float).reshape(-1, 1)
    q2 = (df["question_level"].astype(float).to_numpy() == 2.0).astype(float).reshape(-1, 1)
    return np.column_stack([adapt, nodes, q2])


def load_high(model: str) -> Optional[pd.DataFrame]:
    path = HIGH_ROOT / model / f"{model}_features.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    return df if len(df) else None


def load_features() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if LOW_CSV.is_file():
        low = pd.read_csv(LOW_CSV)
        low["source"] = "low_level"
        low["graph_image"] = low["graph_image"].map(norm_image)
        low["mark_density"] = 765.0 - _numeric(low["digital_ink"])
        keep = [
            "graph_image",
            "number_of_nodes",
            "question_level",
            "adaptation",
            "question",
            "source",
        ] + [c for c in LOW_FEATURES if c in low.columns or c == "mark_density"]
        frames.append(low[keep].copy())
        print(f"Loaded low-level: {len(low)} rows")
    else:
        print(f"Missing low-level CSV: {LOW_CSV}")

    for model in ("visalformer", "sum", "clipgaze"):
        hdf = load_high(model)
        if hdf is None:
            print(f"Skipping {model}: no features CSV")
            continue
        hdf = hdf.copy()
        hdf["source"] = model
        hdf["graph_image"] = hdf["graph_image"].map(norm_image)
        feats = list(SALIENCY_FEATURES)
        if model == "clipgaze":
            feats = feats + SCANPATH_FEATURES
        cols = [
            "graph_image",
            "number_of_nodes",
            "question_level",
            "adaptation",
            "question",
            "source",
        ] + [c for c in feats if c in hdf.columns]
        frames.append(hdf[cols].copy())
        print(f"Loaded {model}: {len(hdf)} rows")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate_behaviour(path: Path) -> pd.DataFrame:
    trials = pd.read_csv(path)
    trials["graph_image"] = trials["graph_image"].map(norm_image)
    trials["is_correct"] = trials["is_correct"].astype(bool)
    trials["responded_in_time"] = trials["responded_in_time"].astype(bool)
    trials["rt_observed"] = _numeric(trials["rt_observed"])
    rows = []
    for img, g in trials.groupby("graph_image"):
        responded = g[g["responded_in_time"] == True]  # noqa: E712
        rows.append(
            {
                "graph_image": img,
                "n_trials": int(len(g)),
                "n_participants": int(g["participant"].nunique()),
                "response_rate": float(g["responded_in_time"].mean()),
                "accuracy": float(g["is_correct"].mean()),
                "rt_mean": float(responded["rt_observed"].mean()) if len(responded) else np.nan,
                "rt_median": float(responded["rt_observed"].median()) if len(responded) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def aggregate_et(path: Path) -> pd.DataFrame:
    et = pd.read_csv(path)
    et = et[et["modality"] == "et"].copy()
    et["graph_image"] = et["graph_image"].map(norm_image)
    if "qc_pass" in et.columns:
        et = et[et["qc_pass"] == True].copy()  # noqa: E712
    cols = ["et_n_fixations", "et_mean_fix_dur", "et_sum_fix_dur", "et_mean_saccade_mag"]
    for c in cols:
        et[c] = _numeric(et[c])
    out = (
        et.groupby("graph_image", as_index=False)[cols]
        .median()
        .rename(columns={c: c for c in cols})
    )
    out["n_et_trials"] = et.groupby("graph_image").size().reindex(out["graph_image"]).to_numpy()
    return out


def wide_features(feat_long: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """One row per stimulus, columns source__feature. Only real (non-empty) columns."""
    keys = ["graph_image", "number_of_nodes", "question_level", "adaptation", "question"]
    meta = feat_long[keys].drop_duplicates("graph_image")
    pairs: List[Tuple[str, str]] = []
    wide = meta.copy()
    for source, sub in feat_long.groupby("source"):
        value_cols = [
            c
            for c in sub.columns
            if c not in keys + ["source"] and c not in SKIP_FEATURES
        ]
        for col in value_cols:
            vals = _numeric(sub[col])
            if vals.notna().sum() < 3:
                continue
            name = f"{source}__{col}"
            piece = sub[["graph_image"]].copy()
            piece[name] = vals.to_numpy()
            wide = wide.merge(piece, on="graph_image", how="left")
            pairs.append((source, col))
    return wide, pairs


def add_fdr_by_group(df: pd.DataFrame, p_col: str, group_col: str, out_col: str) -> pd.DataFrame:
    df = df.copy()
    df[out_col] = np.nan
    for _, idx in df.groupby(group_col).groups.items():
        fam = pd.Index(idx)
        df.loc[fam, out_col] = benjamini_hochberg(df.loc[fam, p_col].to_numpy())
    return df


def stimulus_correlations(
    wide: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    outcomes: List[str],
    kind: str,
    residualize: bool,
) -> pd.DataFrame:
    rows = []
    X = design_matrix(wide) if residualize else None
    for source, feat in pairs:
        col = f"{source}__{feat}"
        if col not in wide.columns:
            continue
        x = _numeric(wide[col]).to_numpy()
        if residualize:
            x = ols_residuals(x, X)
        for outcome in outcomes:
            if outcome not in wide.columns:
                continue
            y = _numeric(wide[outcome]).to_numpy()
            if residualize:
                y = ols_residuals(y, X)
            r, p, n = spearman(x, y)
            if n < 5 or pd.isna(p):
                continue
            rows.append(
                {
                    "kind": kind,
                    "source": source,
                    "feature": feat,
                    "outcome": outcome,
                    "n": n,
                    "spearman_r": r,
                    "p_raw": p,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = add_fdr_by_group(out, "p_raw", "outcome", "p_fdr_outcome")
    out["sig_raw_0.05"] = out["p_raw"].lt(0.05)
    out["sig_fdr_outcome"] = out["p_fdr_outcome"].lt(0.05)
    return out


def pair_non_vs_full(wide: pd.DataFrame, pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """Matched non vs full by question text and node count."""
    non = wide[wide["adaptation"] == "non_adapted"].copy()
    full = wide[wide["adaptation"] == "fully_adapted"].copy()
    merged = non.merge(
        full,
        on=["question", "number_of_nodes"],
        suffixes=("_non", "_full"),
        how="inner",
    )
    rows = []
    delta_cols = []
    for outcome in HUMAN_BEHAV + HUMAN_ET:
        if f"{outcome}_non" not in merged.columns:
            continue
        merged[f"delta_{outcome}"] = _numeric(merged[f"{outcome}_non"]) - _numeric(
            merged[f"{outcome}_full"]
        )
        delta_cols.append(outcome)
    for source, feat in pairs:
        col = f"{source}__{feat}"
        c_non, c_full = f"{col}_non", f"{col}_full"
        if c_non not in merged.columns:
            continue
        feat_delta = _numeric(merged[c_non]) - _numeric(merged[c_full])
        for outcome in delta_cols:
            r, p, n = spearman(feat_delta.to_numpy(), merged[f"delta_{outcome}"].to_numpy())
            if n < 5 or pd.isna(p):
                continue
            rows.append(
                {
                    "kind": "paired_non_minus_full",
                    "source": source,
                    "feature": feat,
                    "outcome": f"delta_{outcome}",
                    "n": n,
                    "spearman_r": r,
                    "p_raw": p,
                    "mean_feature_delta": float(np.nanmean(feat_delta.to_numpy())),
                    "mean_outcome_delta": float(np.nanmean(merged[f"delta_{outcome}"].to_numpy())),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out, merged
    out = add_fdr_by_group(out, "p_raw", "outcome", "p_fdr_outcome")
    out["sig_raw_0.05"] = out["p_raw"].lt(0.05)
    out["sig_fdr_outcome"] = out["p_fdr_outcome"].lt(0.05)
    return out, merged


def clipgaze_vs_et(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat, et_col in CLIPGAZE_TO_ET:
        fcol = f"clipgaze__{feat}"
        if fcol not in wide.columns or et_col not in wide.columns:
            continue
        r, p, n = spearman(_numeric(wide[fcol]).to_numpy(), _numeric(wide[et_col]).to_numpy())
        if n >= 5 and pd.notna(p):
            rows.append(
                {
                    "kind": "clipgaze_vs_human_et",
                    "source": "clipgaze",
                    "feature": feat,
                    "outcome": et_col,
                    "n": n,
                    "spearman_r": r,
                    "p_raw": p,
                }
            )
        x = ols_residuals(_numeric(wide[fcol]).to_numpy(), design_matrix(wide))
        y = ols_residuals(_numeric(wide[et_col]).to_numpy(), design_matrix(wide))
        r2, p2, n2 = spearman(x, y)
        if n2 >= 5 and pd.notna(p2):
            rows.append(
                {
                    "kind": "clipgaze_vs_human_et_residual",
                    "source": "clipgaze",
                    "feature": feat,
                    "outcome": et_col,
                    "n": n2,
                    "spearman_r": r2,
                    "p_raw": p2,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = add_fdr_by_group(out, "p_raw", "kind", "p_fdr_outcome")
    out["sig_raw_0.05"] = out["p_raw"].lt(0.05)
    out["sig_fdr_outcome"] = out["p_fdr_outcome"].lt(0.05)
    return out


def condition_alignment(
    wide: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    isc: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Rank-correlation of 3 adaptation means: feature vs human RT/accuracy/ISC.
    N=3 is tiny — descriptive only, no p-value claims.
    """
    rows = []
    adapt_means: Dict[str, Dict[str, float]] = {}
    for level, g in wide.groupby("adaptation"):
        adapt_means[level] = {
            "rt_median": float(_numeric(g["rt_median"]).mean()),
            "accuracy": float(_numeric(g["accuracy"]).mean()),
            "response_rate": float(_numeric(g["response_rate"]).mean()),
        }
    isc_adapt: Dict[str, Dict[str, float]] = {}
    if isc is not None and len(isc):
        sub = isc[(isc["condition_factor"] == "adaptation")].copy()
        for _, r in sub.iterrows():
            sig = str(r["signal"])
            isc_adapt.setdefault(sig, {})[str(r["condition_level"])] = float(r["isc_mean_r"])

    human_series = {
        "rt_median": [adapt_means[a]["rt_median"] for a in ADAPT_ORDER],
        "accuracy": [adapt_means[a]["accuracy"] for a in ADAPT_ORDER],
        "response_rate": [adapt_means[a]["response_rate"] for a in ADAPT_ORDER],
    }
    for sig, d in isc_adapt.items():
        if all(a in d for a in ADAPT_ORDER):
            human_series[f"isc_{sig}"] = [d[a] for a in ADAPT_ORDER]

    for source, feat in pairs:
        col = f"{source}__{feat}"
        if col not in wide.columns:
            continue
        feat_means = []
        ok = True
        for a in ADAPT_ORDER:
            v = _numeric(wide.loc[wide["adaptation"] == a, col]).mean()
            if pd.isna(v):
                ok = False
                break
            feat_means.append(float(v))
        if not ok:
            continue
        for hname, hvals in human_series.items():
            r, p, n = spearman(np.array(feat_means), np.array(hvals))
            rows.append(
                {
                    "kind": "adaptation_mean_alignment_n3",
                    "source": source,
                    "feature": feat,
                    "outcome": hname,
                    "n": n,
                    "spearman_r": r,
                    "p_raw": p,
                    "mean_non": feat_means[0],
                    "mean_semi": feat_means[1],
                    "mean_full": feat_means[2],
                    "human_non": hvals[0],
                    "human_semi": hvals[1],
                    "human_full": hvals[2],
                }
            )
    return pd.DataFrame(rows)


def _fmt_p(p: float) -> str:
    if pd.isna(p):
        return "NA"
    return f"{p:.3g}"


def _fmt_r(r: float) -> str:
    if pd.isna(r):
        return "NA"
    return f"{r:+.3f}"


def write_block(lines: List[str], title: str, df: pd.DataFrame, use_fdr: bool) -> None:
    lines.append(title)
    if df is None or df.empty:
        lines.append("  none")
        lines.append("")
        return
    flag = "sig_fdr_outcome" if use_fdr and "sig_fdr_outcome" in df.columns else "sig_raw_0.05"
    sub = df[df[flag] == True].copy()  # noqa: E712
    if sub.empty:
        lines.append("  none")
        lines.append("")
        return
    sort_col = "p_fdr_outcome" if use_fdr and "p_fdr_outcome" in sub.columns else "p_raw"
    for _, r in sub.sort_values(sort_col).iterrows():
        p_fdr = r["p_fdr_outcome"] if "p_fdr_outcome" in sub.columns else np.nan
        lines.append(
            f"  {r['source']:12s} {r['feature']:24s} vs {r['outcome']:28s}  "
            f"r={_fmt_r(r['spearman_r'])}  p_raw={_fmt_p(r['p_raw'])}  "
            f"p_fdr={_fmt_p(p_fdr)}  n={int(r['n'])}"
        )
    lines.append("")


def write_report(
    out_path: Path,
    stim: pd.DataFrame,
    raw_corr: pd.DataFrame,
    partial_corr: pd.DataFrame,
    paired: pd.DataFrame,
    clip_et: pd.DataFrame,
    align: pd.DataFrame,
) -> None:
    lines = []
    lines.append("POST_ANALYSIS — IMAGE FEATURES vs HUMAN DATA (Sphynx-13)")
    lines.append("N participants = 13, N stimuli = 48. Spearman. FDR-BH within each outcome.")
    lines.append("")
    lines.append("WHY THIS SCRIPT EXISTS")
    lines.append("Behaviour already showed that full adaptation helps (faster, more accurate).")
    lines.append("Script 03 only showed that some image statistics differ by condition label.")
    lines.append("That can be true even if the feature has nothing to do with human load.")
    lines.append("Here: do feature values (and non−full deltas) track RT / accuracy / ET / ISC?")
    lines.append("")
    lines.append("Three questions:")
    lines.append("  1. Stimulus-level: on images with feature X, are people slower / less accurate?")
    lines.append("  2. Residual: after removing adaptation + nodes + question, does X still track RT?")
    lines.append("     If no, X is mostly restating the design. If yes, X captures extra image difficulty.")
    lines.append("  3. Paired benefit: for the same question, does a bigger feature change (non vs full)")
    lines.append("     go with a bigger RT/accuracy benefit?")
    lines.append("")
    if stim is not None and len(stim):
        lines.append(
            f"Stimuli joined: {len(stim)}  |  median RT {stim['rt_median'].median():.2f}s  |  "
            f"mean accuracy {stim['accuracy'].mean():.3f}"
        )
        lines.append("")

    lines.append("══ 1. STIMULUS-LEVEL (raw Spearman, FDR within outcome) ══")
    lines.append("This still confounds design: 12-node images have both more ink and slower RT.")
    write_block(lines, "", raw_corr, use_fdr=True)

    lines.append("══ 2. RESIDUAL (design partialled out: adaptation + 6/12 + Q1/Q2) ══")
    lines.append("This is the test of whether the feature is more than a condition-label echo.")
    write_block(lines, "", partial_corr, use_fdr=True)

    lines.append("══ 3. PAIRED NON vs FULL (same question + node count; FDR within outcome) ══")
    lines.append("delta = non − full. Positive RT delta = adaptation made people faster.")
    write_block(lines, "", paired, use_fdr=True)

    lines.append("══ CLIPGAZE vs HUMAN EYE-TRACKING ══")
    write_block(lines, "", clip_et, use_fdr=True)

    lines.append("══ ADAPTATION-MEAN ALIGNMENT (N=3 levels; descriptive only) ══")
    lines.append("Spearman of (non, semi, full) means. Do not treat p-values as confirmatory.")
    if align is not None and len(align):
        interesting = align[align["spearman_r"].abs() >= 0.99]
        if interesting.empty:
            lines.append("  no |r|=1 alignments")
        else:
            for _, r in interesting.sort_values("outcome").iterrows():
                lines.append(
                    f"  {r['source']:12s} {r['feature']:24s} vs {r['outcome']:20s}  "
                    f"r={_fmt_r(r['spearman_r'])}  feat {r['mean_non']:.4g}/{r['mean_semi']:.4g}/{r['mean_full']:.4g}"
                )
        lines.append("")
    else:
        lines.append("  none")
        lines.append("")

    lines.append("HOW TO READ THIS")
    lines.append("A useful adaptation-engine feature should (a) survive FDR in the paired test or")
    lines.append("the residual test, not only the raw stimulus correlation, and (b) move in the")
    lines.append("same direction as the human benefit (more marks / flatter saliency → faster RT).")
    lines.append("CLIPGaze vs human ET tests whether predicted scanpaths resemble real looking.")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare image features to Sphynx-13 human data")
    parser.add_argument("--trials", type=Path, default=TRIAL_CSV)
    parser.add_argument("--et", type=Path, default=ET_CSV)
    parser.add_argument("--isc", type=Path, default=ISC_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feat_long = load_features()
    if feat_long.empty:
        print("No feature CSVs found.")
        return
    if not args.trials.is_file():
        print(f"Missing trial CSV (read-only human data): {args.trials}")
        return

    beh = aggregate_behaviour(args.trials)
    print(f"Behaviour aggregates: {len(beh)} stimuli from {args.trials}")
    et = aggregate_et(args.et) if args.et.is_file() else pd.DataFrame()
    if len(et):
        print(f"ET aggregates: {len(et)} stimuli")
    isc = pd.read_csv(args.isc) if args.isc.is_file() else None
    if isc is not None:
        print(f"ISC timecourse table: {len(isc)} rows")

    wide, pairs = wide_features(feat_long)
    wide = wide.merge(beh, on="graph_image", how="left")
    if len(et):
        wide = wide.merge(et, on="graph_image", how="left")
    wide.to_csv(args.out_dir / "stimulus_joined.csv", index=False)

    outcomes = [c for c in HUMAN_BEHAV + HUMAN_ET if c in wide.columns]
    raw_corr = stimulus_correlations(wide, pairs, outcomes, "stimulus_raw", residualize=False)
    partial_corr = stimulus_correlations(wide, pairs, outcomes, "stimulus_residual", residualize=True)
    paired, merged_pairs = pair_non_vs_full(wide, pairs)
    if len(merged_pairs):
        merged_pairs.to_csv(args.out_dir / "paired_non_vs_full_stimuli.csv", index=False)
    clip_et = clipgaze_vs_et(wide)
    align = condition_alignment(wide, pairs, isc)

    raw_corr.to_csv(args.out_dir / "spearman_stimulus_raw.csv", index=False)
    partial_corr.to_csv(args.out_dir / "spearman_stimulus_residual.csv", index=False)
    if not paired.empty:
        paired.to_csv(args.out_dir / "spearman_paired_non_vs_full.csv", index=False)
    if not clip_et.empty:
        clip_et.to_csv(args.out_dir / "spearman_clipgaze_vs_et.csv", index=False)
    if not align.empty:
        align.to_csv(args.out_dir / "adaptation_mean_alignment.csv", index=False)

    report = args.out_dir / "feature_vs_human_report.txt"
    write_report(report, wide, raw_corr, partial_corr, paired, clip_et, align)
    print(f"Wrote {report}")
    print(
        f"  raw FDR hits={int(raw_corr['sig_fdr_outcome'].sum()) if len(raw_corr) else 0}  "
        f"residual FDR hits={int(partial_corr['sig_fdr_outcome'].sum()) if len(partial_corr) else 0}  "
        f"paired FDR hits={int(paired['sig_fdr_outcome'].sum()) if len(paired) else 0}"
    )


if __name__ == "__main__":
    main()
