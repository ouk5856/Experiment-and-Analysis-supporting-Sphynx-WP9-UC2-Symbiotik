#!/usr/bin/env python3
"""
Sphynx pilot cohort (13 participants): RT, response rate, and accuracy tests.

Data folder (default):
  All data/experiment_for_Sphynx_pilot/exp data/<person>/*Practice Trials*.csv

RT: PsychoPy absolute-clock fix for ALL in this cohort
  (rt = keypress.rt - image.started). Unanswered trials have no RT
  (excluded from RT tests unless --impute-rt-timeout is set).

Conditions:
  - graph_nodes: 6 vs 12 (graph_complexity)
  - question_level: 1 vs 2 (question_complexity)
  - adaptation: non_adapted, fully_adapted, semi_adapted
      (semi = any visualisation_type other than non_adapted / fully_adapted)

Outcomes:
  - responded_in_time: keypress.keys present (not None / empty)
  - rt: corrected seconds; RT tests use responded trials only unless --impute-rt-timeout is set
  - is_correct: correct keypress; no keypress / wrong key = incorrect

Runs tests per participant, pooled, and pairwise (Bonferroni within each test family).
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_DATA_DIR = Path("All data/experiment_for_Sphynx_pilot/exp data")
DEFAULT_OUTPUT_DIR = Path("All data/experiment_for_Sphynx_pilot/analysis_sphynx13")
DEFAULT_ANSWER_KEY = Path("psychopy_experiment_stimuli_update_3_all_48_answered.csv")


def discover_practice_csvs(data_dir: Path) -> list[Path]:
    """One PsychoPy Practice Trials CSV per participant subfolder."""
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")
    found = sorted(data_dir.glob("*/*Practice Trials*.csv"))
    # Prefer one CSV per parent folder (first by name if multiples)
    by_parent: dict[Path, Path] = {}
    for p in found:
        parent = p.parent
        if parent not in by_parent:
            by_parent[parent] = p
    return sorted(by_parent.values(), key=lambda p: p.parent.name.lower())

ADAPTATION_ORDER = ["non_adapted", "semi_adapted", "fully_adapted"]
EXTREME_PAIRS_ALL = [
    ("non_adapted", "fully_adapted"),
    ("non_adapted", "semi_adapted"),
    ("semi_adapted", "fully_adapted"),
]
EXTREME_PAIRS_NON_FULL = [("non_adapted", "fully_adapted")]
FACTOR_COLS = {
    "graph_nodes": "graph_nodes",
    "question_level": "question_level",
    "adaptation": "adaptation",
}


def map_adaptation(visualisation_type: str) -> str:
    if visualisation_type in ("non_adapted", "fully_adapted"):
        return visualisation_type
    return "semi_adapted"


def load_trials(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    trials = df[df["graph_complexity"].notna()].copy()
    if trials.empty:
        return trials

    trials["source_file"] = csv_path.name
    if "participant" not in trials.columns or trials["participant"].isna().all():
        trials["participant"] = csv_path.stem
    else:
        trials["participant"] = trials["participant"].ffill().bfill()

    trials["graph_nodes"] = trials["graph_complexity"].astype(int)
    trials["question_level"] = trials["question_complexity"].astype(int)
    trials["adaptation"] = trials["visualisation_type"].map(map_adaptation)
    trials["semi_subtype"] = trials["visualisation_type"].where(
        trials["adaptation"] == "semi_adapted", other=pd.NA
    )

    keys = trials["keypress.keys"]
    key_str = keys.astype(str).str.strip().str.lower()
    trials["keypress_norm"] = keys.apply(normalize_keypress)
    trials["responded_in_time"] = trials["keypress_norm"].notna()
    keypress_rt = pd.to_numeric(trials["keypress.rt"], errors="coerce")
    image_started = pd.to_numeric(trials["image.started"], errors="coerce")
    # PsychoPy RT fix (all Sphynx cohort CSVs store absolute clocks in keypress.rt).
    trials["rt_observed"] = keypress_rt - image_started
    trials.loc[~trials["responded_in_time"], "rt_observed"] = np.nan
    trials["rt"] = trials["rt_observed"].copy()
    trials["question_id"] = trials["update_questions"].astype(str).str.strip()
    trials["folder_name"] = csv_path.parent.name

    return trials


def normalize_keypress(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("", "none", "nan", "n"):
        return None
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def normalize_answer(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower()
    if s in ("", "none", "nan"):
        return None
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def load_answer_key(path: Path) -> pd.DataFrame:
    ans = pd.read_csv(path)
    if "graph_image" not in ans.columns or "answer" not in ans.columns:
        raise ValueError("Answer key CSV must contain graph_image and answer columns")
    out = ans[["graph_image", "answer"]].copy()
    out["graph_image"] = out["graph_image"].astype(str).str.strip()
    out["answer_norm"] = out["answer"].apply(normalize_answer)
    return out


def attach_accuracy(trials: pd.DataFrame, answer_key: pd.DataFrame) -> pd.DataFrame:
    out = trials.copy()
    out["graph_image"] = out["graph_image"].astype(str).str.strip()
    out = out.merge(answer_key[["graph_image", "answer_norm"]], on="graph_image", how="left")
    out["has_answer_key"] = out["answer_norm"].notna()
    # Accuracy tests use all scorable trials; no response counts as incorrect.
    out["is_correct"] = False
    scorable = out["has_answer_key"]
    answered = scorable & out["keypress_norm"].notna()
    out.loc[answered, "is_correct"] = (
        out.loc[answered, "keypress_norm"] == out.loc[answered, "answer_norm"]
    )
    return out


def apply_rt_imputation(df: pd.DataFrame, timeout_seconds: float | None) -> pd.DataFrame:
    """Set rt to timeout_seconds when no in-time response or RT is missing."""
    out = df.copy()
    if timeout_seconds is None:
        out["rt_imputed"] = False
        return out
    miss = ~out["responded_in_time"] | out["rt"].isna()
    out.loc[miss, "rt"] = timeout_seconds
    out["rt_imputed"] = miss
    return out


def rt_suffix(impute_seconds: float | None) -> str:
    if impute_seconds is None:
        return ""
    return f"_imputed{int(impute_seconds)}s"


def bonferroni(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    m = len(p_values)
    return [min(1.0, p * m) for p in p_values]


def test_binary_two_groups(
    a: np.ndarray, b: np.ndarray, label: str, scope: str
) -> dict | None:
    """Fisher exact on 2x2 (responded vs not) for two groups."""
    if len(a) < 2 or len(b) < 2:
        return None
    table = np.array(
        [
            [int(a.sum()), int((~a.astype(bool)).sum())],
            [int(b.sum()), int((~b.astype(bool)).sum())],
        ]
    )
    if table.sum() == 0:
        return None
    _, p = stats.fisher_exact(table)
    return {
        "scope": scope,
        "outcome": "responded_in_time",
        "test": "fisher_exact",
        "comparison": label,
        "n_a": len(a),
        "n_b": len(b),
        "rate_a": float(a.mean()),
        "rate_b": float(b.mean()),
        "p_raw": float(p),
        "p_adj": np.nan,
    }


def test_binary_omnibus(groups: dict, label: str, scope: str) -> dict | None:
    levels = list(groups.keys())
    if len(levels) < 2:
        return None
    rows = []
    for lev in levels:
        g = groups[lev].astype(bool)
        rows.append([int(g.sum()), int((~g).sum())])
    table = np.array(rows)
    if table.sum() == 0 or table.shape[0] < 2:
        return None
    if table.shape[0] == 2:
        try:
            _, p = stats.fisher_exact(table)
        except ValueError:
            return None
        test_name = "fisher_exact"
    else:
        try:
            _, p, _, _ = stats.chi2_contingency(table)
        except ValueError:
            return None
        test_name = "chi2_contingency"
    return {
        "scope": scope,
        "outcome": "responded_in_time",
        "test": test_name,
        "comparison": label,
        "n_a": np.nan,
        "n_b": np.nan,
        "rate_a": np.nan,
        "rate_b": np.nan,
        "p_raw": float(p),
        "p_adj": np.nan,
    }


def test_rt_two_groups(
    a: np.ndarray,
    b: np.ndarray,
    label: str,
    scope: str,
    *,
    impute_seconds: float | None = None,
    use_responded_only: bool = False,
) -> dict | None:
    if use_responded_only:
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
    else:
        a = a.astype(float)
        b = b.astype(float)
    if len(a) < 2 or len(b) < 2:
        return None
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "scope": scope,
        "outcome": "rt",
        "test": "mannwhitneyu",
        "comparison": label,
        "n_a": len(a),
        "n_b": len(b),
        "rate_a": float(np.median(a)),
        "rate_b": float(np.median(b)),
        "p_raw": float(p),
        "p_adj": np.nan,
        "statistic": float(stat),
        "rt_imputation_s": impute_seconds if impute_seconds is not None else np.nan,
    }


def test_rt_omnibus(
    groups: dict,
    label: str,
    scope: str,
    *,
    impute_seconds: float | None = None,
    use_responded_only: bool = False,
) -> dict | None:
    if use_responded_only:
        arrays = [g[~np.isnan(g)] for g in groups.values()]
    else:
        arrays = [np.asarray(g, dtype=float) for g in groups.values()]
    if sum(len(x) >= 2 for x in arrays) < 2:
        return None
    if len(arrays) == 2:
        stat, p = stats.mannwhitneyu(arrays[0], arrays[1], alternative="two-sided")
        test_name = "mannwhitneyu"
    else:
        stat, p = stats.kruskal(*[x for x in arrays if len(x) >= 1])
        test_name = "kruskal"
    return {
        "scope": scope,
        "outcome": "rt",
        "test": test_name,
        "comparison": label,
        "n_a": np.nan,
        "n_b": np.nan,
        "rate_a": np.nan,
        "rate_b": np.nan,
        "p_raw": float(p),
        "p_adj": np.nan,
        "statistic": float(stat),
        "rt_imputation_s": impute_seconds if impute_seconds is not None else np.nan,
    }


def collect_rt_group(df: pd.DataFrame, mask: pd.Series, use_responded_only: bool) -> np.ndarray:
    sub = df.loc[mask]
    if use_responded_only:
        sub = sub[sub["responded_in_time"]]
    return sub["rt"].to_numpy(dtype=float)


def subset_by_factor(df: pd.DataFrame, factor: str, value) -> pd.DataFrame:
    return df[df[factor] == value]


def run_factor_tests(
    df: pd.DataFrame,
    factor: str,
    scope: str,
    *,
    impute_seconds: float | None,
    use_responded_only: bool,
) -> list[dict]:
    col = FACTOR_COLS[factor]
    results: list[dict] = []
    levels = sorted(df[col].dropna().unique())
    groups_resp = {lev: df.loc[df[col] == lev, "responded_in_time"].to_numpy() for lev in levels}
    groups_rt = {
        lev: collect_rt_group(df, df[col] == lev, use_responded_only) for lev in levels
    }
    sfx = rt_suffix(impute_seconds)

    label = f"{factor}_omnibus({','.join(map(str, levels))}){sfx}"
    r = test_binary_omnibus(groups_resp, label.replace(sfx, ""), scope)
    if r:
        results.append(r)
    r = test_rt_omnibus(
        groups_rt,
        label,
        scope,
        impute_seconds=impute_seconds,
        use_responded_only=use_responded_only,
    )
    if r:
        results.append(r)

    pairwise: list[dict] = []
    for lev_a, lev_b in combinations(levels, 2):
        la = f"{factor}:{lev_a}"
        lb = f"{factor}:{lev_b}"
        comp = f"{la}_vs_{lb}{sfx}"
        ra = groups_resp[lev_a]
        rb = groups_resp[lev_b]
        row = test_binary_two_groups(ra, rb, comp.replace(sfx, ""), scope)
        if row:
            pairwise.append(row)
        rta = groups_rt[lev_a]
        rtb = groups_rt[lev_b]
        row = test_rt_two_groups(
            rta,
            rtb,
            comp,
            scope,
            impute_seconds=impute_seconds,
            use_responded_only=use_responded_only,
        )
        if row:
            pairwise.append(row)

    # Bonferroni within pairwise family per outcome
    for outcome in ("responded_in_time", "rt"):
        idx = [i for i, row in enumerate(pairwise) if row["outcome"] == outcome]
        if not idx:
            continue
        raw = [pairwise[i]["p_raw"] for i in idx]
        adj = bonferroni(raw)
        for i, p_adj in zip(idx, adj):
            pairwise[i]["p_adj"] = p_adj

    results.extend(pairwise)
    return results


def run_accuracy_factor_tests(df: pd.DataFrame, factor: str, scope: str) -> list[dict]:
    df = df[df["has_answer_key"]]
    col = FACTOR_COLS[factor]
    levels = sorted(df[col].dropna().unique())
    if len(levels) < 2:
        return []
    groups = {lev: df.loc[df[col] == lev, "is_correct"].astype(bool).to_numpy() for lev in levels}
    results: list[dict] = []

    label = f"accuracy_{factor}_omnibus({','.join(map(str, levels))})"
    row = test_binary_omnibus(groups, label, scope)
    if row:
        row["outcome"] = "accuracy"
        results.append(row)

    pairwise: list[dict] = []
    for a, b in combinations(levels, 2):
        comp = f"accuracy_{factor}:{a}_vs_{factor}:{b}"
        r = test_binary_two_groups(groups[a], groups[b], comp, scope)
        if r:
            r["outcome"] = "accuracy"
            pairwise.append(r)

    idx = [i for i, r in enumerate(pairwise) if r["outcome"] == "accuracy"]
    if idx:
        adj = bonferroni([pairwise[i]["p_raw"] for i in idx])
        for i, p in zip(idx, adj):
            pairwise[i]["p_adj"] = p
    results.extend(pairwise)
    return results


def run_accuracy_extreme_tests(
    df: pd.DataFrame,
    scope: str,
    *,
    extreme_pairs: list[tuple[str, str]] | None = None,
) -> list[dict]:
    df = df[df["has_answer_key"]]
    pairs = extreme_pairs or EXTREME_PAIRS_ALL
    results: list[dict] = []
    for q in (None, 1, 2):
        sub = df if q is None else df[df["question_level"] == q]
        suffix = "" if q is None else f"_q{q}"
        for a, b in pairs:
            da = sub[sub["adaptation"] == a]["is_correct"].astype(bool).to_numpy()
            db = sub[sub["adaptation"] == b]["is_correct"].astype(bool).to_numpy()
            comp = f"accuracy_adapt:{a}_vs_{b}{suffix}"
            r = test_binary_two_groups(da, db, comp, scope)
            if r:
                r["outcome"] = "accuracy"
                results.append(r)
    idx = [i for i, r in enumerate(results) if r["outcome"] == "accuracy"]
    if idx:
        adj = bonferroni([results[i]["p_raw"] for i in idx])
        for i, p in zip(idx, adj):
            results[i]["p_adj"] = p
    return results


def run_paired_accuracy_non_full(df: pd.DataFrame, scope: str) -> list[dict]:
    rows: list[dict] = []
    for graph_filter, tag in ((None, "accuracy_paired_non_vs_full_all_graphs"), (12, "accuracy_paired_non_vs_full_12node")):
        pairs = build_question_aligned_pairs(df, graph_nodes=graph_filter)
        if pairs.empty:
            continue
        if "has_answer_key_non_adapted" in pairs.columns and "has_answer_key_fully_adapted" in pairs.columns:
            pairs = pairs[pairs["has_answer_key_non_adapted"] & pairs["has_answer_key_fully_adapted"]]
        if pairs.empty:
            continue
        need = ["is_correct_non_adapted", "is_correct_fully_adapted"]
        if any(c not in pairs.columns for c in need):
            continue
        non = pairs["is_correct_non_adapted"].astype(bool).to_numpy()
        full = pairs["is_correct_fully_adapted"].astype(bool).to_numpy()
        r = test_paired_mcnemar(non, full, tag, scope)
        if r:
            r["outcome"] = "accuracy"
            rows.append(r)
    return rows


def participant_extremes_only(
    participant: str, source_file: str, patterns: list[str] | None
) -> bool:
    if not patterns:
        return False
    hay = f"{participant} {source_file}".lower()
    return any(p.lower() in hay for p in patterns)


def filter_extremes_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["adaptation"].isin(["non_adapted", "fully_adapted"])].copy()


def run_extreme_pair_tests(
    df: pd.DataFrame,
    scope: str,
    *,
    impute_seconds: float | None,
    use_responded_only: bool,
    extreme_pairs: list[tuple[str, str]] | None = None,
    adaptation_order: list[str] | None = None,
) -> list[dict]:
    """Pairwise adaptation tests, overall and split by question_level."""
    results: list[dict] = []
    sfx = rt_suffix(impute_seconds)
    pairs = extreme_pairs or EXTREME_PAIRS_ALL
    for q in (None, 1, 2):
        sub = df if q is None else df[df["question_level"] == q]
        suffix = "" if q is None else f"_q{q}"
        for a, b in pairs:
            da = sub[sub["adaptation"] == a]
            db = sub[sub["adaptation"] == b]
            comp = f"adapt:{a}_vs_{b}{suffix}"
            row = test_binary_two_groups(
                da["responded_in_time"].to_numpy(),
                db["responded_in_time"].to_numpy(),
                comp,
                scope,
            )
            if row:
                results.append(row)
            if use_responded_only:
                rt_a = da.loc[da["responded_in_time"], "rt"].to_numpy(dtype=float)
                rt_b = db.loc[db["responded_in_time"], "rt"].to_numpy(dtype=float)
            else:
                rt_a = da["rt"].to_numpy(dtype=float)
                rt_b = db["rt"].to_numpy(dtype=float)
            row = test_rt_two_groups(
                rt_a,
                rt_b,
                f"{comp}{sfx}",
                scope,
                impute_seconds=impute_seconds,
                use_responded_only=use_responded_only,
            )
            if row:
                results.append(row)

    for outcome in ("responded_in_time", "rt"):
        idx = [i for i, r in enumerate(results) if r["outcome"] == outcome]
        if not idx:
            continue
        adj = bonferroni([results[i]["p_raw"] for i in idx])
        for i, p in zip(idx, adj):
            results[i]["p_adj"] = p
    return results


PAIR_KEY = ["participant", "question_id", "graph_nodes", "question_level"]


def build_question_aligned_pairs(
    df: pd.DataFrame, graph_nodes: int | None = None
) -> pd.DataFrame:
    """
    One row per matched question: same update_questions text and question_level,
    same graph size, non_adapted vs fully_adapted within participant.
    """
    sub = df[df["adaptation"].isin(["non_adapted", "fully_adapted"])].copy()
    if graph_nodes is not None:
        sub = sub[sub["graph_nodes"] == graph_nodes]

    idx_cols = PAIR_KEY
    wide = sub.pivot_table(
        index=idx_cols,
        columns="adaptation",
        values=["responded_in_time", "rt", "is_correct", "has_answer_key"],
        aggfunc="first",
    )
    wide.columns = [f"{val}_{adapt}" for val, adapt in wide.columns]
    wide = wide.reset_index()
    need = ["responded_in_time_non_adapted", "responded_in_time_fully_adapted"]
    wide = wide.dropna(subset=need)
    if graph_nodes is not None:
        wide["pair_filter"] = f"graph_{graph_nodes}"
    else:
        wide["pair_filter"] = "all_graphs"
    return wide


def mcnemar_exact_p(b: int, c: int) -> float | None:
    """Two-sided exact McNemar p-value from discordant counts b and c."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    return float(min(1.0, 2 * stats.binomtest(k, n, 0.5).pvalue))


def test_paired_mcnemar(
    non: np.ndarray, full: np.ndarray, label: str, scope: str
) -> dict | None:
    """McNemar exact test on paired responded (1) vs missed (0)."""
    if len(non) < 2:
        return None
    # discordant pairs: b = non=1 full=0, c = non=0 full=1
    b = int(np.sum(non & ~full))
    c = int(np.sum(~non & full))
    p_two = mcnemar_exact_p(b, c)
    if p_two is None:
        return None
    return {
        "scope": scope,
        "outcome": "responded_in_time",
        "test": "mcnemar_exact",
        "comparison": label,
        "n_pairs": len(non),
        "n_a": int(non.sum()),
        "n_b": int(full.sum()),
        "rate_a": float(non.mean()),
        "rate_b": float(full.mean()),
        "discordant_non_only": b,
        "discordant_full_only": c,
        "p_raw": p_two,
        "p_adj": np.nan,
        "statistic": float(b + c),
    }


def test_paired_wilcoxon(
    non: np.ndarray,
    full: np.ndarray,
    label: str,
    scope: str,
    *,
    impute_seconds: float | None = None,
) -> dict | None:
    """Wilcoxon signed-rank on RT (non - full) for all pairs with finite RT."""
    non = np.asarray(non, dtype=float)
    full = np.asarray(full, dtype=float)
    mask = ~(np.isnan(non) | np.isnan(full))
    non = non[mask]
    full = full[mask]
    if len(non) < 2:
        return None
    diff = non - full
    if np.all(diff == 0):
        return None
    stat, p = stats.wilcoxon(non, full, alternative="two-sided")
    return {
        "scope": scope,
        "outcome": "rt",
        "test": "wilcoxon_signed_rank",
        "comparison": label,
        "n_pairs": len(non),
        "n_a": len(non),
        "n_b": len(full),
        "rate_a": float(np.median(non)),
        "rate_b": float(np.median(full)),
        "median_diff_non_minus_full": float(np.median(diff)),
        "p_raw": float(p),
        "p_adj": np.nan,
        "statistic": float(stat),
        "rt_imputation_s": impute_seconds if impute_seconds is not None else np.nan,
    }


def run_paired_non_full_tests(
    df: pd.DataFrame,
    scope: str,
    *,
    impute_seconds: float | None,
    use_responded_only: bool,
) -> tuple[list[dict], pd.DataFrame]:
    """Question-aligned paired non vs fully; all graphs and 12-node only."""
    results: list[dict] = []
    pair_frames: list[pd.DataFrame] = []

    rt_sfx = rt_suffix(impute_seconds)
    for graph_filter, tag in ((None, "paired_non_vs_full_all_graphs"), (12, "paired_non_vs_full_12node")):
        pairs = build_question_aligned_pairs(df, graph_nodes=graph_filter)
        if pairs.empty:
            continue
        pairs["scope"] = scope
        pair_frames.append(pairs)

        non_resp = pairs["responded_in_time_non_adapted"].astype(bool).to_numpy()
        full_resp = pairs["responded_in_time_fully_adapted"].astype(bool).to_numpy()
        row = test_paired_mcnemar(non_resp, full_resp, tag, scope)
        if row:
            results.append(row)

        if use_responded_only:
            rt_pairs = pairs[
                pairs["responded_in_time_non_adapted"] & pairs["responded_in_time_fully_adapted"]
            ]
            rt_label = f"{tag}_rt_both_responded"
        else:
            rt_pairs = pairs
            rt_label = f"{tag}_rt{rt_sfx}"

        if len(rt_pairs) >= 2:
            non_rt = rt_pairs["rt_non_adapted"].to_numpy(dtype=float)
            full_rt = rt_pairs["rt_fully_adapted"].to_numpy(dtype=float)
            row = test_paired_wilcoxon(
                non_rt,
                full_rt,
                rt_label,
                scope,
                impute_seconds=impute_seconds,
            )
            if row:
                results.append(row)

    paired_df = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    return results, paired_df


def run_two_way_tests(
    df: pd.DataFrame,
    scope: str,
    *,
    impute_seconds: float | None,
    use_responded_only: bool,
    adaptation_order: list[str] | None = None,
) -> list[dict]:
    """adaptation x question_level and adaptation x graph_nodes (omnibus chi2 / kruskal per slice)."""
    results: list[dict] = []
    sfx = rt_suffix(impute_seconds)
    adapt_levels = adaptation_order or ADAPTATION_ORDER
    for other, name in (("question_level", "adapt_x_question"), ("graph_nodes", "adapt_x_graph")):
        for olev in sorted(df[other].unique()):
            sub = df[df[other] == olev]
            groups_resp = {
                a: sub.loc[sub["adaptation"] == a, "responded_in_time"].to_numpy()
                for a in adapt_levels
                if (sub["adaptation"] == a).any()
            }
            groups_rt = {
                a: collect_rt_group(sub, sub["adaptation"] == a, use_responded_only)
                for a in adapt_levels
                if (sub["adaptation"] == a).any()
            }
            if len(groups_resp) < 2:
                continue
            label = f"{name}_{other}={olev}{sfx}"
            r = test_binary_omnibus(groups_resp, label.replace(sfx, ""), scope)
            if r:
                results.append(r)
            r = test_rt_omnibus(
                groups_rt,
                label,
                scope,
                impute_seconds=impute_seconds,
                use_responded_only=use_responded_only,
            )
            if r:
                results.append(r)
    return results


def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(["participant", "adaptation", "graph_nodes", "question_level"]):
        part, adapt, gn, ql = keys
        n = len(g)
        resp = g["responded_in_time"]
        rt_obs = g.loc[g["responded_in_time"], "rt_observed"]
        rt_all = g["rt"]
        rows.append(
            {
                "participant": part,
                "adaptation": adapt,
                "graph_nodes": gn,
                "question_level": ql,
                "n_trials": n,
                "n_responded": int(resp.sum()),
                "response_rate": float(resp.mean()) if n else np.nan,
                "rt_mean_observed": float(rt_obs.mean()) if len(rt_obs) else np.nan,
                "rt_median_observed": float(rt_obs.median()) if len(rt_obs) else np.nan,
                "rt_mean_analysis": float(rt_all.mean()) if n else np.nan,
                "rt_median_analysis": float(rt_all.median()) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def analyze_scope(
    df: pd.DataFrame,
    scope: str,
    *,
    impute_seconds: float | None,
    use_responded_only: bool,
    extremes_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict] = []
    rt_kw = {"impute_seconds": impute_seconds, "use_responded_only": use_responded_only}
    extreme_kw = {
        **rt_kw,
        "extreme_pairs": EXTREME_PAIRS_NON_FULL if extremes_only else None,
    }
    two_way_kw = {
        **rt_kw,
        "adaptation_order": ["non_adapted", "fully_adapted"] if extremes_only else None,
    }
    acc_extreme_kw = {
        "extreme_pairs": EXTREME_PAIRS_NON_FULL if extremes_only else None,
    }
    for factor in ("graph_nodes", "question_level", "adaptation"):
        all_rows.extend(run_factor_tests(df, factor, scope, **rt_kw))
        all_rows.extend(run_accuracy_factor_tests(df, factor, scope))
    all_rows.extend(run_extreme_pair_tests(df, scope, **extreme_kw))
    all_rows.extend(run_accuracy_extreme_tests(df, scope, **acc_extreme_kw))
    all_rows.extend(run_two_way_tests(df, scope, **two_way_kw))
    paired_rows, paired_df = run_paired_non_full_tests(df, scope, **rt_kw)
    all_rows.extend(paired_rows)
    all_rows.extend(run_paired_accuracy_non_full(df, scope))
    if not all_rows:
        return pd.DataFrame(), paired_df
    out = pd.DataFrame(all_rows)
    out["significant_0.05"] = out["p_adj"].fillna(out["p_raw"]) < 0.05
    return out.sort_values(["outcome", "comparison"]), paired_df


def print_summary(desc: pd.DataFrame, tests: pd.DataFrame, scope: str) -> None:
    print("\n" + "=" * 72)
    print(f"SCOPE: {scope}")
    print("=" * 72)
    print("\nCell descriptives (rate = responded in time; rt among responders):")
    if desc.empty:
        print("  (no data)")
    else:
        show_desc = desc if scope != "pooled" else desc
        print(show_desc.to_string(index=False, max_rows=60))

    if tests.empty:
        print("\nNo statistical tests (insufficient data).")
        return

    sig = tests[tests["significant_0.05"]]
    print(f"\nSignificant results (α=0.05, Bonferroni for pairwise families): {len(sig)}")
    if len(sig):
        cols = ["outcome", "test", "comparison", "p_raw", "p_adj", "rate_a", "rate_b", "n_a", "n_b"]
        print(sig[cols].to_string(index=False))

    print("\nAll tests (sorted by p_adj / p_raw):")
    cols = ["outcome", "test", "comparison", "p_raw", "p_adj", "significant_0.05"]
    show = tests.sort_values("p_adj", na_position="last")
    print(show[cols].to_string(index=False, max_rows=80))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sphynx 13-participant pilot significance analysis (RT + accuracy)"
    )
    parser.add_argument(
        "csvs",
        nargs="*",
        type=Path,
        help="PsychoPy CSV paths (default: auto-discover in DEFAULT_DATA_DIR)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"Folder of participant subfolders (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=f"Directory for CSV outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--impute-rt-timeout",
        type=float,
        default=None,
        metavar="SEC",
        help="Sensitivity analysis: set RT to SEC (e.g. 18) when no keypress or RT missing. "
        "Default: off — RT tests only include trials with a keypress.",
    )
    parser.add_argument(
        "--extremes-only-substr",
        nargs="*",
        default=None,
        metavar="TEXT",
        help="For participants/files whose id/path contains TEXT (case-insensitive), "
        "exclude semi_adapted and test only non_adapted vs fully_adapted.",
    )
    parser.add_argument(
        "--write-summary",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write a text summary report to PATH (default: <output-dir>/sphynx13_summary.txt).",
    )
    parser.add_argument(
        "--answer-key-csv",
        type=Path,
        default=DEFAULT_ANSWER_KEY,
        metavar="PATH",
        help="CSV with graph_image and answer columns for accuracy scoring.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or (script_dir / DEFAULT_DATA_DIR)
    if not data_dir.is_absolute():
        data_dir = script_dir / data_dir

    if args.csvs:
        csv_paths = [p if p.is_absolute() else script_dir / p for p in args.csvs]
        csv_paths = [p for p in csv_paths if p.suffix.lower() == ".csv" and not p.name.startswith(".")]
    else:
        csv_paths = discover_practice_csvs(data_dir)

    if len(csv_paths) == 0:
        raise SystemExit(f"No Practice Trials CSVs found under {data_dir}")

    frames = [load_trials(p) for p in csv_paths]
    if not frames or all(f.empty for f in frames):
        raise SystemExit("No trial rows found in input CSVs.")

    impute_seconds = args.impute_rt_timeout
    use_responded_only = impute_seconds is None
    extremes_patterns = args.extremes_only_substr
    answer_key_path = args.answer_key_csv
    if not answer_key_path.is_absolute():
        answer_key_path = script_dir / answer_key_path
    answer_key = load_answer_key(answer_key_path)

    processed: list[pd.DataFrame] = []
    for p, raw in zip(csv_paths, frames):
        part = str(raw["participant"].iloc[0])
        extremes = participant_extremes_only(part, p.name, extremes_patterns)
        t = apply_rt_imputation(raw, impute_seconds)
        t = attach_accuracy(t, answer_key)
        t["extremes_only_analysis"] = extremes
        t["analysis_note"] = (
            "non_adapted vs fully_adapted only (semi excluded)"
            if extremes
            else "all adaptation levels"
        )
        if extremes:
            t = filter_extremes_only(t)
        processed.append(t)

    combined = pd.concat(processed, ignore_index=True)
    analyze_kw = {"impute_seconds": impute_seconds, "use_responded_only": use_responded_only}

    out_dir = args.output_dir or (script_dir / DEFAULT_OUTPUT_DIR)
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.write_summary or (out_dir / "sphynx13_summary.txt")
    if not summary_path.is_absolute():
        summary_path = script_dir / summary_path

    n_imputed = int(combined.get("rt_imputed", pd.Series(dtype=bool)).sum())
    print("Loaded trials:")
    print(f"  Participants: {len(processed)} CSVs from {data_dir if not args.csvs else 'CLI paths'}")
    print("  RT: keypress.rt - image.started (PsychoPy clock fix)")
    if impute_seconds is not None:
        print(f"  RT imputation: {n_imputed} trials set to {impute_seconds}s (no keypress or missing RT)")
    else:
        print("  RT imputation: off (responded trials only for RT tests)")
    if extremes_patterns:
        print(f"  Extremes-only filter when id/path contains: {extremes_patterns}")
    for p, t in zip(csv_paths, processed):
        folder = t["folder_name"].iloc[0] if "folder_name" in t.columns else p.parent.name
        print(
            f"  [{folder}] {p.name}: n={len(t)}, participant={t['participant'].iloc[0]}, "
            f"resp={int(t['responded_in_time'].sum())}/{len(t)}"
        )
        print(f"    mode: {t['analysis_note'].iloc[0]}")
        print(f"    adaptation counts:\n{t['adaptation'].value_counts().to_string()}")

    desc = descriptive_table(combined)

    test_frames = []
    paired_frames = []
    for part, g in combined.groupby("participant"):
        eo = bool(g["extremes_only_analysis"].iloc[0])
        tdf, pdf = analyze_scope(g, scope=str(part), extremes_only=eo, **analyze_kw)
        test_frames.append(tdf)
        if not pdf.empty:
            paired_frames.append(pdf)
    tdf, pdf = analyze_scope(combined, scope="pooled", extremes_only=False, **analyze_kw)
    test_frames.append(tdf)
    if not pdf.empty:
        paired_frames.append(pdf)
    tests = pd.concat(test_frames, ignore_index=True)
    paired_all = pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame()

    desc_path = out_dir / "pilot_descriptives_by_cell.csv"
    tests_path = out_dir / "pilot_significance_tests.csv"
    paired_path = out_dir / "pilot_paired_non_vs_full_by_question.csv"
    trial_acc_path = out_dir / "pilot_trial_level_with_accuracy.csv"
    incorrect_path = out_dir / "pilot_incorrect_trials.csv"
    incorrect_cell_path = out_dir / "pilot_incorrect_by_cell.csv"
    desc.to_csv(desc_path, index=False)
    tests.to_csv(tests_path, index=False)
    combined.to_csv(trial_acc_path, index=False)
    incorrect_trials = build_incorrect_trials_export(combined)
    incorrect_trials.to_csv(incorrect_path, index=False)
    build_incorrect_summary_by_cell(combined).to_csv(incorrect_cell_path, index=False)
    if not paired_all.empty:
        paired_all.to_csv(paired_path, index=False)

    print_summary(desc, tests[tests["scope"] == "pooled"], "pooled")
    for part in combined["participant"].unique():
        d = desc[desc["participant"] == part]
        t = tests[tests["scope"] == part]
        print_summary(d, t, str(part))

    print("\n" + "-" * 72)
    print("NOTES")
    print("-" * 72)
    if impute_seconds is not None:
        rt_note = (
            f"- RT tests include ALL trials; no keypress / missing RT imputed as {impute_seconds}s.\n"
        )
    else:
        rt_note = "- RT tests use only trials with a keypress (responded in time).\n"
    print(
        rt_note
        + "- semi_adapted = shape_nodes, size_nodes, color_nodes, edge_width, etc.\n"
        + "- Pooled tests ignore participant; n is small — treat p-values as exploratory.\n"
        + "- Pairwise p_adj uses Bonferroni within each outcome × scope family.\n"
        + "- Extreme pairs: non↔full, non↔semi, semi↔full; also split by question_level 1/2.\n"
        + "- Question-aligned paired non vs full: match on update_questions + question_level\n"
        + "  + graph_nodes (McNemar response; Wilcoxon RT on all aligned pairs when imputing)."
    )
    rt_sig = tests[(tests["outcome"] == "rt") & (tests["significant_0.05"])]
    if not rt_sig.empty:
        print("\nSignificant RT tests:")
        print(
            rt_sig[["scope", "comparison", "rate_a", "rate_b", "n_a", "n_b", "p_raw", "rt_imputation_s"]]
            .to_string(index=False)
        )
    paired_tests = tests[tests["test"].isin(["mcnemar_exact", "wilcoxon_signed_rank"])]
    if not paired_tests.empty:
        print("\n" + "=" * 72)
        print("QUESTION-ALIGNED PAIRED: non_adapted vs fully_adapted")
        print("=" * 72)
        cols = [
            "scope",
            "outcome",
            "comparison",
            "n_pairs",
            "rate_a",
            "rate_b",
            "p_raw",
            "significant_0.05",
        ]
        print(paired_tests[cols].to_string(index=False))
    print(f"\nWrote: {desc_path}\n       {tests_path}\n       {trial_acc_path}")
    print(f"       {incorrect_path} ({len(incorrect_trials)} incorrect trials)")
    print(f"       {incorrect_cell_path}")
    if not paired_all.empty:
        print(f"       {paired_path}")

    write_analysis_summary(
        summary_path,
        csv_paths=csv_paths,
        processed=processed,
        desc=desc,
        tests=tests,
        paired_all=paired_all,
        extremes_patterns=extremes_patterns,
        impute_seconds=impute_seconds,
    )
    print(f"       {summary_path}")


INCORRECT_EXPORT_COLS = [
    "participant",
    "graph_nodes",
    "question_level",
    "adaptation",
    "semi_subtype",
    "error_type",
    "keypress_norm",
    "answer_norm",
    "responded_in_time",
    "rt_observed",
    "update_questions",
    "graph_image",
]


def build_incorrect_trials_export(df: pd.DataFrame) -> pd.DataFrame:
    """Trials scored incorrect (wrong answer or no response) among scorable items."""
    sub = df[df["has_answer_key"] & ~df["is_correct"]].copy()
    if sub.empty:
        return pd.DataFrame(columns=INCORRECT_EXPORT_COLS)
    sub["error_type"] = np.where(
        sub["responded_in_time"], "wrong_answer", "no_response"
    )
    out = sub[INCORRECT_EXPORT_COLS].sort_values(
        ["graph_nodes", "question_level", "adaptation"]
    )
    return out


def build_incorrect_summary_by_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Error counts by design cell for quick visual inspection."""
    scorable = df[df["has_answer_key"]].copy()
    scorable["error_type"] = np.where(
        scorable["is_correct"],
        "correct",
        np.where(scorable["responded_in_time"], "wrong_answer", "no_response"),
    )
    rows = []
    for keys, g in scorable.groupby(
        ["participant", "graph_nodes", "question_level", "adaptation"], dropna=False
    ):
        part, gn, ql, adapt = keys
        n = len(g)
        n_correct = int((g["error_type"] == "correct").sum())
        n_wrong = int((g["error_type"] == "wrong_answer").sum())
        n_miss = int((g["error_type"] == "no_response").sum())
        rows.append(
            {
                "participant": part,
                "graph_nodes": int(gn),
                "question_level": int(ql),
                "adaptation": adapt,
                "n_trials": n,
                "n_correct": n_correct,
                "n_wrong_answer": n_wrong,
                "n_no_response": n_miss,
                "n_incorrect": n_wrong + n_miss,
                "accuracy": n_correct / n if n else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["participant", "graph_nodes", "question_level", "adaptation"]
    )


def _rate_table(df: pd.DataFrame) -> str:
    rows = []
    for (gn, adapt), g in df.groupby(["graph_nodes", "adaptation"]):
        n = len(g)
        r = g["responded_in_time"].mean()
        rows.append(f"  {int(gn)}-node {adapt:14s}: {r:.0%} responded ({int(g['responded_in_time'].sum())}/{n})")
    return "\n".join(rows) if rows else "  (no data)"


def write_analysis_summary(
    path: Path,
    *,
    csv_paths: list[Path],
    processed: list[pd.DataFrame],
    desc: pd.DataFrame,
    tests: pd.DataFrame,
    paired_all: pd.DataFrame,
    extremes_patterns: list[str] | None,
    impute_seconds: float | None,
) -> None:
    """Write human-readable summary for a pilot batch (RT + accuracy)."""
    combined = pd.concat(processed, ignore_index=True)
    lines: list[str] = []
    lines.append("SYMBIOTIK SPHYNX-13 PILOT ANALYSIS SUMMARY (RT + ACCURACY)")
    lines.append("=" * 72)
    lines.append("")
    lines.append("DATA FILES")
    for p in csv_paths:
        lines.append(f"  - [{p.parent.name}] {p.name}")
    lines.append("")
    lines.append("SETTINGS")
    lines.append("  - Match answers by graph_image to answer-key CSV")
    lines.append("  - keypress.keys in {None, n, empty} treated as no answer")
    lines.append("  - Accuracy tests: all scorable trials; no response counts as incorrect")
    lines.append("  - RT = keypress.rt - image.started (PsychoPy clock fix for all participants)")
    if impute_seconds is not None:
        lines.append(f"  - RT imputation: {impute_seconds}s for misses")
    else:
        lines.append("  - RT imputation: off (RT tests use responded trials only)")
    if extremes_patterns:
        lines.append(
            f"  - Extremes-only filter for ids/path containing {extremes_patterns}: "
            "semi_adapted excluded"
        )
    lines.append("")
    lines.append(f"TOTAL TRIALS ANALYSED: {len(combined)}")
    lines.append(f"PARTICIPANTS: {combined['participant'].nunique()}")
    lines.append(f"Trials with matched answer key: {int(combined['has_answer_key'].sum())}/{len(combined)}")
    lines.append("")

    for t in processed:
        part = t["participant"].iloc[0]
        lines.append(f"PARTICIPANT: {part}")
        lines.append(f"  Mode: {t['analysis_note'].iloc[0]}")
        lines.append(f"  Response rate: {t['responded_in_time'].mean():.1%} ({int(t['responded_in_time'].sum())}/{len(t)})")
        acc = t[t["has_answer_key"]]
        if len(acc):
            lines.append(f"  Accuracy (all trials): {acc['is_correct'].mean():.1%} ({int(acc['is_correct'].sum())}/{len(acc)})")
            ans = acc[acc["responded_in_time"]]
            if len(ans):
                lines.append(
                    f"  Accuracy (answered only): {ans['is_correct'].mean():.1%} "
                    f"({int(ans['is_correct'].sum())}/{len(ans)})"
                )
        lines.append("  Response rate by graph x adaptation:")
        lines.append(_rate_table(t))
        scope_sig = tests[(tests["scope"] == part) & (tests["significant_0.05"])]
        if len(scope_sig):
            lines.append("  Significant tests:")
            for _, r in scope_sig.sort_values("p_raw").iterrows():
                lines.append(f"    - {r['outcome']}: {r['comparison']} p={r['p_raw']:.4f}")
        else:
            lines.append("  Significant tests: none")
        lines.append("")

    pooled = tests[tests["scope"] == "pooled"]
    lines.append("POOLED SIGNIFICANT TESTS")
    sig = pooled[pooled["significant_0.05"]].sort_values("p_raw")
    if len(sig):
        for _, r in sig.iterrows():
            lines.append(f"  - {r['outcome']}: {r['comparison']} p={r['p_raw']:.4f}")
    else:
        lines.append("  - none")
    lines.append("")

    lines.append("PAIRED non_adapted vs fully_adapted (same question/image conditions)")
    paired = tests[tests["test"].isin(["mcnemar_exact", "wilcoxon_signed_rank"])]
    for scope in paired["scope"].unique():
        lines.append(f"  [{scope}]")
        sub = paired[paired["scope"] == scope]
        for _, r in sub.iterrows():
            lines.append(
                f"    {r['outcome']:10s} {r['comparison']:44s} "
                f"p={r['p_raw']:.3f} n={r.get('n_pairs', r.get('n_a', ''))}"
            )
    lines.append("")

    lines.append("=" * 72)
    lines.append("DESIGN BULLET SUMMARY")
    lines.append("=" * 72)
    lines.append("- Cohort: 13 Sphynx pilot participants from exp data folder.")
    lines.append("- RT corrected via keypress.rt - image.started for all CSVs.")
    lines.append("- Unanswered trials excluded from RT; count as incorrect for accuracy.")
    lines.append("- Pooled tests ignore participant clustering; treat as exploratory.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
