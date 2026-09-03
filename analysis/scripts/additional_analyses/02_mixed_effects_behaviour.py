#!/usr/bin/env python3
"""
Mixed-effects models for behavioural data.

Addresses caveat: pooled tests treat trials as independent, inflating df.
Fits LMM for RT and logistic regression with clustered SEs for accuracy,
both with participant as grouping variable.

  python 02_mixed_effects_behaviour.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

BEH_PATH = ROOT / "All data/experiment_for_Sphynx_pilot/analysis_sphynx13/pilot_trial_level_with_accuracy.csv"
OUT_DIR = HERE / "results"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    beh = pd.read_csv(BEH_PATH)

    beh = beh[beh["graph_complexity"].notna()].copy()
    beh["rt_numeric"] = pd.to_numeric(beh["rt"], errors="coerce")
    beh["correct"] = beh["is_correct"].astype(int)
    beh["adapt"] = beh["adaptation"].astype(str)
    beh["nodes"] = beh["graph_nodes"].astype(int).astype(str)
    beh["qlevel"] = beh["question_level"].astype(int).astype(str)
    beh["pid"] = beh["participant"].astype(str)

    lines = ["MIXED-EFFECTS BEHAVIOURAL MODELS", "=" * 50, ""]
    coef_rows = []

    # --- RT LMM ---
    rt_data = beh.dropna(subset=["rt_numeric"]).copy()
    rt_data["rt_numeric"] = rt_data["rt_numeric"].astype(float)
    lines.append(f"RT model: N={len(rt_data)} trials, {rt_data.pid.nunique()} participants")

    # Main effects
    md_rt = smf.mixedlm(
        "rt_numeric ~ C(adapt, Treatment(reference='non_adapted')) + C(nodes) + C(qlevel)",
        rt_data,
        groups=rt_data["pid"],
    )
    mdf_rt = md_rt.fit(reml=True)
    lines.append("")
    lines.append("RT — main effects (reference: non_adapted, nodes=6, Q=1)")
    lines.append(str(mdf_rt.summary().tables[1]))
    for name, val in mdf_rt.fe_params.items():
        coef_rows.append({
            "model": "RT_main",
            "term": name,
            "coef": float(val),
            "se": float(mdf_rt.bse_fe[name]) if name in mdf_rt.bse_fe else np.nan,
            "z": float(mdf_rt.tvalues[name]) if name in mdf_rt.tvalues else np.nan,
            "p": float(mdf_rt.pvalues[name]) if name in mdf_rt.pvalues else np.nan,
        })

    # Interaction model
    lines.append("")
    lines.append("RT — adaptation x question interaction")
    md_rt_int = smf.mixedlm(
        "rt_numeric ~ C(adapt, Treatment(reference='non_adapted')) * C(qlevel) + C(nodes)",
        rt_data,
        groups=rt_data["pid"],
    )
    mdf_rt_int = md_rt_int.fit(reml=True)
    lines.append(str(mdf_rt_int.summary().tables[1]))
    for name, val in mdf_rt_int.fe_params.items():
        coef_rows.append({
            "model": "RT_interaction",
            "term": name,
            "coef": float(val),
            "se": float(mdf_rt_int.bse_fe[name]) if name in mdf_rt_int.bse_fe else np.nan,
            "z": float(mdf_rt_int.tvalues[name]) if name in mdf_rt_int.tvalues else np.nan,
            "p": float(mdf_rt_int.pvalues[name]) if name in mdf_rt_int.pvalues else np.nan,
        })

    # --- Accuracy: logistic with clustered SEs ---
    lines.append("")
    lines.append(f"Accuracy model (logit + clustered SEs): N={len(beh)} trials")

    acc_data = beh.copy()
    acc_data = pd.get_dummies(acc_data, columns=["adapt", "nodes", "qlevel"], drop_first=False)
    dep = acc_data["correct"]
    indep_cols = [
        "adapt_semi_adapted", "adapt_fully_adapted",
        "nodes_12", "qlevel_2",
    ]
    missing = [c for c in indep_cols if c not in acc_data.columns]
    if missing:
        lines.append(f"  Missing dummy columns: {missing}")
    else:
        X = acc_data[indep_cols].astype(float)
        X = sm.add_constant(X)
        try:
            logit = sm.Logit(dep, X)
            res = logit.fit(cov_type="cluster", cov_kwds={"groups": acc_data["pid"]}, disp=False)
            lines.append("")
            lines.append("Accuracy — logistic regression with clustered SEs (ref: non_adapted, 6-node, Q1)")
            lines.append(str(res.summary2().tables[1]))
            for name in res.params.index:
                coef_rows.append({
                    "model": "accuracy_clustered",
                    "term": name,
                    "coef": float(res.params[name]),
                    "se": float(res.bse[name]),
                    "z": float(res.tvalues[name]),
                    "p": float(res.pvalues[name]),
                })
        except Exception as e:
            lines.append(f"  Logistic model failed: {e}")

    # Accuracy interaction
    indep_int = indep_cols + ["adapt_semi_adapted_x_q2", "adapt_fully_adapted_x_q2"]
    acc_data["adapt_semi_adapted_x_q2"] = acc_data.get("adapt_semi_adapted", 0).astype(float) * acc_data.get("qlevel_2", 0).astype(float)
    acc_data["adapt_fully_adapted_x_q2"] = acc_data.get("adapt_fully_adapted", 0).astype(float) * acc_data.get("qlevel_2", 0).astype(float)
    X_int = acc_data[indep_int].astype(float)
    X_int = sm.add_constant(X_int)
    try:
        logit_int = sm.Logit(dep, X_int)
        res_int = logit_int.fit(cov_type="cluster", cov_kwds={"groups": acc_data["pid"]}, disp=False)
        lines.append("")
        lines.append("Accuracy — with adaptation x Q interaction")
        lines.append(str(res_int.summary2().tables[1]))
        for name in res_int.params.index:
            coef_rows.append({
                "model": "accuracy_interaction",
                "term": name,
                "coef": float(res_int.params[name]),
                "se": float(res_int.bse[name]),
                "z": float(res_int.tvalues[name]),
                "p": float(res_int.pvalues[name]),
            })
    except Exception as e:
        lines.append(f"  Interaction model failed: {e}")

    # Comparison note
    lines.append("")
    lines.append("COMPARISON TO POOLED TESTS")
    lines.append("- Mixed model accounts for within-participant clustering")
    lines.append("- If main effects remain significant, pooled tests were not misleading on direction")
    lines.append("- Check whether p-values increase (less significant) vs pooled — expected for N=13")

    coef_df = pd.DataFrame(coef_rows)
    coef_path = OUT_DIR / "mixed_model_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)

    report_path = OUT_DIR / "mixed_effects_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {coef_path} ({len(coef_df)} coefficients)")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
