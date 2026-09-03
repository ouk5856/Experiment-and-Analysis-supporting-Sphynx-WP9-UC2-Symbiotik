# Behaviour analysis (Sphynx-13)

## Two layers in this package

| Layer | Role | Location |
|-------|------|----------|
| **Primary (report claims)** | MixedLM RT + clustered logit accuracy; participant clustering | `scripts/additional_analyses/02_mixed_effects_behaviour.py` → `results/multimodal/mixed_*` |
| **Descriptive / historical** | Cell descriptives, pooled & per-participant tests, paired non vs full, text summaries | This folder + `results/behaviour/` |

Pooled significance tests in `pilot_significance_tests.csv` **inflate df** by treating trials as independent. Use them for exploration and narrative descriptives; **cite mixed models** for inferential behavioural claims (`FINAL_REPORT.md` §b.1, §b.6).

## Script (archival copy)

`pilot_significance_analysis_accuracy_rt_sphynx13.py` — original pipeline that wrote `analysis_sphynx13/` outputs.

**Run from repo root** (paths assume full repo layout):

```bash
python pilot_significance_analysis_accuracy_rt_sphynx13.py
# or from this copy, with --data-dir / --output-dir overrides pointing at
#   All data/experiment_for_Sphynx_pilot/exp data
#   and a chosen output folder
```

See `../../RUN_PATHS.md` §3 and `../../SOURCE_PATHS.md`.

## Results in `results/behaviour/`

| File | Contents |
|------|----------|
| `pilot_trial_level_with_accuracy.csv` | 624 trials: RT, accuracy, conditions |
| `pilot_descriptives_by_cell.csv` | Means/medians by condition cells |
| `pilot_significance_tests.csv` | Pooled / per-participant / pairwise tests (**historical**) |
| `pilot_paired_non_vs_full_by_question.csv` | Same-question non vs full |
| `pilot_incorrect_trials.csv`, `pilot_incorrect_by_cell.csv` | Error patterns |
| `sphynx13_summary.txt`, `sphynx13_short_report.txt` | Human-readable behaviour reports |
| `CROSS_ANALYSIS_SUMMARY.txt` | Early cross-layer narrative (behaviour + sensors) |
