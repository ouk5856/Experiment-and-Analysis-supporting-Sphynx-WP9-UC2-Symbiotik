# Additional analyses — addressing pilot caveats

**Package copies** under `for_report/scripts/additional_analyses/`.  
**Canonical outputs:** `for_report/results/multimodal/`.

## Which caveat each script addresses

| Script | Caveat | What it does |
|--------|--------|-------------|
| `01_isc_vs_behaviour.py` | ISC-behaviour comparison is qualitative (condition-level only) | Spearman correlation between **per-participant** ISC (LOO Fisher-z) and per-participant mean RT / accuracy; paired difference correlation; bootstrap 95% CIs |
| `02_mixed_effects_behaviour.py` | Behavioural tests pool trials as independent, inflating df | Mixed-effects models with participant random intercept: RT (LMM) and accuracy (logistic with clustered SEs); adaptation × question interaction |
| `03_multimodal_isc_corrected.py` | ET/EDA/pupil ISC contrasts have no correction or permutation | Re-extracts per-subject LOO ISC from v1 ET/physio epochs; applies FDR-BH + sign-flip permutation |
| `04_effect_sizes.py` | No Cohen's d or confidence intervals reported | Paired Cohen's d + bootstrap 95% CIs for key contrasts across all modalities |

## Run from this package (01 and 04 refreshed)

`01` and `04` resolve the repo root automatically and write to `for_report/results/multimodal/`.

| Script | Reads from (for_report) |
|--------|-------------------------|
| **01** | `results/eeg_isc_v2_antialias/analysisA_subject_loo_v2.csv`, `results/behaviour/pilot_trial_level_with_accuracy.csv` |
| **04** | Same anti-alias LOO + `results/multimodal/multimodal_subject_loo.csv` + behaviour trial CSV |

```bash
cd "All data/post_analysis/for_report/scripts/additional_analyses"
python 01_isc_vs_behaviour.py
python 04_effect_sizes.py
```

Historical archives (pre anti-alias refresh):  
`../for me/results/multimodal/isc_vs_behaviour_from_eeg_isc_v2_historical.csv`,  
`../for me/results/multimodal/effect_sizes_from_eeg_isc_v2_historical.csv`.

## Scripts 02–03 (heavier / original layout)

`02` and `03` still assume the original `post_analysis/additional_analyses/` path layout when re-run from the living tree. Prefer packaged CSVs in `results/multimodal/` for reporting. Multimodal contrasts use **Wilcoxon FDR + sign-flip** only (no circular-shift ISC-above-chance). See archive `for me/docs/RUN_PATHS.md`.

## Dependencies

- `numpy`, `pandas`, `scipy`, `statsmodels` (02)
- `03` also needs `mne`, v1 `pilot_eeg_et_isc_sphynx13.py`, and raw data (see `RUN_PATHS.md`)
