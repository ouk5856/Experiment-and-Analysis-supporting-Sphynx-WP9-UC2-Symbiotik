# EEG ISC v2 — anti-alias (**canonical pipeline for `for_report`**)

Same as EEG ISC v2 ICA+FASTER, with **Nyquist-safe downsampling** for ISC:

1. ICA+FASTER on bandpassed EEG (1–40 Hz) — artifact cleaning.
2. GFP / occipital traces at native sampling rate.
3. **Zero-phase Butterworth lowpass at 12 Hz** (`ISC_ANTIALIAS_HZ`) — required before decimation to 25 Hz (Nyquist = 12.5 Hz).
4. Resample to 25 Hz → LOO ISC, Wilcoxon + FDR, circular-shift / sign-flip permutation.

**Theory:** Content above \(f_s/2\) must be lowpass-filtered before downsampling (Nyquist–Shannon). See `METHODS_SOURCES_LOG.md` §8c.

Outputs: `for_report/results/eeg_isc_v2_antialias/`  
Historical comparison (no step 3): `results/eeg_isc_v2/` + `ANTIALIAS_COMPARISON.md`

Does not modify the original `post_analysis/eeg_isc_v2/` tree.

## Run

From repo root:

```bash
python "All data/post_analysis/for_report/scripts/eeg_isc_v2_antialias/01_preprocess_eeg_ica_faster.py"
python "All data/post_analysis/for_report/scripts/eeg_isc_v2_antialias/02_run_eeg_isc_corrected.py"
python "All data/post_analysis/for_report/scripts/eeg_isc_v2_antialias/03_permutation_isc.py"
```

## Alternative (not used)

Raise ISC rate to ≥80 Hz so Nyquist ≥40 Hz — keeps beta-range content but changes all ISC timelines vs multimodal 25 Hz.
