# Methods and sources log

**Purpose:** Justify each analysis technique used in the Sphynx-13 pilot, cite standard references, point to our code, and record audit notes.  
**Step-ordered walkthrough (report package root):** [`../../ANALYSIS_STEPS_METHODS.md`](../../ANALYSIS_STEPS_METHODS.md)  
**D9.4 results brief:** [`../../D9_4_WP9_UC2_RESULTS.md`](../../D9_4_WP9_UC2_RESULTS.md)  
**Companions in this archive:** [`AUDIT_FINDINGS.md`](../audit/AUDIT_FINDINGS.md), [`METHODS_SOUNDNESS_CITATIONS_v1_reference.txt`](METHODS_SOUNDNESS_CITATIONS_v1_reference.txt).

For each method: **what we did → why justified → sources → where in code → audit note**.

---

## 0. External ISC methodology — alignment check

This section maps our pipeline to commonly cited ISC implementations found during report preparation.

### Primary alignment: C-PAC / BrainIAK leave-one-out timecourse ISC

**Reference (web):** [C-PAC documentation — Inter-subject Correlation (ISC)](https://fcp-indi.github.io/docs/latest/user/group_isc)

C-PAC (adapted from [BrainIAK](http://brainiak.org)) defines LOO ISC as:

> For each subject, compute the correlation of that subject’s timecourse with the **mean timecourse of all other subjects**; average across subjects. Use **phase randomization** for a null distribution.

**Our implementation (identical core logic):**

For subject \(i\), condition \(c\), signal \(s\):

\[
\bar{x}_{-i,c}(t) = \frac{1}{N-1}\sum_{j \neq i} \bar{x}_{j,c,s}(t), \quad
r_{i,c,s} = \mathrm{corr}\big(\bar{x}_{i,c,s}(t),\, \bar{x}_{-i,c,s}(t)\big)
\]

Group summary via Fisher-\(z\):

\[
\bar{r}_{c,s} = \tanh\!\left(\frac{1}{N}\sum_i \operatorname{artanh}(r_{i,c,s})\right)
\]

| Element | C-PAC / BrainIAK | Our pipeline |
|---------|------------------|--------------|
| LOO group-mean correlation | Yes | Yes (`leave_one_out_isc`) |
| Stimulus-locked epochs | Yes | Yes (S4 → 18 s) |
| Null: destroy temporal alignment | Phase randomization | Circular time shift per subject |
| Condition / group comparison | On ISC maps or ROIs | Wilcoxon + FDR + sign-flip on paired LOO \(z\) |
| Spatial summary | ROI or voxel timecourses | **Univariate** GFP / occipital / gaze / EDA traces |

**Conclusion:** Our ISC **estimator and inference philosophy align with C-PAC/BrainIAK LOO timecourse ISC**. This is the strongest external validation for what we did.

**Supporting references cited by C-PAC:** Hasson et al. (2004); Kauppi et al. (2010); Simony et al. (2016); Nastase et al. (2019 review tradition). See [C-PAC ISC references](https://fcp-indi.github.io/docs/latest/user/group_isc).

### Related but different: Parra lab CorrCA ISC

**References (web):**

- [Parra Lab — Inter Subject Correlation in EEG](https://www.parralab.org/isc/)
- [ML-D00M / ISC-Inter-Subject-Correlations (CorrCA implementation)](https://github.com/ML-D00M/ISC-Inter-Subject-Correlations)

Parra-lab ISC uses **Correlated Component Analysis (CorrCA)** on full **multichannel** EEG (`time × electrodes × subjects`) to find spatial filters that maximize inter-subject correlation, then computes ISC on those components. Nulls often use **Fourier phase scrambling** (similar in purpose to our circular-shift null).

| | CorrCA (Parra / ML-D00M) | Our pipeline |
|---|--------------------------|--------------|
| Input | Multichannel EEG | Univariate timecourses (GFP, occ, gaze, EDA) |
| Spatial optimization | Yes (CorrCA weights \(W\)) | No — prespecified summaries |
| LOO correlation logic | Often applied after component extraction | Direct LOO Pearson on 1D traces |
| Same scientific question? | Shared stimulus-locked dynamics | Yes |
| Same algorithm? | No — we use a **simpler LOO timecourse ISC** tier | |

**Conclusion:** Same **family** of analysis (inter-subject synchrony under shared stimulation), but we do **not** implement CorrCA. For the report, cite C-PAC LOO ISC as the direct methodological parallel; cite Parra lab as related multichannel ISC literature.

### Suggested methods sentence (report-ready)

> Inter-subject correlation was computed as leave-one-out Pearson correlation between each participant’s condition-averaged timecourse and the mean timecourse of all other participants, with group ISC summarized via Fisher-\(z\) averaging, following the LOO ISC framework described in C-PAC/BrainIAK documentation ([link](https://fcp-indi.github.io/docs/latest/user/group_isc)). Significance was assessed with circular-shift nulls for **EEG ISC levels** and sign-flip permutations for **condition contrasts** (EEG and multimodal), analogous to phase-randomization approaches in the ISC literature. Multimodal ET/EDA do not include circular-shift level tests.

---

## 1. Leave-one-out (LOO) timecourse ISC

**What we did.** For each condition, average each subject’s trial timecourses, then correlate subject *i* with the mean of all other subjects; report mean LOO *r* (via Fisher-z).

**Why justified.** Standard LOO timecourse ISC for stimulus-locked group designs — explicitly matches [C-PAC’s ISC definition](https://fcp-indi.github.io/docs/latest/user/group_isc). Classic estimator of shared dynamics without an explicit stimulus regressor (Hasson et al., 2004; Nastase et al., 2019).

**Sources (with links).**

- **C-PAC / BrainIAK LOO ISC (primary alignment):** https://fcp-indi.github.io/docs/latest/user/group_isc
- Hasson, U., et al. (2004). Intersubject synchronization of cortical activity during natural vision. *Science*, 303(5664), 1634–1640. https://doi.org/10.1126/science.1089506
- Nastase, S. A., et al. (2019). Measuring shared responses across subjects using intersubject correlation. *Social Cognitive and Affective Neuroscience*, 14(6), 667–685. https://doi.org/10.1093/scan/nsz037
- Kauppi, J.-P., et al. (2010). Inter-subject correlation of brain hemodynamic responses during watching a movie. *Frontiers in Neuroinformatics*, 4, 5. (phase-scramble null tradition; cited by C-PAC)
- **Related (CorrCA, not our method):** https://www.parralab.org/isc/ ; https://github.com/ML-D00M/ISC-Inter-Subject-Correlations

**Where.** [`scripts/eeg_isc_v2/sphynx13_eeg_utils.py`](scripts/eeg_isc_v2/sphynx13_eeg_utils.py) `leave_one_out_isc`; anti-alias variant in `scripts/eeg_isc_v2_antialias/`; multimodal via `03_multimodal_isc_corrected.py`.

**Audit note.** LOO scores within a condition are cross-correlated. Core estimator aligns with C-PAC; we added permutation/FDR beyond exploratory Wilcoxon.

---

## 2. Fisher *z* transform for correlations

**What we did.** Convert Pearson *r* to artanh(*r*) before averaging or testing; map means back with tanh.

**Why justified.** Correlation coefficients are not on an additive scale; Fisher *z* is the standard transform for averaging and parametric/paired tests on *r*.

**Sources.**

- Fisher, R. A. (1915/1921). Frequency distribution of the values of the correlation coefficient… *Biometrika* / related works.
- Silver, N. C., & Dunlap, W. P. (1987). Averaging correlation coefficients: should Fisher’s *z* transformation be used? *Journal of Applied Psychology* (discussion of practice).

**Where.** `fisher_z` in `sphynx13_eeg_utils.py`; contrasts on `loo_fisher_z` in `02_run_eeg_isc_corrected.py`, `03_permutation_isc.py`, `03_multimodal_isc_corrected.py`.

**Audit note.** Standard. Clipping |*r*| < 1 for numerical stability is fine.

---

## 3. Paired Wilcoxon signed-rank on LOO Fisher-*z*

**What we did.** For each subject, difference of LOO *z* between two condition levels; Wilcoxon signed-rank.

**Why justified.** Nonparametric paired test; robust with small N and non-Gaussian *z* differences.

**Sources.**

- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*.
- Common in ISC pilot contrasts when permutation infrastructure is limited (Nastase et al., 2019 note permutation as preferred).

**Where.** `02_run_eeg_isc_corrected.py`; also computed in `03_multimodal_isc_corrected.py` alongside permutation *p*.

**Audit note.** Acceptable (pilot). Primary EEG claims additionally require FDR and sign-flip agreement.

---

## 4. FDR-BH (Benjamini–Hochberg)

**What we did.** Adjust *p*-values within `condition_factor` families (primary); also report global FDR and Bonferroni.

**Why justified.** Controls false discovery rate under multiple related contrasts (GFP + occipital × several pairs).

**Sources.**

- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS B*, 57(1), 289–300.

**Where.** `benjamini_hochberg`, `apply_pvalue_corrections` in `sphynx13_eeg_utils.py`; image-feature scripts `03`/`04`; multimodal `03`.

**Audit note.** Standard. FDR does not fix LOO dependence or trial non-independence in behaviour.

---

## 5. Circular time-shift null (ISC level)

**What we did.** Independently circularly shift each subject’s condition-mean timecourse, recompute mean LOO ISC, build null (default 5000).

**Why justified.** Destroys cross-subject temporal alignment while preserving each subject’s autocorrelation — standard nonparametric null for “ISC > chance.” Analogous to **phase randomization** nulls recommended in [C-PAC ISC documentation](https://fcp-indi.github.io/docs/latest/user/group_isc) and Parra-lab / ML-D00M ISC pipelines (Fourier phase scrambling).

**Sources (with links).**

- **C-PAC ISC null (phase randomization):** https://fcp-indi.github.io/docs/latest/user/group_isc
- Nastase et al. (2019) — ISC inference and null models: https://doi.org/10.1093/scan/nsz037
- [Parra Lab ISC](https://www.parralab.org/isc/) — phase-scramble surrogate data (CorrCA context)
- [ML-D00M ISC repo](https://github.com/ML-D00M/ISC-Inter-Subject-Correlations) — phase scramble → surrogate ISC null (step5)
- Kauppi, J.-P., et al. (2010). *Frontiers in Neuroinformatics*, 4, 5.

**Where.** `circular_shift_isc_null` in `sphynx13_eeg_utils.py`; EEG `03_permutation_isc.py` (anti-alias / v2). **Not** used in multimodal `03` (contrasts use sign-flip only).

**Audit note.** Standard. Shift of 0 is allowed with small probability; negligible at large *n_perm*.

---

## 6. Sign-flip (paired permutation) null for contrasts

**What we did.** For paired Fisher-*z* differences, randomly multiply by ±1, recompute mean difference; two-sided (or one-sided for GFP non > adapted).

**Why justified.** Exact (or Monte Carlo) permutation test for a paired mean difference under exchangeability of sign under the null.

**Sources.**

- Good, P. (2005). *Permutation, Parametric and Bootstrap Tests of Hypotheses*. Springer.
- Ernst, M. D. (2004). Permutation methods: a basis for exact inference. *Statistical Science*.
- Common paired-permutation practice in neuroimaging group contrasts.

**Where.** `sign_flip_contrast_null`; EEG `03_permutation_isc.py`; multimodal `03_multimodal_isc_corrected.py`.

**Audit note.** Standard for paired mean diffs. Applied to LOO-derived *z*; still report LOO dependence caveat.

---

## 7. FASTER + ICA artifact cleaning

**What we did.** Bandpass 1–40 Hz, notch 50 Hz, average reference; FASTER bad channels/epochs/components/per-epoch channels; ICA (FastICA, ≤20 comps); metrics without `eog_correlation` (no EOG channel).

**Why justified.** Automated EEG cleaning reducing reliance on hard peak-to-peak rejection alone; ICA is standard for ocular/muscle artifact removal when EOG available — here we use non-EOG FASTER component metrics.

**Sources.**

- Nolan, H., Whelan, R., & Reilly, R. B. (2010). FASTER: Fully Automated Statistical Thresholding for EEG artifact Rejection. *Journal of Neuroscience Methods*, 192(1), 152–162.
- mne-faster: https://github.com/wmvanvliet/mne-faster (Van Vliet).
- Hyvärinen, A., & Oja, E. (2000). Independent component analysis: algorithms and applications. *Neural Networks* (ICA background).
- MNE-Python ICA documentation (Gramfort et al., MNE ecosystem).

**Where (canonical).** `scripts/eeg_isc_v2_antialias/01_preprocess_eeg_ica_faster.py` (anti-alias package). Historical twin: `scripts/eeg_isc_v2/01_preprocess_eeg_ica_faster.py`.

**Audit note.** Standard toolkit, but **sparse component rejection** and **no EOG** are caveats. Retention 97% vs v1 63%.

---

## 8. Global Field Power (GFP) and occipital ROI

### 8a. Global Field Power (GFP)

**What GFP is.** At each time sample \(t\), GFP summarizes how much the scalp EEG **topography varies in amplitude** across channels — a reference-free index of overall field strength. In our code:

\[
\mathrm{GFP}(t) = \mathrm{std}_{c \in \text{scalp}}\{ x_c(t) \}
\]

(i.e. standard deviation across EEG channels at time \(t\), equivalent up to scaling to the classic “global field power” / global field strength used in topographic ERP analysis).

**Why use GFP for ISC?** It collapses multichannel EEG into one **stimulus-locked timecourse per subject**, enabling the same LOO timecourse ISC used for gaze and EDA. GFP is widely used when a single global summary of brain activity is needed without committing to one electrode or source.

**Sources (with links).**

- **Lehmann, D., & Skrandies, W. (1980).** Reference-free identification of components of cognitive brain potentials. *International Journal of Psychology*, 15(1–4), 127–128. (introduces reference-free GFP-style summaries)
- **Skrandies, W. (1990).** Global field power and topographic similarity. *Brain Topography*, 3(1), 137–141. https://doi.org/10.1007/BF01128870 (GFP definition and use)
- **Michel, C. M., & Murray, M. M. (2012).** Towards the utilization of EEG as a brain imaging technique. *NeuroImage*, 61(2), 371–385. https://doi.org/10.1016/j.neuroimage.2011.12.018 (GFP / topographic EEG imaging context)
- **MNE-Python `Evoked.gfp()`:** https://mne.tools/stable/generated/mne.Evoked.html#mne.Evoked.gfp (standard tooling; our pipeline computes the same quantity from epoch data via channel SD)

**Audit note.** GFP is **not** source localization and **not** CorrCA — it is a deliberate univariate summary for LOO ISC (see §0). Residual ocular/muscle artifact can still affect GFP; ICA+FASTER mitigates but does not eliminate (see §7).

### 8b. Occipital ROI (O1, Oz, O2)

**What ROI is here.** A **region of interest (ROI)** is a prespecified subset of electrodes chosen to approximate activity from a brain area — here, **visual/occipital cortex** during graph viewing. We use the mean of international 10–20 electrodes **O1, Oz, O2**:

\[
x_{\mathrm{occ}}(t) = \frac{1}{3}\big(x_{O1}(t) + x_{Oz}(t) + x_{O2}(t)\big)
\]

**Why justified.** Occipital electrodes are standard proxies for visual processing in ERP/EEG paradigms with visual stimuli; complements GFP (global) with a **visual-cortex–focused** trace for the same LOO ISC framework.

**Sources (with links).**

- **International 10–20 system (electrode placement):** Jasper, H. H. (1958). Report of the committee on methods of clinical examination in electroencephalography. *Electroencephalography and Clinical Neurophysiology*, 10, 370–375. (O1/Oz/O2 occipital locations)
- **Visual ERP / occipital activity:** Luck, S. J. (2014). *An Introduction to the Event-Related Potential Technique* (2nd ed.). MIT Press. (occipital electrodes for visual evoked activity — standard ERP textbook)
- **Topographic / ROI EEG practice:** Michel & Murray (2012), as above — https://doi.org/10.1016/j.neuroimage.2011.12.018

**Audit note.** Coarse ROI (3 electrodes); occipital ISC pattern differed from GFP (nuanced adaptation story). Do not over-interpret as localized source activity.

### 8c. Resampling to 25 Hz for ISC — anti-alias (canonical)

**Theory (why anti-alias is required).** After bandpass 1–40 Hz, GFP/occ timecourses are still defined at native EEG rate (\(f_s \approx 250\) Hz). Downsampling to ISC rate \(f_{\mathrm{ISC}} = 25\) Hz sets Nyquist frequency \(f_N = f_{\mathrm{ISC}}/2 = 12.5\) Hz. By the **Nyquist–Shannon sampling theorem**, any content above \(f_N\) must be removed **before** decimation or it **aliases** into \([0, f_N]\) and contaminates LOO correlations. Standard DSP practice: apply a lowpass with cutoff \(f_c < f_N\), then resample (Oppenheim & Schafer, *Discrete-Time Signal Processing*).

**What we did (canonical in `for_report`).**

1. ICA+FASTER on 1–40 Hz EEG (cleaning bandwidth — unchanged).
2. Compute GFP / occipital traces at native rate.
3. **Zero-phase 4th-order Butterworth lowpass at 12 Hz** (`ISC_ANTIALIAS_HZ = 12`, margin below 12.5 Hz Nyquist).
4. Linear resample to 450 samples / 18 s (25 Hz).

**Why 12 Hz.** Preserves ISC-relevant slow dynamics (theta/alpha and below) while removing beta/gamma band energy that cannot be represented faithfully at 25 Hz.

**Historical note.** Original `eeg_isc_v2` skipped step 3 (linear interp only). Kept in `results/eeg_isc_v2/` for sensitivity comparison; adaptation/node GFP claims survived, but the **anti-alias pipeline is methodologically preferred** for the report.

**Sources.**

- Oppenheim, A. V., & Schafer, R. W. *Discrete-Time Signal Processing* (anti-aliasing / decimation).
- Smith, J. O. *Digital Audio Resampling* (CCRMA) — practical resampling theory.
- Comparison in this package: `results/eeg_isc_v2_antialias/ANTIALIAS_COMPARISON.md`.

**Where.** `scripts/eeg_isc_v2_antialias/sphynx13_eeg_utils.py` (`lowpass_1d`, `resample_1d_antialias`); `01_preprocess_eeg_ica_faster.py`.

**Audit note.** **Standard / theory-supported.** This is the **canonical EEG ISC preprocessing** for `for_report` claims. Original v2 without lowpass = documented risk only.

---

## 9. Mixed-effects / clustered SEs for behaviour

**What we did.** RT: linear mixed model with participant random intercept. Accuracy: logistic regression with cluster-robust SEs by participant. Interaction models for adaptation × question.

**Why justified.** Repeated trials within participants violate independence; mixed models / clustered SEs are standard remedies.

**Sources.**

- Baayen, R. H., Davidson, D. J., & Bates, D. M. (2008). Mixed-effects modeling with crossed random effects for subjects and items. *Journal of Memory and Language*.
- Barr, D. J., et al. (2013). Random effects structure for confirmatory hypothesis testing. *JML*.
- Liang, K.-Y., & Zeger, S. L. (1986). Longitudinal data analysis using generalized linear models (GEE / clustered inference lineage).
- statsmodels MixedLM / Logit documentation.

**Where (primary).** `scripts/additional_analyses/02_mixed_effects_behaviour.py` → `results/multimodal/mixed_*`.

**Historical descriptives / pooled tests (not primary claims).** Script `scripts/behaviour/pilot_significance_analysis_accuracy_rt_sphynx13.py`; outputs in `results/behaviour/` (descriptives, `pilot_significance_tests.csv`, text reports). Pooled Mann–Whitney / Kruskal treat trials as independent — superseded for inference by mixed models (§9 primary).

**Audit note.** Acceptable (pilot). Convergence warnings on RT MixedLM; accuracy is clustered logit not binomial GLMM.

---

## 10. Spearman correlation + bootstrap CIs (ISC ↔ behaviour)

**What we did.** Rank correlation between subject LOO ISC and mean RT/accuracy; 5000 bootstrap CIs; also paired-difference correlations.

**Why justified.** Nonparametric association for small N; bootstrap CIs avoid fragile parametric CI assumptions.

**Sources.**

- Spearman, C. (1904). The proof and measurement of association between two things.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.

**Where.** `scripts/additional_analyses/01_isc_vs_behaviour.py` → `results/multimodal/isc_vs_behaviour.csv` (anti-alias LOO). Historical archive: `isc_vs_behaviour_from_eeg_isc_v2_historical.csv`.

**Audit note.** Acceptable (pilot). Exploratory; many tests without FDR.

---

## 11. Paired Cohen’s *d* + bootstrap CI on mean differences

**What we did.** For paired condition contrasts, *d* = mean(diff) / SD(diff); bootstrap percentile CI on mean difference.

**Why justified.** Effect sizes and uncertainty beyond *p*-values; APA / open-science reporting norms.

**Sources.**

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
- Lakens, D. (2013). Calculating and reporting effect sizes… *Frontiers in Psychology*.

**Where.** `scripts/additional_analyses/04_effect_sizes.py` → `results/multimodal/effect_sizes.csv` (EEG domain `eeg_isc_v2_antialias`). Historical archive: `effect_sizes_from_eeg_isc_v2_historical.csv`.

**Audit note.** Standard reporting practice.

---

## 12. Image-feature condition tests and human comparison

**What we did.** Kruskal/Mann–Whitney by condition with FDR; Spearman feature ↔ human outcomes (raw, residualized, paired non−full).

**Why justified.** Stimulus QA (do features track design factors?) separate from predictive human alignment.

**Sources.**

- Benjamini & Hochberg (1995) for FDR.
- Residualization / partialling as confound control (standard regression residual practice; interpret cautiously if adaptation signal is removed with confounds).

**Where.** `scripts/image_features/03_*.py`, `04_*.py`.

**Audit note.** Stimulus-check claims OK with FDR. Proxy claims not supported.

---

## Citation checklist for the report

| Claim theme | Primary method sources |
|-------------|------------------------|
| LOO timecourse ISC (core method) | **[C-PAC ISC docs](https://fcp-indi.github.io/docs/latest/user/group_isc)**; Hasson 2004; Nastase 2019 |
| ISC null / permutation | C-PAC (phase randomization); our circular-shift + sign-flip; Kauppi 2010 |
| Related multichannel ISC (not our method) | [Parra Lab ISC](https://www.parralab.org/isc/); [ML-D00M CorrCA repo](https://github.com/ML-D00M/ISC-Inter-Subject-Correlations) |
| GFP definition and use | Skrandies 1990; Michel & Murray 2012; [MNE Evoked.gfp](https://mne.tools/stable/generated/mne.Evoked.html#mne.Evoked.gfp) |
| Occipital ROI (O1/Oz/O2) | 10–20 system; Luck 2014 (visual ERP); Michel & Murray 2012 |
| Condition differences in ISC | Nastase 2019; Madsen & Parra 2024 |
| EEG cleaning | Nolan 2010 FASTER; MNE / ICA literature |
| Multiple testing | Benjamini & Hochberg 1995 |
| Permutation / sign-flip | Good 2005; Ernst 2004 |
| Behaviour clustering | Baayen 2008; Barr 2013 |
| Anti-alias before 25 Hz ISC downsample | Oppenheim & Schafer; `METHODS_SOURCES_LOG.md` §8c; `RUN_PATHS.md` |
| Effect sizes | Cohen 1988; Lakens 2013 |
| When FDR is / is not applied | **`FINAL_REPORT.md` §b.6** (complete test-by-test justification) |
