# Analysis steps — what, why, how justified, and how to verify

**Purpose:** For each step in the WP9 UC2 pipeline (behaviour → ISC → image features → human comparison), state **what** was done, **why**, **how it is justified / correct**, and **which sources / result files** verify it.

**Companion results brief:** [`D9_4_WP9_UC2_REPORT_SUMMARY.md`](D9_4_WP9_UC2_REPORT_SUMMARY.md)  
**Deep citations & technique bank (archive):** [`docs/METHODS_SOURCES_LOG.md`](/docs/METHODS_SOURCES_LOG.md)  
---

## Step overview

| Step | Question asked | Primary claim layer |
|------|----------------|---------------------|
| 1. Behaviour | Does adaptation (and secondarily nodes/Q) change RT/accuracy? | Mixed models + clustered logit |
| 2. Multimodal ISC | Do shared sensor dynamics differ by the same conditions? | LOO ISC + FDR + permutation |
| 3. Image features | Do stimuli differ by condition on computed features? | KW/MW + FDR within factor |
| 4. Features vs human | Can features proxy behaviour / ISC / ET? | Residual + paired tests + FDR |

---

## Step 1 — Behavioural analysis (RT and accuracy)

### What happened

- Trials coded for RT (keypress − image onset) and accuracy (unanswered = incorrect).
- Descriptives by adaptation / nodes / question cells.
- **Primary inference:** linear mixed model for RT with participant random intercept; logistic regression with **cluster-robust SEs** by participant for accuracy; interaction models for adaptation × question.

### Why

The design is within-subject with many trials per person. The scientific claim is that **visual adaptation** changes performance; nodes and question level are design factors that must be accounted for and that help interpret semi vs full (Q1 vs Q2 asymmetry).

### Why this is justified / correct

- Treating 624 trials as independent (pooled Mann–Whitney / Kruskal alone) **inflates df** — superseded for claims.
- MixedLM / clustered SEs are standard remedies for repeated measures (Baayen et al., 2008; Barr et al., 2013; Liang–Zeger clustered inference lineage).
- Few **planned** fixed effects → FDR on model coefficients optional; the critical fix is **clustering**, not FDR

### Sources to verify

| Item | Location |
|------|----------|
| Literature / method notes | `/docs/METHODS_SOURCES_LOG.md` §9 |
| Script (primary) | `scripts/additional_analyses/02_mixed_effects_behaviour.py` |
| Script (descriptives / historical) | `scripts/behaviour/pilot_significance_analysis_accuracy_rt_sphynx13.py` |
| Results | `results/multimodal/mixed_model_coefficients.csv`, `mixed_effects_report.txt`, `effect_sizes.csv` |
| Descriptives | `results/behaviour/pilot_descriptives_by_cell.csv`, `sphynx13_summary.txt` |

---

## Step 2 — Partition, preprocess, ISC (EEG, eye-tracking, aux)

### What happened

1. **Partition** epochs by condition (adaptation, nodes, question) from PsychoPy-aligned sensor streams.
2. **Preprocess**
   - EEG (canonical): ICA + FASTER cleaning; GFP and occipital summaries; **12 Hz lowpass then resample to 25 Hz** (anti-alias).
   - ET / EDA / pupil / PPG: condition-averaged timecourses (v1 cleaning path for multimodal script).
3. **ISC:** leave-one-out (LOO) Pearson correlation of each subject’s condition mean timecourse with the mean of all others; Fisher-*z* averaging; condition contrasts via paired Wilcoxon + **FDR within factor** + sign-flip permutation. **ISC-above-chance (circular-shift nulls) applies to EEG** in the anti-alias permutation scripts; multimodal ET/EDA uses Wilcoxon FDR + sign-flip contrasts only (no circular-shift level tests).

### Why

Behaviour shows *whether* people perform better. ISC asks whether **shared stimulus-locked dynamics** across participants also change with adaptation / difficulty — a complementary group-level signature (harder → often more similar dynamics).

### Why this is justified / correct

- LOO timecourse ISC matches the **C-PAC / BrainIAK** definition ([C-PAC ISC docs](https://fcp-indi.github.io/docs/latest/user/group_isc)); classic lineage Hasson 2004, Nastase 2019 — see `/docs/METHODS_SOURCES_LOG.md` §0–1.
- Fisher-*z* before averaging/testing correlations — standard.
- Paired nonparametric contrasts at **participant** level match the LOO unit of analysis.
- FDR within factor because many related ISC contrasts; permutation complements Wilcoxon.
- Anti-alias lowpass before 25 Hz downsample is **Nyquist–Shannon** DSP, not a hypothesis test.
- Not Parra CorrCA: we use univariate GFP/ROI/gaze/EDA for multimodal comparability.

### Sources to verify

| Item | Location |
|------|----------|
| Method + citations | `docs/METHODS_SOURCES_LOG.md` §0–8c |
| EEG scripts / results | `scripts/eeg_isc_v2_antialias/`, `results/eeg_isc_v2_antialias/` |
| Multimodal script / results | `scripts/additional_analyses/03_multimodal_isc_corrected.py`, `results/multimodal/multimodal_isc_contrasts.csv`, `multimodal_isc_report.txt`, `multimodal_subject_loo.csv` |


---

## Step 3 — Image feature extraction per condition

### What happened

- Low- and high-level features computed on the 48 stimuli (digital ink / layout metrics; SUM (non/semi-contextual saliency generator of high performance); VisSalFormer (saliency generator with contextual support trained on graph and UI Q/A tasks); CLIPGaze (contextual scanpath generator of high performance, not trained on graph Q/A), etc.).
- Features tested for differences across adaptation, nodes, and question (Kruskal–Wallis / Mann–Whitney) with **FDR within factor**. **Note**: Contextual models were given the same questions asked to participants on the same stimuli input.

### Why

If stimuli are constructed differently by condition, measurable image properties should reflect that (stimulus QA). Separately, features are candidates for **automatic proxies** of human difficulty — tested in Step 4, not assumed here.

### Why this is justified / correct

- Nonparametric tests fit skewed feature distributions; FDR within factor controls multiplicity across many features × levels (`METHODS_SOURCES_LOG.md` §12 in archive).
- “Feature differs by label” is a **stimulus check**, not a human claim.

### Sources to verify

| Item | Location |
|------|----------|
| Scripts | `scripts/image_features/` |
| Condition tests | `results/image_features/comparisons/condition_tests.csv` |
| Method notes | `/docs/METHODS_SOURCES_LOG.md` §12 |

---

## Step 4 — Compare features to behaviour and ISC (proxy test)

### What happened

Three association tiers (Spearman + FDR within outcome family):

1. **Raw** feature ↔ RT / accuracy / ISC (often confounded by design).
2. **Residual** after partialling adaptation + nodes + question.
3. **Paired** non−full Δfeature ↔ ΔRT / Δaccuracy (does feature *change* track the adaptation *benefit*?).

Also: CLIPGaze vs real ET; adaptation-mean ordering vs human/ISC means; person-level ISC ↔ behaviour (exploratory Spearmans).

### Why

D9.4-relevant question: can computed image features **stand in for** human performance or shared sensor dynamics? Steps 1–2 establish the human patterns; Step 4 tests whether features track them.

### Why this is justified / correct

- Raw correlations alone are insufficient (Q2/12-node images look different *and* are harder).
- Residual and especially **paired Δ** tests ask the proxy question directly.
- FDR on feature–outcome families; sparse survival → correct **null / weak proxy** conclusion.
- Person-level ISC↔RT left **exploratory** (no FDR claim) because of a large Spearman battery

### Sources to verify

| Item | Location |
|------|----------|
| Feature vs human report | `results/image_features/human_compare/feature_vs_human_report.txt` |
| ISC ↔ behaviour (anti-alias LOO) | `results/multimodal/isc_vs_behaviour.csv` |
| Method notes | `docs/METHODS_SOURCES_LOG.md` §§10, 12 |

---

## Multiplicity and “what counts as significant” (short)

| Analysis | Correction |
|----------|------------|
| Behaviour mixed / clustered models | Clustering required; FDR on few planned coeffs optional |
| ISC contrasts / levels | **FDR within factor** (+ permutation preferred); circular-shift “> chance” = **EEG only** |
| Features × condition | **FDR within factor** |
| Features × human | **FDR within outcome**; prefer residual/paired for proxy claims |
| Person ISC ↔ RT | Exploratory unless FDR applied across the battery |

Full table: archive `FINAL_REPORT.md` §b.6 · citations: archive `METHODS_SOURCES_LOG.md`.

---

## Limitations that affect interpretation (not pipeline bugs)

- N = 13 — directional pilot, not confirmatory precision.
- LOO ISC values within a condition are cross-dependent; permutations help contrasts, not full exchangeability.
- EEG ICA without EOG; ET/EDA on lighter cleaning than EEG ICA/FASTER.
- Semi is not a linear midpoint on all sensors; ink is partly design-confounded with full adaptation marks.
- Scanpath chosen model was not trained on Q/A tasks
