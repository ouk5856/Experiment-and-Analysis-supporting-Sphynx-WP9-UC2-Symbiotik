# WP9 Use Case 2 — analysis results (support for D9.4)
**Code and this report were generated using Cursor and carefully curated by me**
**Sample:** N = 13 participants · 624 trials · 48 graph stimuli  
**Canonical EEG ISC:** `results/eeg_isc_v2_antialias/`  
Methods / justification per step:** [`ANALYSIS_STEPS_METHODS.md`](ANALYSIS_STEPS_METHODS.md)


**How to read significance here:** Each claimed effect is marked **significant** or **not significant**, with the **test**, the **decision rule** (e.g. model *p* after clustering; *p*_FDR within factor), and a one-line **why that test**. Descriptives alone are never treated as inferential.

## Purpose and analysis flow

The following report contains the results of the analysis for **WP9 use case 2**, in order to support **D9.4** for that use case.

The flow of the analysis is as follows:

1. **Behaviour** — response time (RT) and accuracy compared across conditions (primary focus: **adaptation**; secondary: graph size and question level).
2. **Multimodal ISC** — EEG, eye-tracking, and auxiliary sensors were partitioned, preprocessed, and passed through an **inter-subject correlation (ISC)** pipeline, again comparing participants across the same conditions.
3. **Image features** — features were extracted from stimuli per condition (ink, saliency models, CLIPGaze, etc.).
4. **Cross-check** — patterns from behaviour and ISC were compared to image features to test whether those features **correlate with** and can **act as proxies for** the human data.

**Conditions briefly:** Adaptation (non / semi / fully adapted) is the main contrast. Graph size (6 vs 12 nodes) and question level (Q1 vs Q2) are reported mainly where they **clarify or support** the adaptation story (e.g. why semi-adaptation looks “enough” on Q1 but not on Q2).



## 1. Behaviour — adaptation makes the task easier

**Descriptives:** median RT / accuracy by adaptation — from `results/behaviour/`.

| Adaptation | Median RT | Accuracy |
|------------|-----------|----------|
| Non-adapted | **9.53 s** | **71%** |
| Semi-adapted | 7.70 s | 78% |
| Fully adapted | **7.19 s** | **80%** |

### Statistically significant behavioural effects

| Finding | Significant? | Test | Why this test |
|---------|--------------|------|----------------|
| **Full vs non: faster RT** (β ≈ **−2.75 s**, *p* ≈ 4×10⁻²²; paired *d* ≈ **2.35**) | **Yes** | **Linear mixed model (MixedLM):** `rt ~ adaptation + nodes + question + (1 \| participant)` | Continuous RT with **repeated trials within people**; random intercept fixes trial non-independence that pooled tests ignore |
| **Semi vs non: faster RT** (β ≈ **−1.60 s**, *p* ≈ 2×10⁻⁸; *d* ≈ **1.33**) | **Yes** | Same MixedLM | Same clustering rationale; planned adaptation contrast vs non-adapted reference |
| **Full & semi vs non: higher accuracy** (log-odds β ≈ +0.58 / +0.44; *p* ≈ **.018** / **.017**) | **Yes** | **Logistic regression + cluster-robust SEs** by participant | Binary accuracy; clustered SEs correct within-participant dependence without requiring a fragile binomial GLMM at N=13 |
| **Q2 slower / less accurate than Q1**; **12-node harder than 6-node** (large main-effect *p*s in same models) | **Yes** (secondary) | Same MixedLM / clustered logit | Planned design factors; support “harder context → worse behaviour” |
| **Adaptation × question on RT** (full×Q2 β ≈ −1.54, *p* ≈ .005; semi×Q2 β ≈ +2.17, *p* ≈ 9×10⁻⁵) | **Yes** (secondary, design-motivated) | MixedLM **interaction** model | Formal test that semi helps Q1 RT but not Q2 RT; full adds extra Q2 benefit |

**Decision rule for behaviour:** coefficient *p* < .05 **after participant clustering**. FDR across the small set of planned coefficients was **not** required (clustering is the Type I control that mattered)

**Descriptive pattern supporting the interaction (not itself a *p*-value):** Q1 median RT semi ≈ full (≈5.3–5.6 s); Q2 median RT semi ≈ non (≈12.1–12.3 s), full lower (≈8.8 s).

**Detail files:** `results/multimodal/mixed_model_coefficients.csv`, `mixed_effects_report.txt`, `effect_sizes.csv` · `results/behaviour/`

**Summary:** Semi and fully adapted conditions have **statistically significant** lower response times and higher accuracy than non-adapted, so **adaptation helps participants perform the task** (faster and more correct); Q2 and 12-node graphs remain harder, and semi-adaptation alone is not enough for Q2 RT. Results follow the same direction as **D2.2**, with the intuition that visual adaptation **helps** the participant solve the task. Important note here: This is in effect **when the adaptation matched the question context**. (as the experiment mimics D2.2 in design, it **does not test** random adaptation effects)

---


## 2. ISC by sensor — shared dynamics track difficulty / adaptation

**Estimator (all sensors below):** leave-one-out (LOO) Pearson ISC on condition-averaged timecourses, summarized with Fisher-*z* (C-PAC / BrainIAK-style LOO ISC).  
**Why LOO ISC:** estimates **shared stimulus-locked dynamics** across participants without a stimulus regressor — complementary to mean RT.

**Condition contrasts (default decision rule):** paired **Wilcoxon signed-rank** on each participant’s LOO Fisher-*z*, with **Benjamini–Hochberg FDR within factor**; complemented by **sign-flip permutation** *p*s (also FDR’d).  
**Why Wilcoxon + FDR:** within-subject design, N=13, nonparametric on *z*-differences; many related contrasts → FDR.  
**Why permutation:** Monte Carlo null for mean paired Δ*z* = 0 under sign exchangeability.

**ISC > chance (levels):** circular time-shift null + FDR — **EEG only** (anti-alias / v2 permutation scripts). Destroys cross-subject alignment while keeping each subject’s autocorrelation. Multimodal ET/EDA use Wilcoxon FDR + sign-flip **contrasts** only (no circular-shift level tests).




### EEG (GFP — canonical anti-alias)

| Condition | Mean LOO ISC *r* | Above chance? |
|-----------|------------------|---------------|
| Non-adapted | **0.411** | **Yes** (*p*_perm FDR) |
| Semi-adapted | **0.111** | **Yes** (*p*_perm FDR) |
| Fully adapted | **0.226** | **Yes** (*p*_perm FDR) |

| Finding | Significant? | Test + rule | Why this test |
|---------|--------------|-------------|---------------|
| **GFP non > full** (0.411 vs 0.226) | **Yes** | Wilcoxon *p*_FDR ≈ **.009**; perm FDR ≈ **.004** | Paired participant LOO *z*; FDR within adaptation family |
| **GFP non > semi** (0.411 vs 0.111) | **Yes** | Wilcoxon *p*_FDR ≈ **.004**; perm FDR ≈ **.001** | Same |
| **GFP semi vs full** | **No** | *p*_FDR ≈ .120; perm FDR ≈ .085 | Same tests — contrast does not survive correction |
| **GFP 12-node > 6-node** (0.40 vs 0.25) | **Yes** (secondary) | Wilcoxon + perm, both FDR **sig** | Harder graphs → higher shared dynamics; supports adaptation/difficulty reading |
| **GFP Q2 > Q1** (0.46 vs 0.30) | **Partial** | Wilcoxon *p*_FDR ≈ .065 (**ns**); perm FDR ≈ **.049** (**sig**) | Direction consistent; claim only with permutation agreement noted |

**Note (not a failed test):** semi GFP is **lowest** (*r*=0.111), not midway — descriptive ordering after significant non>semi / non>full.

**Detail files:** `results/eeg_isc_v2_antialias/analysisA_isc_contrasts_v2.csv`, `permutation_isc_contrasts.csv`, level CSVs

**Summary (EEG):** Non-adapted GFP ISC is **significantly higher** than fully and semi-adapted (FDR + permutation), i.e. people share more EEG dynamics when the graph is unadapted; larger graphs also raise shared GFP ISC.




### Eye-tracking (ISC)

Same LOO ISC + Wilcoxon FDR + sign-flip contrasts as EEG (no circular-shift “> chance” tests for ET).

| Finding | Significant? | Test + rule | Why this test |
|---------|--------------|-------------|---------------|
| **Gaze Y: non > semi, non > full** (0.73 vs 0.53 / 0.56) | **Yes** | Paired Wilcoxon, *p*_FDR ≈ **.014** (both) | Same within-subject ISC contrast logic as EEG |
| **Gaze X: non > full** only (not non > semi) | **Yes** (non>full) | *p*_FDR ≈ **.014** | Same; do not claim non>semi for gaze X |
| **Pupil adaptation contrasts** | **No** | No FDR-surviving adaptation contrast | High baseline pupil ISC; adaptation does not modulate shared pupil dynamics here |

**Summary (eye-tracking):** Shared **gaze Y** dynamics are **significantly higher** in non-adapted than in semi/full; **gaze X** is significant only for **non > full**. That matches the behavioural “harder without adaptation” story on the vertical axis; **pupil** ISC does not show a significant adaptation effect.



### Auxiliary sensors (ISC)

| Finding | Significant? | Test + rule | Why this test |
|---------|--------------|-------------|---------------|
| **EDA: non > full** (≈0.42 vs ≈0.01) | **Yes** | Wilcoxon *p*_FDR ≈ **.014** | Same ISC contrast pipeline |
| **EDA: semi vs non / semi vs full** (semi ≈ non ≫ full) | Semi tracks **non**, not midpoint | Contrast family as above; highlight is non>full | Autonomic shared arousal drops mainly with **full** adaptation |
| **PPG adaptation contrasts** | **No** | No FDR-surviving contrast | Low SNR / weak shared PPG structure in this design |

**Summary (aux):** **EDA** ISC is **significantly higher** for non-adapted than fully adapted (shared autonomic dynamics drop with full adaptation); **PPG** shows no significant adaptation contrast.



### Condition-level picture (adaptation focus)

```
Non-adapted  → slower / less accurate  → higher GFP, gaze Y, EDA ISC   [sig contrasts above]
Fully adapted → faster / more accurate → lower GFP, gaze Y, EDA ISC  [sig contrasts above]
```

**Detail files:** `results/multimodal/multimodal_isc_contrasts.csv`, `multimodal_isc_report.txt` (adaptation level means), `multimodal_subject_loo.csv`

**Summary (ISC overall):** Across EEG GFP, **gaze Y** (and gaze X **non>full**), and EDA, **non-adapted yields significantly higher inter-subject synchrony** than full adaptation — consistent with more shared processing when the task is harder without encoding help; pupil and PPG do not add a significant adaptation signal.

---



## 3. Image features — separate condition images, not human outcomes

**Test:** Kruskal–Wallis / Mann–Whitney across condition levels, **FDR within factor**.  
**Why:** Many skewed features × three design factors → nonparametric + multiplicity control. This step is a **stimulus QA** claim (“do images differ by label?”), not a human-proxy claim.

| Finding | Significant? | Test + rule |
|---------|--------------|-------------|
| **Digital ink: non vs full** | **Yes** (*p*_FDR ≈ .031) | MW/KW + FDR within adaptation |
| **SUM saliency stats: 6 vs 12 nodes** | **Yes** (several FDR hits) | FDR within nodes factor |
| **CLIPGaze: Q1 vs Q2** (e.g. predicted fixations / saccades) | **Yes** | FDR within question factor |
| **ViSalFormer × conditions** | **No** FDR-significant splits | Same family — model does not separate labels under FDR |
| **SUM / other adaptation splits** | Mostly **no** after FDR (some raw *p*s only) | Do not claim adaptation separation without *p*_FDR |

**Detail files:** `results/image_features/comparisons/condition_tests.csv`

**Summary:** Some image features **significantly differ by condition label** (ink by adaptation, SUM by graph size, CLIPGaze by question), so stimuli are distinguishable as designed — but that alone does **not** mean they track human performance. Importantly, eye-tracking predictive models look to not distinguish betweent he main adaptations, which is surprising for models such as **ViSalFormer**, which is specifically designed and trained on contextual Q/A on similar graphs. **Caveat**: the scanpath predicting model CLIPGaze is not specifically trained on graph Q/A, and was used only due to high performance on contextual image scanpath prediction.



---

## 4. Do features align with behaviour and ISC? (mostly no)

**Proxy tests:** Spearman correlations in three tiers — **raw**, **residual** (adaptation+nodes+Q partialled), **paired** non−full Δfeature ↔ ΔRT/Δaccuracy — with **FDR within outcome family**.  
**Why these tests:** Raw associations are confounded by design (harder conditions look different). Residual and especially **paired Δ** ask whether feature *change* tracks the human adaptation *benefit*.

| Claim | Significant as proxy? | Test + rule | Why this test |
|-------|----------------------|-------------|---------------|
| Features predict adaptation RT/accuracy benefit | **No** | Paired Δfeature ↔ ΔRT / Δaccuracy; **no FDR survivors** for RT/accuracy | Direct test of “feature change = human benefit” |
| Features track GFP / gaze Y ISC adaptation pattern | **No** | Qualitative adaptation-mean ordering (CSV Spearmans may be empty; not a formal test) | Feature means do not reproduce non ≫ adapted ISC ordering |
| CLIPGaze matches real eye-tracking | **No** | Spearman CLIPGaze ↔ ET; **no FDR hits** | Predicted scanpaths ≠ human gaze under correction |
| Residual CLIPGaze ↔ accuracy | **Some FDR hits** (complexity / cognitive_load / scanpath_length) | Residual Spearman + FDR | Not an adaptation-benefit proxy; no paired ΔRT link |
| Raw feature ↔ RT (many hits) | **Not interpretable as proxy** | Raw Spearman (often FDR in raw tier) | Confounded by Q2/12-node co-varying with appearance |

**Person-level ISC ↔ individual RT/accuracy:** Spearmans + bootstrap CIs on **anti-alias** LOO (`results/multimodal/isc_vs_behaviour.csv`) — **exploratory** (no FDR claim). GFP×RT null; occipital×RT in non-adapted is **ns** after anti-alias refresh (*p*≈.071).

**Verdict for D9.4:** **Significant** human evidence that adaptation eases behaviour (MixedLM / clustered logit) and that shared GFP / gaze Y / EDA ISC is higher in non-adapted than fully adapted (Wilcoxon FDR ± perm). Image features **significantly** separate some **stimulus labels**, but **do not significantly** proxy human RT benefit under paired FDR tests (residual accuracy hits are not adaptation-benefit proxies).

**Detail files:** `results/image_features/human_compare/feature_vs_human_report.txt` · `results/multimodal/isc_vs_behaviour.csv`

### Known non-alignments within human data (not feature failures)

- GFP **semi lowest**; EDA **semi ≈ non** — different sensors ≠ one linear adaptation dial (semi vs full GFP **not** significant).
- Occipital EEG ISC can diverge from GFP on semi (visual ROI vs global).
- Person-level ISC does **not** reliably predict individual RT (exploratory; anti-alias refresh).

**Summary:** Image features **do not significantly act as proxies** for the human adaptation **RT/accuracy benefit** under paired FDR tests; useful for stimulus QA, **not** as stand-ins for behaviour or sensors in this pilot.

---

## Artifact map (verify numbers)

| Step | Prefer these files |
|------|-------------------|
| Behaviour descriptives | `results/behaviour/pilot_descriptives_by_cell.csv`, `sphynx13_summary.txt` |
| Behaviour inference | `results/multimodal/mixed_model_coefficients.csv`, `effect_sizes.csv` |
| EEG ISC | `results/eeg_isc_v2_antialias/` |
| ET / EDA / pupil / PPG ISC | `results/multimodal/multimodal_isc_contrasts.csv`, `multimodal_isc_report.txt`, `multimodal_subject_loo.csv` |
| Features × condition | `results/image_features/comparisons/` |
| Features × human | `results/image_features/human_compare/` |

**How / why each step is justified:** [`ANALYSIS_STEPS_METHODS.md`](ANALYSIS_STEPS_METHODS.md)  


