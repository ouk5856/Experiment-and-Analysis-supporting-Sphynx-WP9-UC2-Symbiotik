# Sphynx post-analysis — image features for adaptation

**2026-08-18**

Stage 2: can image features differentiate the experiment conditions (adaptation, graph size, question level)? That is a **stimulus check**, not the behavioural claim.

Stage 3 (script 04): do those feature differences track Sphynx-13 human RT / accuracy / ET / ISC? This is the test that matters for an adaptation engine.

**Integrated narrative (behaviour + ISC + sensors + features):** see [`INTEGRATED_PILOT_SUMMARY.md`](INTEGRATED_PILOT_SUMMARY.md).

This folder writes only under `post_analysis/`. Script 04 **reads** existing Sphynx-13 analysis CSVs next door and does not modify them.

---

## Stimuli

The experiment uses **48** unique images (not 120). Source catalog:

`experiment/psychopy_experiment_stimuli_update_3_all_48_answered.csv`

Copied images (CSV-relative paths): `images/samples/...`

Conditions from the catalog:

- **adaptation:** `non_adapted` / `fully_adapted` / `semi_adapted`  
  (semi = `shape_nodes`, `color_nodes`, `size_nodes`, `edge_width`)
- **graph size:** 6 vs 12 (`graph_complexity`)
- **question level:** Q1 vs Q2 (`question_complexity`)

Text for VisalFormer and CLIPGaze = `update_questions`.

We copy **only this stimulus catalog** into this folder. Script 04 joins human outcomes from `../experiment_for_Sphynx_pilot/analysis_sphynx13*` by `graph_image` (case-insensitive).

---

## Layout

```
post_analysis/
  experiment/     stimulus catalog CSV
  images/         48 experiment PNGs
  models/         visalformer, sum, clipgaze (inference code only)
  results/low_level/
  results/high_level/{visalformer,sum,clipgaze}/
  results/comparisons/
  01_extract_low_level_features.py
  02_extract_high_level_features.py
  03_compare_features_across_conditions.py
  04_compare_features_to_human.py
```

HAT is **not** included: it does not take real question text (only TP/TA/FV task IDs).

`models/clipgaze/scanpath_to_saliency_methods.py` is copied from the old GazeFormer folder; CLIPGaze’s `standardized_inference.py` imports it.

---

## Why these models

| Model | Role | Why it fits | Caveat |
|-------|------|-------------|--------|
| **VisalFormer** | Saliency + **text** | Used/trained in the previous project on **QA graph** stimuli with a question string — closest to this experiment. | Saliency only (no scanpath). |
| **SUM** | Saliency, **no text** | Strong saliency-benchmark performer. Condition **3 = UI** (graphs as interface, not e-commerce condition 2). | Cannot condition on the question. |
| **CLIPGaze** | Scanpath + **text**, **TP** weights | Strong scanpath-benchmark performer; CLIP text encoder. | **Not** QA-graph trained (COCO-Search-style). A scanpath model with QA graphs + text would be a better fit later. |

**Possible later scanpath upgrades** (same remote server, not run here): GazeFormer (`gazeformer_cocosearch_TP.pkg` under the priority tree) or EyeFormer. HAT stays out (no free-form text).

---

## Remote checkpoints (already on the server)

Copy this `post_analysis` folder to a new directory on the remote machine and run. Weights stay at these paths (from the old lightningnet scripts). Override with `--ckpt` if needed.

- VisalFormer: `/data/gkoutr/BCI-RS/synthetic_eyetracking_works/non-priority/visalformer/Code/VisSalFormer_weights.tar`
- SUM: `/data/gkoutr/BCI-RS/synthetic_eyetracking_works/non-priority/sum/SUM-main/net/pre_trained_weights/sum_model.pth`
- CLIPGaze TP: `/data/gkoutr/BCI-RS/synthetic_eyetracking_works/priority/clipgaze/CLIPGaze-main/checkpoint/CLIPGaze_TP.pkg`

GPU: **`cuda:1`** (GPU 1). One model per process so VRAM is not shared.

---

## How to run

### 1. Low-level (already run locally)

Digital ink (resize 800×600, mean RGB sum), data utility `2 * nodes + 1`, effectiveness = utility / ink. Nodes from the catalog.

```bash
python 01_extract_low_level_features.py
```

Output: `results/low_level/low_level_features.csv`

### 2. High-level (remote server — one model at a time)

```bash
python 02_extract_high_level_features.py --model visalformer
python 02_extract_high_level_features.py --model sum
python 02_extract_high_level_features.py --model clipgaze
```

Optional: `--device cuda:1` (default), `--ckpt /path/to/weights`.

Per model:

- `results/high_level/{model}/outputs/` — saliency / scanpath files
- `results/high_level/{model}/{model}_features.csv`
- `results/high_level/{model}/{model}_run.log` — each stimulus: input image, question (or SUM condition 3), checkpoint, device, output paths, status

Saliency features (all three): mean, std, max, min, entropy, center bias, spread, skewness, kurtosis.

Scanpath features (CLIPGaze): n fixations, duration mean/std, scanpath length, complexity, mean saccade mag, cognitive-load and info-processing indices (same formulas as the old lightningnet synthetic feature extractor; no ecommerce AOIs).

If a checkpoint is missing, the script logs the path and exits without running inference.

### 3. Compare across conditions

```bash
python 03_compare_features_across_conditions.py
```

Kruskal / Mann–Whitney on adaptation, 6 vs 12, Q1 vs Q2. FDR-BH and Bonferroni
within each factor family (near-constant features excluded). Runs low-level even if high-level CSVs are not back from the server yet.

Outputs:

- `results/comparisons/condition_tests.csv`
- `results/comparisons/condition_comparison_report.txt`

Copy high-level `*_features.csv` (and optionally `outputs/`) back here, then re-run script 3.

### 4. Compare features to human data

```bash
python 04_compare_features_to_human.py
```

Joins the 48 stimuli to Sphynx-13 trial RT/accuracy and ET features. Reports Spearman (raw, design-residual, and paired non−full deltas), FDR-BH within each outcome.

Outputs: `results/human_compare/`

---

## Isolation

Only this folder is new/written. Experiment analysis under `experiment_for_Sphynx_pilot/` is untouched aside from reading the 48 PNGs to copy them.
