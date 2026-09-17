# WP9 Use Case 2 — Experiment & Analysis

This repository contains the experiment design and post-experiment analysis for **WP9 Use Case 2**, supporting deliverable **D9.4** of project **Symbtiotik**.

---

## Folder Structure

### [`experiment files/`](experiment%20files/)

Experiment design, stimuli, and justification behind decisions.

Contains the PsychoPy experiment files, participant instructions, stimulus lists, visual legends, and a detailed `readme.md` describing:

- How and why the CTI graph stimuli were redesigned from the original Sphynx material.
- The question-answering paradigm and its ISC-compatibility constraints.
- Pilot testing (two rounds, 4 participants) and the changes that resulted.
- The final protocol used for the 13-participant main experiment.
- Key design decisions with literature justification.

### [`analysis/`](analysis/)

Post-experiment analysis pipeline and results for the D9.4 report.

Contains analysis scripts, result files, and two summary documents:

- **`D9_4_WP9_UC2_REPORT_SUMMARY.md`** — Results brief focusing on adaptation conditions, with statistical significance, tests used, and justifications.
- **`ANALYSIS_STEPS_METHODS.md`** — Step-by-step description of what analysis was performed, why, and how it is justified.
- **`scripts/`** — Behavioural mixed models, multimodal ISC (EEG, eye-tracking, PPG), image feature extraction, cross-comparisons, and effect sizes.
- **`results/`** — Canonical CSV outputs and reports referenced by the documents above.

The analysis flow is: behavioural analysis → multimodal ISC across conditions → image feature extraction → cross-comparison of features with human data (behaviour and sensors).

---

# Important: Added a brief slide-deck summarizing the results.
Can be found inside analysis folder



## Tools & Authorship

Analysis scripts were created using [Cursor](https://www.cursor.com/) and curated by the author. Report and documentation files were also organised and improved in clarity with Cursor assistance, then reviewed and curated manually.
