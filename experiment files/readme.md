# Experiment Files — WP9 Use Case 2

This folder documents the design, piloting, and final protocol of the graph-based experiment that generated the data analysed in [`../analysis/`](../analysis/).

---

## 1. Experiment Objective

The experiment was designed as a graph-based question-answering task analogous to the paradigm used in D2.2, using CTI (Cyber-Threat Intelligence) graph material from the Sphynx platform.
The design was made compatible with **inter-subject correlation (ISC)** analysis, following the guidelines of Nastase et al. (2019) ([PMC6688448](https://pmc.ncbi.nlm.nih.gov/articles/PMC6688448/)).

The main objectives were to:

1. Verify whether workload-related patterns were comparable to those observed in the previous graph-based experiment (D2.2).
2. Identify inter-subject behavioural and neurophysiological patterns across **adaptation conditions** (non-adapted, semi-adapted, fully adapted graphs).
3. Assess whether these patterns correlate with graph/image features, to evaluate features as potential proxies for human data.

---

## 2. Stimuli Redesign

The initially shared stimuli from Sphynx could not be used directly. The main issues were:

- Low resolution/PPI and inconsistent image sizes.
- Some adaptation conditions were incomplete or incorrect (missing, partially adapted, or unreadable information).
- Inconsistent visual encodings across shared material.
- Saliency and scanpath prediction models tested on the original images were not appropriate for this task; their outputs did not represent realistic viewing behaviour for graph stimuli.
- The original graph samples did not sufficiently reflect realistic SOC tasks — they had been selected somewhat arbitrarily from a larger GoTrimm propagation graph, and the small sizes constrained what questions could be asked.

Following meetings with Sphynx SOC experts, graphs were **redesigned from scratch** (generation code not included in this repo). Redesigned graphs were initialised with randomly allocated variable values while maintaining structures and variable distributions similar to the Sphynx examples.

### Visual encodings (final version)

| Variable       | Encoding                                                    |
|----------------|-------------------------------------------------------------|
| Node type      | Shape (3 categories, reduced from the original 5)           |
| TLP level      | Colour matching the actual TLP colour names                 |
| Node size      | Numerical value (1–10, grouped: Low 1–3, Medium 4–6, High 7–10) |
| Edge width     | Numerical value (1–10, same grouping)                       |

Graph sizes of **6 and 12 nodes** (plus the GoTrimm-related node) were retained. Graphs with only 3 additional nodes were excluded — they were too easy and would introduce mostly noise into neurophysiological signals, which is problematic for ISC.

---

## 3. Questions

Questions were based on graph variables and had two difficulty levels:

- **Easy (Q1):** required inspection of one variable.
- **Hard (Q2):** required inspection of two variables.

Key changes driven by the review process:

- **Binary (yes/no) questions were removed** entirely, keeping only count-type questions. This eliminated ambiguity around chance level and simplified the analysis.
- Questions were **disambiguated** where the original wording was unclear (e.g., clarifying "connected with low confidence level edges" and "with the biggest size" when ties were possible).
- Hints on how to answer were added to the visual instructions.

---

## 4. Experimental Procedure (Final Protocol)

Each trial followed this sequence:

1. **Question display** — fixed duration (~5 s), sufficient for reading and comprehension. Justified by reading-speed literature: 2.5–4 s suffice for comprehension of up to 16 words (Brysbaert, 2019 — [doi:10.1016/j.jml.2019.104047](https://www.sciencedirect.com/science/article/abs/pii/S0749596X19300786)).
2. **Fixation cross** — brief visual anchor.
3. **Graph display** — fixed 18 s viewing period (increased from the initial 10 s after pilot feedback). Participants answered during this phase; response time and accuracy were recorded by PsychoPy.
4. **Inter-trial break** — short fixed pause.

### ISC compatibility constraints

Following Nastase et al. (2019) and discussion with ISC expert:

- The experiment is **not self-paced**: all timing is fixed and identical across participants, as required by ISC assumptions.
- Trials are presented in a **grouped random order** (identical for all participants) to allow epoch-wise ISC while controlling for sequence effects.
- **No answer-confirmation pop-up** is shown, as it would introduce timing variability and noise. Participants were instructed that answering once is final and that not answering is also considered a valid response.

### Fatigue management

Based on evidence that cognitive fatigue affects task performance after ~20 minutes (Hopstaken et al., 2015 — [doi:10.1111/psyp.14126](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.14126)):

- **Fixed breaks** of 3 minutes were introduced approximately every 20 minutes of active task time.
- The number of conditions and trials per condition pair was set to keep total active duration reasonable while maintaining sufficient data for ISC (4 trials per condition pair).

---

## 5. Pilot Testing

Four laboratory personnel participated in pilot testing (personal details not shared for data-protection reasons).

### First pilot group (2 participants) — initial design

- Too many visual variables to remember.
- TLP categories had colour-based names but used a different encoding, while colour was already used for node type → confusing.
- Participants could not easily recognise what information had been adapted.
- Graph viewing time was too short (10 s).
- No significant response-time differences across adaptation conditions.
- High rate of unanswered or late responses.

### Second pilot group (2 participants) — revised design

After reducing node types, aligning TLP encoding to colour names, and increasing viewing time to 18 s:

- ~90% of questions received an answer within the allowed time.
- 100% accuracy among answered questions.
- Response-time analysis showed **significant differences between the extreme conditions** (non-adaptation vs. full adaptation).
- Semi-adaptation conditions showed trends consistent with this difference.

This revised version was adopted as the final protocol. The same behavioural pattern was confirmed in the first participant of the main experiment.

---

## 6. Main Experiment — 13 Participants

The final experiment was run with **13 participants**. Data collected per participant:

| Modality             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| Behavioural          | Answer per question, accuracy, response time, non-response/late info    |
| Eye-tracking         | Gaze behaviour during graph viewing (GazePoint device)                  |
| EEG                  | Electroencephalography during the full experiment                       |
| PPG                  | Photoplethysmography from the GazePoint sensor (auxiliary physiological signal) |

> **Note:** An earlier description (`readme_v1.txt`) listed "ECG and EDA" instead of PPG. This was a mistake in that draft; the actual auxiliary sensor used was PPG from the GazePoint device.

---

## 7. Key Design Decisions — Summary

| Decision | Rationale | Source |
|----------|-----------|--------|
| Fixed timing (no self-pacing) | Required by ISC; all participants must experience identical temporal structure | Nastase et al. (2019) |
| 18 s graph viewing | Increased from 10 s after pilot failure; balances difficulty across graph sizes | Pilot results |
| Grouped random trial order | Allows epoch-wise ISC while reducing sequence effects | Nastase et al. (2019) |
| Remove binary questions | Eliminates chance-level ambiguity; homogenises cognitive demands | Review discussion |
| Reduce node types 5→3 | Pilot participants could not memorise 5 types reliably | Pilot results |
| TLP encoded as colour | Aligns encoding with the variable's colour-based name; eliminates confusion | Pilot results |
| Fixed breaks every ~20 min | Fatigue degrades cognitive performance after ~20 min | Hopstaken et al. (2015) |
| No answer confirmation | Avoids timing variability that would violate ISC assumptions | Expert advice |

---

## 8. Files in This Folder

| File | Description |
|------|-------------|
| `draft_3.psyexp` | Final PsychoPy experiment file used for the main study |
| `draft__trial.psyexp` | Earlier PsychoPy draft (trial version) |
| `participant_instructions_improved_18s.html` | Participant instruction page for the 18 s viewing-time version |
| `psychopy_experiment_stimuli_update_3_all_48_answered.csv` | Stimulus list / condition file used by PsychoPy (all 48 trials) |
| `cti_experiment_legend_2x2_compact.png` | Visual legend of the graph encodings (with titles) |
| `cti_experiment_legend_2x2_compact_no_titles.png` | Visual legend of the graph encodings (without titles) |

---

## 9. References

- Nastase, S. A., Gazzola, V., Hasson, U., & Keysers, C. (2019). Measuring shared responses across subjects using intersubject correlation. *Social Cognitive and Affective Neuroscience*, 14(6), 667–685. [PMC6688448](https://pmc.ncbi.nlm.nih.gov/articles/PMC6688448/)
- Hopstaken, J. F., van der Linden, D., Bakker, A. B., & Kompier, M. A. J. (2015). A multifaceted investigation of the link between mental fatigue and task disengagement. *Psychophysiology*, 52(3), 305–315. [doi:10.1111/psyp.14126](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.14126)
- Brysbaert, M. (2019). How many words do we read per minute? A review and meta-analysis of reading rate. *Journal of Memory and Language*, 109, 104047. [doi:10.1016/j.jml.2019.104047](https://www.sciencedirect.com/science/article/abs/pii/S0749596X19300786)
- Dmochowski, J. P., Bezdek, M. A., Abelson, B. P., Johnson, J. S., Schumacher, E. H., & Parra, L. C. (2014). Audience preferences are predicted by temporal reliability of neural processing. *Nature Communications*, 5, 4567. [doi:10.1038/ncomms5567](https://www.nature.com/articles/ncomms5567)
