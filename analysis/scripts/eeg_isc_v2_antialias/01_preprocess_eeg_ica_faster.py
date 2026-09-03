#!/usr/bin/env python3
"""
Sphynx-13 EEG preprocessing — anti-alias sensitivity (for_report only).

Same as EEG ISC v2 ICA+FASTER, but before downsampling GFP/occ to 25 Hz ISC rate,
applies a zero-phase Butterworth lowpass at 12 Hz (< Nyquist 12.5 Hz).

Does not modify the original eeg_isc_v2 pipeline.

  python 01_preprocess_eeg_ica_faster.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne_faster import (
    find_bad_channels,
    find_bad_channels_in_epochs,
    find_bad_components,
    find_bad_epochs,
)

from sphynx13_eeg_utils import (
    DEFAULT_DATA_DIR,
    EPOCH_SEC,
    ISC_ANTIALIAS_HZ,
    ISC_SFREQ,
    NON_EEG_CHS,
    OCCIPITAL_ROI,
    discover_participants,
    load_psychopy_trials,
    normalize_stim_code,
    resample_1d_antialias,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
FOR_REPORT = HERE.parents[1]  # .../for_report
REPO_ROOT = FOR_REPORT.parents[2]  # .../sphynx pilots stuff
OUT_DIR = FOR_REPORT / "results" / "eeg_isc_v2_antialias" / "preprocessed"

ICA_METRICS = ["kurtosis", "power_gradient", "hurst", "median_gradient"]
FASTER_THRES = 3.0
ICA_MAX_COMPONENTS = 20


def pair_s4_s5_events(
    events: np.ndarray,
    s4_id: int,
    s5_id: int,
    sfreq: float,
    epoch_sec: float,
    n_times: int,
) -> list[tuple[int, int]]:
    s4_samples = events[events[:, 2] == s4_id][:, 0]
    s5_samples = events[events[:, 2] == s5_id][:, 0]
    s5_list = list(s5_samples)
    pairs = []
    s5_idx = 0
    for s4 in s4_samples:
        while s5_idx < len(s5_list) and s5_list[s5_idx] <= s4:
            s5_idx += 1
        if s5_idx >= len(s5_list):
            end = int(s4 + epoch_sec * sfreq)
        else:
            end = int(s5_list[s5_idx])
        max_end = int(s4 + epoch_sec * sfreq)
        end = min(end, max_end, n_times - 1)
        if end - s4 < int(0.5 * sfreq):
            continue
        pairs.append((int(s4), end))
    return pairs


def clean_epochs_faster(epochs: mne.Epochs) -> tuple[mne.Epochs, dict]:
    """Run FASTER steps 1–4 (adapted for recordings without EOG)."""
    log: dict = {
        "n_epochs_in": len(epochs),
        "bad_channels_global": [],
        "bad_epochs": [],
        "bad_ica_components": [],
        "bad_channels_per_epoch": {},
    }
    ep = epochs.copy()

    bad_chs = find_bad_channels(ep, thres=FASTER_THRES)
    ep.info["bads"] = sorted(set(ep.info["bads"] + list(bad_chs)))
    log["bad_channels_global"] = list(ep.info["bads"])

    bad_ep = find_bad_epochs(ep, thres=FASTER_THRES)
    log["bad_epochs"] = [int(i) for i in bad_ep]
    good = [i for i in range(len(ep)) if i not in bad_ep]
    ep = ep[good]

    picks = mne.pick_types(ep.info, meg=False, eeg=True, exclude="bads")
    n_comp = min(len(picks) - 1, ICA_MAX_COMPONENTS)
    if n_comp < 2 or len(ep) < 3:
        log["skipped_ica"] = True
        return ep, log

    ica = mne.preprocessing.ICA(
        n_components=n_comp,
        random_state=97,
        max_iter="auto",
        method="fastica",
    )
    ica.fit(ep, picks=picks)
    bad_comp = find_bad_components(ica, ep, thres=FASTER_THRES, use_metrics=ICA_METRICS)
    ica.exclude = list(bad_comp)
    log["bad_ica_components"] = list(ica.exclude)
    ica.apply(ep)

    bad_per_ep = find_bad_channels_in_epochs(ep, thres=FASTER_THRES)
    for i, b in enumerate(bad_per_ep):
        if len(b) > 0:
            log["bad_channels_per_epoch"][str(i)] = list(b)
            epoch = ep[i]
            epoch.info["bads"] = list(set(epoch.info["bads"] + list(b)))
            try:
                epoch.interpolate_bads(reset_bads=True)
                ep._data[i, :, :] = epoch._data[0, :, :]
            except RuntimeError:
                log.setdefault("interpolate_skipped_epochs", []).append(int(i))

    ep.set_eeg_reference("average", projection=False)
    log["n_epochs_out"] = len(ep)
    log["skipped_ica"] = False
    return ep, log


def epochs_to_trial_records(
    epochs: mne.Epochs,
    psychopy: pd.DataFrame,
    folder_name: str,
    kept_indices: list[int],
    cleaning_log: dict,
) -> tuple[list[dict], dict]:
    """Convert cleaned MNE epochs to trial dicts with gfp_isc / occ_isc."""
    n_isc = int(round(EPOCH_SEC * ISC_SFREQ))
    data = epochs.get_data() * 1e6
    ch_names = epochs.ch_names
    occ_idx = [i for i, n in enumerate(ch_names) if n in OCCIPITAL_ROI]
    sfreq = float(epochs.info["sfreq"])
    n_target = int(round(EPOCH_SEC * sfreq))

    records = []
    for out_i, src_i in enumerate(kept_indices):
        if src_i >= len(psychopy):
            break
        row = psychopy.iloc[src_i]
        seg = data[out_i]
        gfp = np.std(seg, axis=0)
        occ = np.mean(seg[occ_idx, :], axis=0) if occ_idx else np.mean(seg, axis=0)

        if len(gfp) >= n_target:
            gfp_fixed = gfp[:n_target]
            occ_fixed = occ[:n_target]
        else:
            pad = n_target - len(gfp)
            gfp_fixed = pd.Series(np.pad(gfp, (0, pad), constant_values=np.nan)).ffill().bfill().to_numpy()
            occ_fixed = pd.Series(np.pad(occ, (0, pad), constant_values=np.nan)).ffill().bfill().to_numpy()

        records.append(
            {
                "folder_name": folder_name,
                "participant_id": str(row["participant_id"]),
                "modality": "eeg",
                "trial_order": int(src_i),
                "graph_nodes": int(row["graph_nodes"]),
                "question_level": int(row["question_level"]),
                "adaptation": str(row["adaptation"]),
                "graph_image": str(row["graph_image"]),
                "duration_s": len(gfp) / sfreq,
                "sfreq": sfreq,
                "qc_pass": True,
                "preprocess": "ica_faster",
                "gfp_isc": resample_1d_antialias(gfp_fixed, n_isc, sfreq, ISC_ANTIALIAS_HZ),
                "occ_isc": resample_1d_antialias(occ_fixed, n_isc, sfreq, ISC_ANTIALIAS_HZ),
            }
        )

    summary = {
        "folder_name": folder_name,
        "n_trials_psychopy": len(psychopy),
        "n_epochs_created": cleaning_log.get("n_epochs_in", 0),
        "n_epochs_after_faster": cleaning_log.get("n_epochs_out", 0),
        "n_trials_kept": len(records),
        "bad_channels_global": cleaning_log.get("bad_channels_global", []),
        "n_bad_epochs": len(cleaning_log.get("bad_epochs", [])),
        "bad_ica_components": cleaning_log.get("bad_ica_components", []),
        "skipped_ica": cleaning_log.get("skipped_ica", False),
    }
    return records, summary


def preprocess_participant(
    vhdr: Path,
    psychopy: pd.DataFrame,
    folder_name: str,
    epoch_sec: float = EPOCH_SEC,
) -> tuple[list[dict], dict, list[str]]:
    notes: list[str] = []
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
    drop = [ch for ch in raw.ch_names if ch in NON_EEG_CHS]
    if drop:
        raw.drop_channels(drop)
    raw.filter(1.0, 40.0, fir_design="firwin", verbose=False)
    try:
        raw.notch_filter(50.0, verbose=False)
    except Exception:
        pass
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")
    except Exception:
        notes.append(f"{folder_name}: could not set standard_1020 montage")
    raw.set_eeg_reference("average", projection=False, verbose=False)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    code_to_id = {}
    for name, eid in event_id.items():
        code = normalize_stim_code(name)
        if code:
            code_to_id[code] = eid
    if "S4" not in code_to_id or "S5" not in code_to_id:
        notes.append(f"{folder_name}: missing S4/S5 markers")
        return [], {}, notes

    pairs = pair_s4_s5_events(
        events,
        code_to_id["S4"],
        code_to_id["S5"],
        float(raw.info["sfreq"]),
        epoch_sec,
        raw.n_times,
    )
    if not pairs:
        notes.append(f"{folder_name}: no S4/S5 epoch pairs")
        return [], {}, notes

    n_psy = len(psychopy)
    if len(pairs) > n_psy:
        notes.append(f"{folder_name}: truncating {len(pairs)} epochs to {n_psy} PsychoPy trials")
        pairs = pairs[:n_psy]

    sfreq = float(raw.info["sfreq"])
    event_arr = np.array([[start, 0, 1] for start, _ in pairs])
    durations = [(end - start) / sfreq for start, end in pairs]
    tmax = min(epoch_sec, max(durations) if durations else epoch_sec)
    epochs = mne.Epochs(
        raw,
        event_arr,
        event_id={"trial": 1},
        tmin=0.0,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
        reject_by_annotation=False,
    )

    cleaned, cleaning_log = clean_epochs_faster(epochs)
    bad_ep_set = set(cleaning_log.get("bad_epochs", []))
    kept_src_indices = [i for i in range(len(pairs)) if i not in bad_ep_set]
    records, summary = epochs_to_trial_records(cleaned, psychopy, folder_name, kept_src_indices, cleaning_log)
    notes.append(
        f"{folder_name}: kept {len(records)}/{len(pairs)} trials after ICA+FASTER "
        f"(bad epochs={len(bad_ep_set)}, ICA excluded={len(cleaning_log.get('bad_ica_components', []))})"
    )
    return records, summary, notes


def save_trial_arrays(records: list[dict], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in records:
        tid = f"{r['folder_name']}_t{r['trial_order']:02d}"
        gfp_path = out_dir / f"{tid}_gfp.npy"
        occ_path = out_dir / f"{tid}_occ.npy"
        np.save(gfp_path, r["gfp_isc"])
        np.save(occ_path, r["occ_isc"])
        rows.append(
            {
                "trial_id": tid,
                "folder_name": r["folder_name"],
                "participant_id": r["participant_id"],
                "trial_order": r["trial_order"],
                "graph_nodes": r["graph_nodes"],
                "question_level": r["question_level"],
                "adaptation": r["adaptation"],
                "graph_image": r["graph_image"],
                "duration_s": r["duration_s"],
                "qc_pass": r["qc_pass"],
                "gfp_npy": str(gfp_path.name),
                "occ_npy": str(occ_path.name),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sphynx-13 EEG ICA+FASTER preprocessing")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("-o", "--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-participants", type=int, default=None)
    args = parser.parse_args()

    root = REPO_ROOT
    data_dir = args.data_dir or (root / DEFAULT_DATA_DIR)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = FOR_REPORT / "results" / "eeg_isc_v2_antialias" / out_dir

    participants = discover_participants(data_dir)
    if args.max_participants:
        participants = participants[: args.max_participants]

    all_records: list[dict] = []
    qc_rows: list[dict] = []
    all_notes: list[str] = []

    for p in participants:
        folder_name = p["folder_name"]
        print(f"\n=== {folder_name} ===")
        if p["practice_csv"] is None or p["vhdr"] is None:
            all_notes.append(f"{folder_name}: missing PsychoPy CSV or EEG — skipped")
            print("  skipped (missing data)")
            continue
        psy = load_psychopy_trials(p["practice_csv"])
        recs, summary, notes = preprocess_participant(p["vhdr"], psy, folder_name)
        all_records.extend(recs)
        if summary:
            qc_rows.append(summary)
        all_notes.extend(notes)
        print(f"  trials kept: {len(recs)}")

    index_df = save_trial_arrays(all_records, out_dir)
    index_path = out_dir / "trial_index.csv"
    index_df.to_csv(index_path, index=False)

    qc_df = pd.DataFrame(qc_rows)
    qc_path = out_dir / "preprocessing_qc.csv"
    qc_df.to_csv(qc_path, index=False)

    meta = {
        "pipeline": "ica_faster",
        "faster_thres": FASTER_THRES,
        "ica_metrics": ICA_METRICS,
        "filter_hz": [1.0, 40.0],
        "isc_antialias_hz": ISC_ANTIALIAS_HZ,
        "epoch_sec": EPOCH_SEC,
        "isc_sfreq": ISC_SFREQ,
        "note": "ICA/FASTER on 1-40 Hz; GFP/occ lowpassed at isc_antialias_hz before resample to isc_sfreq",
        "n_participants": len(qc_rows),
        "n_trials_kept": len(all_records),
        "notes": all_notes,
    }
    meta_path = out_dir / "preprocessing_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nWrote {index_path} ({len(index_df)} trials)")
    print(f"Wrote {qc_path}")
    print(f"Wrote {meta_path}")
    if not qc_df.empty:
        print(
            f"Epoch retention: {qc_df['n_trials_kept'].sum()}/{qc_df['n_epochs_created'].sum()} "
            f"({qc_df['n_trials_kept'].sum()/max(qc_df['n_epochs_created'].sum(),1):.0%})"
        )


if __name__ == "__main__":
    main()
