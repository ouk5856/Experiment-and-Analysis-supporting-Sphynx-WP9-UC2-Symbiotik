#!/usr/bin/env python3
"""
High-level saliency / scanpath features for the 48 Sphynx experiment stimuli.

Run ONE model per invocation (memory). Default device is cuda:1.
Checkpoints default to the remote-server paths from the old lightningnet project.

  python 02_extract_high_level_features.py --model visalformer
  python 02_extract_high_level_features.py --model sum
  python 02_extract_high_level_features.py --model clipgaze

Outputs (per model):
  results/high_level/{model}/outputs/     maps / scanpaths
  results/high_level/{model}/{model}_features.csv
  results/high_level/{model}/{model}_run.log
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "experiment" / "psychopy_experiment_stimuli_update_3_all_48_answered.csv"
DEFAULT_IMAGES = HERE / "images"
DEFAULT_MODELS = HERE / "models"
DEFAULT_DEVICE = "cuda:1"

REMOTE_CKPTS = {
    "visalformer": Path(
        "/data/gkoutr/BCI-RS/synthetic_eyetracking_works/non-priority/visalformer/Code/VisSalFormer_weights.tar"
    ),
    "sum": Path(
        "/data/gkoutr/BCI-RS/synthetic_eyetracking_works/non-priority/sum/SUM-main/net/pre_trained_weights/sum_model.pth"
    ),
    "clipgaze": Path(
        "/data/gkoutr/BCI-RS/synthetic_eyetracking_works/priority/clipgaze/CLIPGaze-main/checkpoint/CLIPGaze_TP.pkg"
    ),
}

MODEL_CODE_DIR = {
    "visalformer": "visalformer",
    "sum": "sum",
    "clipgaze": "clipgaze",
}

SEMI_TYPES = {"shape_nodes", "color_nodes", "size_nodes", "edge_width"}
SUM_CONDITION_UI = 3
TEXT_VERSION = 1


def map_adaptation(visualisation_type: str) -> str:
    vt = str(visualisation_type).strip().lower()
    if vt in ("non_adapted", "fully_adapted"):
        return vt
    if "non_adapted" in vt:
        return "non_adapted"
    if "fully_adapted" in vt:
        return "fully_adapted"
    if vt in SEMI_TYPES or "semi" in vt:
        return "semi_adapted"
    return "semi_adapted"


def resolve_image_path(images_root: Path, graph_image: str) -> Path:
    rel = Path(str(graph_image))
    candidates = [
        images_root / rel,
        images_root / rel.name,
        HERE / rel,
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Image not found for graph_image={graph_image}")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class RunLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, msg: str) -> None:
        line = f"[{utc_now()}] {msg}"
        print(line)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def extract_saliency_features(saliency: np.ndarray) -> Dict[str, float]:
    saliency = np.asarray(saliency, dtype=np.float64)
    if saliency.ndim > 2:
        saliency = np.squeeze(saliency)
    features = {
        "saliency_mean": float(np.mean(saliency)),
        "saliency_std": float(np.std(saliency)),
        "saliency_max": float(np.max(saliency)),
        "saliency_min": float(np.min(saliency)),
        "saliency_skewness": float(stats.skew(saliency.flatten())),
        "saliency_kurtosis": float(stats.kurtosis(saliency.flatten())),
    }

    sal_prob = saliency.flatten() / (np.sum(saliency) + 1e-8)
    sal_prob = sal_prob[sal_prob > 0]
    features["saliency_entropy"] = float(-np.sum(sal_prob * np.log2(sal_prob + 1e-8)))

    cy, cx = saliency.shape[0] // 2, saliency.shape[1] // 2
    y_coords, x_coords = np.meshgrid(
        np.arange(saliency.shape[0]),
        np.arange(saliency.shape[1]),
        indexing="ij",
    )
    distances = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
    max_dist = np.sqrt(cy ** 2 + cx ** 2)
    if max_dist > 0 and np.sum(saliency) > 0:
        norm_dist = distances / max_dist
        features["saliency_center_bias"] = float(np.sum(saliency * norm_dist) / np.sum(saliency))
    else:
        features["saliency_center_bias"] = 0.0

    if np.sum(saliency) > 0:
        weighted_y = np.sum(saliency * y_coords) / np.sum(saliency)
        weighted_x = np.sum(saliency * x_coords) / np.sum(saliency)
        spread = np.sqrt(
            np.sum(saliency * ((y_coords - weighted_y) ** 2 + (x_coords - weighted_x) ** 2))
            / np.sum(saliency)
        )
        features["saliency_spread"] = float(spread)
    else:
        features["saliency_spread"] = 0.0
    return features


def _scanpath_xy(fixations: np.ndarray) -> np.ndarray:
    """Return Nx2 positions. CLIPGaze stores [y, x, t]; generic scanpaths may be [x, y] or [x, y, t]."""
    fixations = np.asarray(fixations, dtype=np.float64)
    if fixations.ndim == 1:
        fixations = fixations.reshape(-1, 1)
    if fixations.shape[1] >= 3:
        # CLIPGaze: [y, x, t] — treat cols 1,0 as x,y for Euclidean length
        # Also works if [x, y, t] for length (same distances if we swap consistently).
        # Use first two columns as a 2D trajectory (order does not change length/complexity).
        return fixations[:, :2]
    if fixations.shape[1] >= 2:
        return fixations[:, :2]
    return np.zeros((0, 2), dtype=np.float64)


def extract_scanpath_features(
    fixations: Optional[np.ndarray],
    saliency: Optional[np.ndarray],
) -> Dict[str, float]:
    features = {
        "fixation_count": 0,
        "fixation_duration_mean": np.nan,
        "fixation_duration_std": np.nan,
        "scanpath_length": np.nan,
        "scanpath_complexity": np.nan,
        "mean_saccade_mag": np.nan,
        "cognitive_load": np.nan,
        "info_processing_idx": np.nan,
    }
    if fixations is None:
        return features
    fixations = np.asarray(fixations, dtype=np.float64)
    if fixations.size == 0:
        return features
    if fixations.ndim == 1:
        if fixations.size < 2:
            return features
        fixations = fixations.reshape(1, -1)

    features["fixation_count"] = int(len(fixations))

    if fixations.shape[1] >= 3:
        durations = []
        for fix in fixations:
            d = float(fix[2]) if float(fix[2]) > 0 else 0.3
            durations.append(d)
    else:
        durations = [0.3] * len(fixations)
    features["fixation_duration_mean"] = float(np.mean(durations))
    features["fixation_duration_std"] = float(np.std(durations))

    positions = _scanpath_xy(fixations)
    if len(positions) > 1:
        step = np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1))
        features["scanpath_length"] = float(np.sum(step))
        features["mean_saccade_mag"] = float(np.mean(step))
        straight_line = float(np.sqrt(np.sum((positions[-1] - positions[0]) ** 2)))
        features["scanpath_complexity"] = float(
            features["scanpath_length"] / straight_line if straight_line > 0 else 1.0
        )
        features["cognitive_load"] = float(
            features["fixation_duration_mean"] * features["scanpath_complexity"]
        )
        dispersion = float(np.std(positions, axis=0).mean())
        transitions = float(np.mean(step))
        denom = 1.0
        if saliency is not None:
            saliency = np.asarray(saliency)
            if saliency.ndim >= 2:
                denom = float(saliency.shape[0] * saliency.shape[1])
        features["info_processing_idx"] = float((dispersion * transitions) / (denom + 1e-8))
    return features


def empty_saliency_features() -> Dict[str, float]:
    keys = [
        "saliency_mean",
        "saliency_std",
        "saliency_max",
        "saliency_min",
        "saliency_skewness",
        "saliency_kurtosis",
        "saliency_entropy",
        "saliency_center_bias",
        "saliency_spread",
    ]
    return {k: np.nan for k in keys}


def find_first(out_dir: Path, pattern: str) -> Optional[Path]:
    hits = sorted(out_dir.rglob(pattern))
    return hits[0] if hits else None


def load_npy(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not path.is_file():
        return None
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None


def build_cmd(
    model: str,
    image_path: Path,
    ckpt: Path,
    output_dir: Path,
    device: str,
    question: str,
) -> List[str]:
    py = sys.executable
    if model == "visalformer":
        return [
            py,
            "standardized_inference.py",
            "--image",
            str(image_path),
            "--text",
            question,
            "--text_version",
            str(TEXT_VERSION),
            "--ckpt",
            str(ckpt),
            "--output_dir",
            str(output_dir),
            "--device",
            device,
        ]
    if model == "sum":
        return [
            py,
            "standardized_inference.py",
            "--image",
            str(image_path),
            "--condition",
            str(SUM_CONDITION_UI),
            "--ckpt",
            str(ckpt),
            "--output_dir",
            str(output_dir),
            "--device",
            device,
        ]
    if model == "clipgaze":
        return [
            py,
            "standardized_inference.py",
            "--image",
            str(image_path),
            "--text",
            question,
            "--text_version",
            str(TEXT_VERSION),
            "--ckpt",
            str(ckpt),
            "--output_dir",
            str(output_dir),
            "--device",
            device,
            "--num_samples",
            "1",
            "--saliency_method",
            "adaptive_temporal",
        ]
    raise ValueError(f"Unknown model: {model}")


def expected_subdir(model: str, image_stem: str) -> str:
    if model == "visalformer":
        return f"{image_stem}_visalformer_text_version_{TEXT_VERSION}"
    if model == "sum":
        return f"{image_stem}_sum"
    if model == "clipgaze":
        return f"{image_stem}_clipgaze_text_version_{TEXT_VERSION}"
    return image_stem


def run_one_model(
    model: str,
    catalog_csv: Path,
    images_root: Path,
    models_root: Path,
    ckpt: Path,
    device: str,
) -> Path:
    code_dir = models_root / MODEL_CODE_DIR[model]
    inf_script = code_dir / "standardized_inference.py"
    out_root = HERE / "results" / "high_level" / model
    maps_root = out_root / "outputs"
    feat_csv = out_root / f"{model}_features.csv"
    log_path = out_root / f"{model}_run.log"
    maps_root.mkdir(parents=True, exist_ok=True)

    log = RunLog(log_path)
    log.write(f"=== START model={model} ===")
    log.write(f"catalog={catalog_csv}")
    log.write(f"images_root={images_root}")
    log.write(f"code_dir={code_dir}")
    log.write(f"ckpt={ckpt}")
    log.write(f"device={device}")
    log.write(f"maps_root={maps_root}")
    log.write(f"features_csv={feat_csv}")
    if model == "sum":
        log.write(f"SUM condition={SUM_CONDITION_UI} (UI)")
    else:
        log.write("text input = catalog update_questions")

    if not inf_script.is_file():
        log.write(f"ERROR: missing {inf_script}")
        log.close()
        raise FileNotFoundError(inf_script)
    if not ckpt.is_file():
        log.write(f"ERROR: checkpoint not found: {ckpt}")
        log.write("Place weights at the remote path or pass --ckpt. Exiting without running inference.")
        log.close()
        print(f"Checkpoint missing for {model}: {ckpt}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(catalog_csv)
    rows: List[Dict[str, Any]] = []

    for i, crow in df.iterrows():
        qid = int(crow["question_id"]) if pd.notna(crow.get("question_id")) else i + 1
        graph_image = str(crow["graph_image"])
        nodes = int(crow["graph_complexity"])
        qlevel = int(crow["question_complexity"])
        vis = str(crow["visualisation_type"])
        adaptation = map_adaptation(vis)
        question = str(crow.get("update_questions", "") or "")
        filename = Path(graph_image).name
        stem = Path(graph_image).stem

        rec: Dict[str, Any] = {
            "question_id": qid,
            "model": model,
            "graph_image": graph_image,
            "file_name": filename,
            "number_of_nodes": nodes,
            "question_level": qlevel,
            "visualisation_type": vis,
            "adaptation": adaptation,
            "question": question,
            "device": device,
            "ckpt": str(ckpt),
            "input_image": "",
            "output_dir": "",
            "saliency_npy": "",
            "scanpath_npy": "",
            "status": "pending",
            "error": "",
            **empty_saliency_features(),
            **extract_scanpath_features(None, None),
        }

        try:
            image_path = resolve_image_path(images_root, graph_image).resolve()
            rec["input_image"] = str(image_path)
            stim_out = (maps_root / f"q{qid:02d}_{stem}").resolve()
            stim_out.mkdir(parents=True, exist_ok=True)
            rec["output_dir"] = str(stim_out)

            cmd = build_cmd(model, image_path, ckpt.resolve(), stim_out, device, question)
            log.write(
                f"INPUT qid={qid} image={image_path} question={question!r} "
                f"adapt={adaptation} nodes={nodes} Q={qlevel}"
            )
            log.write(f"CMD cwd={code_dir} :: {' '.join(cmd)}")

            proc = subprocess.run(
                cmd,
                cwd=str(code_dir),
                capture_output=True,
                text=True,
            )
            if proc.stdout:
                log.write("STDOUT:\n" + proc.stdout[-4000:])
            if proc.stderr:
                log.write("STDERR:\n" + proc.stderr[-4000:])
            if proc.returncode != 0:
                raise RuntimeError(f"inference exit {proc.returncode}")

            nested = stim_out / expected_subdir(model, stem)
            search_root = nested if nested.is_dir() else stim_out
            sal_path = find_first(search_root, "*saliency.npy")
            scan_path = find_first(search_root, "*scanpath.npy")
            rec["saliency_npy"] = str(sal_path) if sal_path else ""
            rec["scanpath_npy"] = str(scan_path) if scan_path else ""
            log.write(f"OUTPUT qid={qid} saliency={rec['saliency_npy']} scanpath={rec['scanpath_npy']}")

            sal = load_npy(sal_path)
            scan = load_npy(scan_path)
            if sal is not None:
                rec.update(extract_saliency_features(sal))
            if model == "clipgaze":
                rec.update(extract_scanpath_features(scan, sal))
            rec["status"] = "ok"
            log.write(f"STATUS qid={qid} ok")
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = str(e)
            log.write(f"STATUS qid={qid} ERROR {e}")

        rows.append(rec)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(feat_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    log.write(f"=== DONE model={model} ok={n_ok}/{len(rows)} features={feat_csv} ===")
    log.close()
    return feat_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-level saliency/scanpath features (one model per run)"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["visalformer", "sum", "clipgaze"],
        help="Run exactly one model (do not chain them in one process)",
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--ckpt", type=Path, default=None, help="Override remote checkpoint path")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Default: cuda:1")
    args = parser.parse_args()

    ckpt = args.ckpt if args.ckpt is not None else REMOTE_CKPTS[args.model]
    run_one_model(
        model=args.model,
        catalog_csv=args.catalog,
        images_root=args.images,
        models_root=args.models_root,
        ckpt=ckpt,
        device=args.device,
    )


if __name__ == "__main__":
    main()
