#!/usr/bin/env python3
"""
Extract low-level graph-image features for the 48 Sphynx experiment stimuli.

Digital ink / data utility / effectiveness follow the Sphinx extract_low_level_features
functions. Node count is taken from the stimulus catalog (graph_complexity), not from
the file path.

Run from this folder:
  python 01_extract_low_level_features.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    cv2 = None
try:
    from PIL import Image
except ImportError:
    Image = None

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "experiment" / "psychopy_experiment_stimuli_update_3_all_48_answered.csv"
DEFAULT_IMAGES = HERE / "images"
DEFAULT_OUT = HERE / "results" / "low_level" / "low_level_features.csv"

SUPPORTED_EXTENTIONS = (".png", ".jpg", ".jpeg", ".webp")
STANDARD_SIZE = (800, 600)

SEMI_TYPES = {"shape_nodes", "color_nodes", "size_nodes", "edge_width"}


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


def digital_ink(image_path: str) -> float:
    """
    Calculates digital ink as the sum of RGB values divided by the number of pixels.
    Digital ink = SumRGB(Pixel_i_j) / Number_of_pixels

    This measures the average RGB intensity per pixel.
    - White image (255,255,255) → ~765
    - Black image (0,0,0) → 0

    Args:
        image_path (str): Full path to the image

    Returns:
        float: Average RGB sum per pixel (0 to 765)
    """
    if cv2 is not None:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image at {image_path}")
        img = cv2.resize(img, STANDARD_SIZE, interpolation=cv2.INTER_AREA)
        sum_rgb = np.sum(img.astype(np.float64))
        num_pixels = img.shape[0] * img.shape[1]
        return sum_rgb / num_pixels

    if Image is None:
        raise ImportError("Need opencv-python (cv2) or Pillow to compute digital ink")
    # Channel-sum is identical for RGB and BGR; PIL fallback matches Sphinx digital-ink formula.
    pil = Image.open(image_path).convert("RGB")
    pil = pil.resize(STANDARD_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(pil, dtype=np.float64)
    sum_rgb = np.sum(arr)
    num_pixels = arr.shape[0] * arr.shape[1]
    return sum_rgb / num_pixels


def data_utility_from_nodes(nodes: int) -> int:
    """
    Data utility = number_of_nodes + number_of_connections = 2 * number_of_nodes + 1
    """
    return 2 * int(nodes) + 1


def effectiveness(data_util: float, dig_ink: float) -> Optional[float]:
    """
    Calculates the data visualization effectiveness.
    Effectiveness = data_utility / digital_ink

    Args:
        data_util (float): Data utility score
        dig_ink (float): Digital ink proportion

    Returns:
        float or None: Effectiveness score, or None if digital ink is zero
    """
    if dig_ink <= 0:
        return None
    return data_util / dig_ink


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


def extract_features(
    catalog_csv: Path,
    images_root: Path,
    output_csv: Path,
) -> Path:
    df = pd.read_csv(catalog_csv)
    results = []
    print(f"Catalog: {catalog_csv} ({len(df)} rows)")
    print(f"Images:  {images_root}")

    for i, row in df.iterrows():
        qid = int(row["question_id"]) if "question_id" in row and pd.notna(row["question_id"]) else i + 1
        graph_image = str(row["graph_image"])
        nodes = int(row["graph_complexity"])
        qlevel = int(row["question_complexity"])
        vis = str(row["visualisation_type"])
        adaptation = map_adaptation(vis)
        question = str(row.get("update_questions", ""))
        filename = Path(graph_image).name

        rec = {
            "question_id": qid,
            "graph_image": graph_image,
            "file_name": filename,
            "number_of_nodes": nodes,
            "question_level": qlevel,
            "visualisation_type": vis,
            "adaptation": adaptation,
            "question": question,
            "digital_ink": None,
            "data_utility": None,
            "effectiveness": None,
            "image_path": None,
            "error": None,
        }

        try:
            image_path = resolve_image_path(images_root, graph_image)
            rec["image_path"] = str(image_path)
            dig_ink = digital_ink(str(image_path))
            data_util = data_utility_from_nodes(nodes)
            eff = effectiveness(data_util, dig_ink)
            rec["digital_ink"] = round(float(dig_ink), 4)
            rec["data_utility"] = data_util
            rec["effectiveness"] = round(float(eff), 8) if eff is not None else None
            print(
                f"  [{qid:02d}] {filename}: nodes={nodes} adapt={adaptation} "
                f"ink={dig_ink:.2f} utility={data_util} eff={eff:.6f}"
            )
        except Exception as e:
            rec["error"] = str(e)
            print(f"  [{qid:02d}] ERROR {filename}: {e}")

        results.append(rec)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "graph_image",
        "file_name",
        "number_of_nodes",
        "question_level",
        "visualisation_type",
        "adaptation",
        "question",
        "digital_ink",
        "data_utility",
        "effectiveness",
        "image_path",
        "error",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    n_ok = sum(1 for r in results if r["error"] is None)
    print(f"\nProcessed {n_ok}/{len(results)} images")
    print(f"Results saved to: {output_csv}")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-level features for Sphynx-48 stimuli")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    extract_features(args.catalog, args.images, args.output)


if __name__ == "__main__":
    main()
