#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strictly merge per-scene prediction files into ONE flat list JSON.

- Input: a root directory containing subfolders 000001/, 000002/, ...
         each subfolder has a prediction file (default 'pred_results.json').
- Exact match to GT images[].file_name is required (no normalization).
- If any prediction's file_name is not in GT, raise ValueError.
- You can choose which scenes to include via --scenes (e.g., '1-7,13-21').

Output item format:
{
  "image_id": <from GT>, "category_id": ..., "bbox": [x,y,w,h], "score": <float>,
  "image_width": <from GT>, "image_height": <from GT>, "scale": 1, "id": <1..N>
}
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Set

SCENE_DIR_RE = re.compile(r"^\d{6}$")

def load_gt_index(gt_path: Path):
    data = json.loads(gt_path.read_text())
    idx = {}
    for im in data.get("images", []):
        fn = im.get("file_name")
        if isinstance(fn, str):
            idx[fn] = {"id": im["id"], "width": im.get("width"), "height": im.get("height")}
    if not idx:
        raise ValueError("GT images[] is empty or missing 'file_name'.")
    return idx

def find_scene_pred_files(root: Path, pred_filename: str):
    """
    Find files like <root>/<000001>/<pred_filename>, only for subdirs named 6 digits.
    Return a list of (scene_name, file_path) sorted by scene numeric order.
    """
    pairs = []
    for child in root.iterdir():
        if child.is_dir() and SCENE_DIR_RE.match(child.name):
            pf = child / pred_filename
            if pf.is_file():
                pairs.append((child.name, pf))
    pairs.sort(key=lambda x: int(x[0]))
    if not pairs:
        raise ValueError(f"No scene prediction files found under: {root} (looking for '{pred_filename}')")
    return pairs

def parse_scenes(arg: str) -> List[str]:
    """
    Parse scenes spec like:
      '1-6' -> ['000001', ... '000006']
      '1,3,6' -> ['000001','000003','000006']
      '1-3,5,8-10' -> combined
    """
    scenes_set: Set[int] = set()
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a); end = int(b)
            if start > end:
                start, end = end, start
            scenes_set.update(range(start, end + 1))
        else:
            scenes_set.add(int(part))
    return ["{:06d}".format(i) for i in sorted(scenes_set)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, type=Path, help="GT COCO json with images[].")
    ap.add_argument("--root", required=True, type=Path, help="Root folder containing 000001/, 000002/, ...")
    ap.add_argument("--out", required=True, type=Path, help="Output merged flat prediction json.")
    ap.add_argument("--pred-filename", default="pred_results_USE_PE_ADAPTER=True_USE_AUGMENTED_SAM=True.json",
                    help="Prediction file name inside each scene folder (default: pred_results.json)")
    ap.add_argument("--scenes", default=None,
                    help="Scenes to include, e.g. '1-7,13-21'. If omitted, include all found scenes.")
    args = ap.parse_args()

    gt_index = load_gt_index(args.gt)
    gt_names = set(gt_index.keys())

    # discover per-scene files
    scene_files = find_scene_pred_files(args.root, args.pred_filename)

    # optional: filter by --scenes
    if args.scenes:
        wanted = set(parse_scenes(args.scenes))
        scene_files = [pair for pair in scene_files if pair[0] in wanted]
        if not scene_files:
            raise ValueError(f"No matching scenes under root for --scenes '{args.scenes}'.")

    merged = []
    next_id = 1
    total_input = 0
    bad_names = set()
    per_scene_counts = []

    for scene, fpath in scene_files:
        arr = json.loads(fpath.read_text())
        if not isinstance(arr, list):
            raise ValueError(f"{fpath} is not a list of predictions.")
        kept = 0
        for rec in arr:
            fn = rec.get("file_name")
            total_input += 1
            if not isinstance(fn, str) or fn not in gt_names:
                bad_names.add(fn if isinstance(fn, str) else "<non-string>")
                continue  # collect first; error after loop
            info = gt_index[fn]
            merged.append({
                "image_id": info["id"],
                "category_id": rec["category_id"],
                "bbox": rec["bbox"],
                "score": float(rec.get("score", 0.0)),
                "image_width": info["width"],
                "image_height": info["height"],
                "scale": 1,
                "id": next_id
            })
            next_id += 1
            kept += 1
        per_scene_counts.append({"scene": scene, "inputs": len(arr), "kept": kept})

    if bad_names:
        sample = list(bad_names)[:20]
        raise ValueError(
            f"[Strict mode] Found {len(bad_names)} unmatched file_name(s) not present in GT. "
            f"Examples: {sample}"
        )

    args.out.write_text(json.dumps(merged, ensure_ascii=False))

    summary = {
        "gt": str(args.gt),
        "root": str(args.root),
        "output": str(args.out),
        "scenes_included": [s for s, _ in scene_files],
        "total_input_predictions": total_input,
        "total_output_predictions": len(merged),
        "note": "Strict mode: exact file_name match required. scale=1; id is incremental.",
        "per_scene_counts": per_scene_counts
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
