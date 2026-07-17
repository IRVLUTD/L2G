#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from typing import List, Dict, Any, Set, Tuple


def parse_image_id_from_file_name(file_name: str) -> int:
    base = os.path.basename(file_name)     # e.g., 00000.jpg
    stem, _ = os.path.splitext(base)       # e.g., 00000
    if re.fullmatch(r"\d+", stem):
        return int(stem)
    m = re.findall(r"(\d+)", stem)
    if not m:
        raise ValueError(f"Cannot parse image_id from file_name={file_name}")
    return int(m[-1])


def load_list_from_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["results", "annotations", "predictions", "instances"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"Unsupported JSON structure in {path}")


def allowed_pairs_from_results_file(results_path: str) -> Set[Tuple[int, int]]:
    items = load_list_from_json(results_path)
    allowed: Set[Tuple[int, int]] = set()
    for it in items:
        if "image_id" in it and "category_id" in it:
            try:
                img_id = int(it["image_id"])
                cat_id = int(it["category_id"])
                allowed.add((img_id, cat_id))
            except Exception:
                continue
    return allowed


def convert_item(item: dict, scale_value: int = 4, image_id_offset: int = 0) -> dict:
    image_id = parse_image_id_from_file_name(item["file_name"]) + image_id_offset
    category_id = int(item["category_id"]) - 1  # Category IDs start from 1; convert to 0-based

    out = {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": item.get("bbox", []),
        "score": float(item.get("score", 0.0)),
        "image_width": item.get("image_width", None),
        "image_height": item.get("image_height", None),
        "scale": scale_value,
    }

    # === Added: resize bbox and image size proportionally so that width becomes 2048 ===
    # Only image_width is used to compute the scaling ratio, and height is scaled accordingly
    iw = out.get("image_width")
    if isinstance(iw, (int, float)) and iw and out.get("bbox") and len(out["bbox"]) == 4:
        resize_num = iw / 2048.0  # original width / target width
        if resize_num != 0:
            factor = 1.0 / resize_num        # = 2048 / image_width
            x, y, w, h = out["bbox"]

            # Convert to float for safety
            x = float(x) * factor
            y = float(y) * factor
            w = float(w) * factor
            h = float(h) * factor
            out["bbox"] = [x, y, w, h]

            # Write back normalized width/height
            out["image_width"] = 2048
            ih = out.get("image_height", None)
            if isinstance(ih, (int, float)) and ih:
                out["image_height"] = int(round(float(ih) * factor))
            # If image_height is missing, do not write it

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Convert detection results to COCO-format, with optional filtering and bbox resize-to-width=2048."
    )

    parser.add_argument("--input", required=True, type=str, help="Input prediction JSON file")
    parser.add_argument("--output", required=True, type=str, help="Output COCO results JSON file")
    parser.add_argument("--scale", default=4, type=int, help="Scale value written into output (default: 4)")
    parser.add_argument("--results_file", default="", type=str,
                        help="Optional: coco_instances_results.json. Only keep (image_id, category_id) pairs present in this file")
    parser.add_argument("--start_image_id", type=int, default=None,
                        help="Optional: starting image_id (inclusive) for filtering")
    parser.add_argument("--end_image_id", type=int, default=None,
                        help="Optional: ending image_id (inclusive) for filtering")
    parser.add_argument("--image_id_offset", type=int, default=0,
                        help="Constant added to the parsed image_id (e.g., to avoid collisions "
                             "when merging scene folders whose query-image indices restart at 0).")

    args = parser.parse_args()

    # Validate image_id range
    if (args.start_image_id is not None) and (args.end_image_id is not None):
        if args.start_image_id > args.end_image_id:
            raise ValueError("--start_image_id cannot be larger than --end_image_id")

    preds = load_list_from_json(args.input)

    # Build allowed pair set (if results_file is provided)
    use_filter_pairs = bool(args.results_file and args.results_file.strip())
    allowed_pairs: Set[Tuple[int, int]] = set()

    if use_filter_pairs:
        allowed_pairs = allowed_pairs_from_results_file(args.results_file)
        if not allowed_pairs:
            print(f"⚠️ Warning: --results_file={args.results_file} was provided, but no valid (image_id, category_id) pairs were found.")

    converted = []
    total = 0

    for item in preds:
        if "file_name" not in item or "category_id" not in item:
            continue

        total += 1
        out = convert_item(item, scale_value=args.scale, image_id_offset=args.image_id_offset)

        # Range filtering (inclusive)
        if args.start_image_id is not None and out["image_id"] < args.start_image_id:
            continue
        if args.end_image_id is not None and out["image_id"] > args.end_image_id:
            continue

        # Pair filtering
        if use_filter_pairs:
            if (out["image_id"], out["category_id"]) not in allowed_pairs:
                continue

        converted.append(out)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"✅ Done. Read {total} items, kept {len(converted)} → {args.output}")

    if use_filter_pairs:
        print(f"ℹ️ Pair filtering based on: {args.results_file} (allowed pairs: {len(allowed_pairs)})")

    if args.start_image_id is not None or args.end_image_id is not None:
        print(f"ℹ️ image_id range: [{args.start_image_id if args.start_image_id is not None else '-∞'}, "
              f"{args.end_image_id if args.end_image_id is not None else '+∞'}] (inclusive)")


if __name__ == "__main__":
    main()