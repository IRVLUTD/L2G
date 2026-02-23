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

def convert_item(item: dict, scale_value: int = 4) -> dict:
    image_id = parse_image_id_from_file_name(item["file_name"])
    category_id = int(item["category_id"]) - 1  # 你的ID从1开始，这里转成0开始
    out = {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": item.get("bbox", []),
        "score": float(item.get("score", 0.0)),
        "image_width": item.get("image_width", None),
        "image_height": item.get("image_height", None),
        "scale": scale_value,
    }

    # === 新增：按目标宽度 2048 进行 bbox 与尺寸等比换算 ===
    # 只用 image_width 来计算比例，height 等比缩放
    iw = out.get("image_width")
    if isinstance(iw, (int, float)) and iw and out.get("bbox") and len(out["bbox"]) == 4:
        resize_num = iw / 2048.0  # 原始宽度 / 目标宽度
        if resize_num != 0:
            factor = 1.0 / resize_num        # = 2048 / image_width
            x, y, w, h = out["bbox"]
            # 安全起见转成 float
            x = float(x) * factor
            y = float(y) * factor
            w = float(w) * factor
            h = float(h) * factor
            out["bbox"] = [x, y, w, h]

            # 写回统一后的 width/height
            out["image_width"] = 2048
            ih = out.get("image_height", None)
            if isinstance(ih, (int, float)) and ih:
                out["image_height"] = int(round(float(ih) * factor))
            # 若没有 image_height，就不写

    return out

def main():
    parser = argparse.ArgumentParser(
        description="Convert detection results to COCO-format, with optional filtering and bbox resize-to-width=2048."
    )
    parser.add_argument("--input", required=True, type=str, help="输入预测 JSON 文件")
    parser.add_argument("--output", required=True, type=str, help="输出 COCO 结果 JSON 文件")
    parser.add_argument("--scale", default=4, type=int, help="写入的 scale 值 (默认 4)")
    parser.add_argument("--results_file", default="", type=str,
                        help="可选：coco_instances_results.json，仅保留其中存在的 (image_id, category_id)")
    parser.add_argument("--start_image_id", type=int, default=None,
                        help="可选：起始 image_id（含），用于区间过滤")
    parser.add_argument("--end_image_id", type=int, default=None,
                        help="可选：结束 image_id（含），用于区间过滤")
    args = parser.parse_args()

    # 校验区间
    if (args.start_image_id is not None) and (args.end_image_id is not None):
        if args.start_image_id > args.end_image_id:
            raise ValueError("--start_image_id 不能大于 --end_image_id")

    preds = load_list_from_json(args.input)

    # 构造允许集合（如果提供了 --results_file）
    use_filter_pairs = bool(args.results_file and args.results_file.strip())
    allowed_pairs: Set[Tuple[int, int]] = set()
    if use_filter_pairs:
        allowed_pairs = allowed_pairs_from_results_file(args.results_file)
        if not allowed_pairs:
            print(f"⚠️ 提示：提供了 --results_file={args.results_file}，但未解析到任何 (image_id, category_id)。")

    converted = []
    total = 0
    for item in preds:
        if "file_name" not in item or "category_id" not in item:
            continue
        total += 1
        out = convert_item(item, scale_value=args.scale)

        # 区间过滤（闭区间）
        if args.start_image_id is not None and out["image_id"] < args.start_image_id:
            continue
        if args.end_image_id is not None and out["image_id"] > args.end_image_id:
            continue

        # pair 过滤
        if use_filter_pairs:
            if (out["image_id"], out["category_id"]) not in allowed_pairs:
                continue

        converted.append(out)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"✅ Done. Read {total} items, kept {len(converted)} → {args.output}")
    if use_filter_pairs:
        print(f"ℹ️ pair 过滤依据：{args.results_file}（允许组合数：{len(allowed_pairs)}）")
    if args.start_image_id is not None or args.end_image_id is not None:
        print(f"ℹ️ image_id 区间：[{args.start_image_id if args.start_image_id is not None else '-∞'}, "
              f"{args.end_image_id if args.end_image_id is not None else '+∞'}]（闭区间）")

if __name__ == "__main__":
    main()