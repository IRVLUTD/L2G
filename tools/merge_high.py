#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


LIST_KEYS = ("results", "annotations", "predictions", "instances")


def load_items(path: Path) -> List[Dict[str, Any]]:
    """
    Load detection results from a JSON file.

    Supported structures:
    1. A top-level list.
    2. A dictionary containing a list under one of LIST_KEYS.
    """
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in LIST_KEYS:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(
        f"Unsupported JSON structure in {path}. "
        f"Expected a list or a dictionary containing one of: {LIST_KEYS}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge detection JSON files from numbered folders into one "
            "COCO-style results file."
        )
    )

    parser.add_argument(
        "--start_folder",
        required=True,
        type=int,
        help="First folder index, for example 1 for folder 000001.",
    )
    parser.add_argument(
        "--end_folder",
        required=True,
        type=int,
        help="Last folder index, for example 10 for folder 000010.",
    )
    parser.add_argument(
        "--input_file_name",
        required=True,
        type=str,
        help=(
            "JSON file name inside each numbered folder, for example "
            "coco_instances_results_converted.json."
        ),
    )
    parser.add_argument(
        "--out_file_name",
        required=True,
        type=str,
        help="Output path for the merged JSON file.",
    )
    parser.add_argument(
        "--root_dir",
        default=".",
        type=str,
        help=(
            "Directory containing the numbered folders. "
            "For example: ../Output/High_Resolution"
        ),
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip missing or unreadable files instead of stopping.",
    )

    args = parser.parse_args()

    if args.start_folder > args.end_folder:
        raise ValueError(
            "--start_folder must be less than or equal to --end_folder."
        )

    root_dir = Path(args.root_dir).expanduser()
    input_path_argument = Path(args.input_file_name)

    # Only a file name should be passed to --input_file_name.
    if input_path_argument.name != args.input_file_name:
        raise ValueError(
            "--input_file_name must contain only a file name, not a path.\n"
            f"Received: {args.input_file_name}\n"
            f"Use: --input_file_name {input_path_argument.name}\n"
            f"Set the parent directory with: --root_dir {root_dir}"
        )

    if not root_dir.exists():
        raise FileNotFoundError(
            f"Root directory does not exist: {root_dir.resolve()}"
        )

    if not root_dir.is_dir():
        raise NotADirectoryError(
            f"Root path is not a directory: {root_dir.resolve()}"
        )

    merged_items: List[Dict[str, Any]] = []
    missing_files: List[Path] = []
    failed_files: List[Path] = []

    total_read = 0
    processed_folders = 0

    for folder_index in range(args.start_folder, args.end_folder + 1):
        folder_name = f"{folder_index:06d}"
        json_path = root_dir / folder_name / args.input_file_name

        if not json_path.is_file():
            print(f"Missing file: {json_path}")
            missing_files.append(json_path)

            if not args.skip_missing:
                continue

            continue

        try:
            items = load_items(json_path)
        except Exception as error:
            print(f"Failed to read {json_path}: {error}")
            failed_files.append(json_path)

            if not args.skip_missing:
                continue

            continue

        merged_items.extend(items)
        total_read += len(items)
        processed_folders += 1

        print(f"Loaded {folder_name}: {len(items)} items")

    if (missing_files or failed_files) and not args.skip_missing:
        error_lines = [
            "Merge stopped because one or more input files could not be loaded.",
            f"Missing files: {len(missing_files)}",
            f"Failed files: {len(failed_files)}",
            "Use --skip_missing only if an incomplete merge is intentional.",
        ]
        raise RuntimeError("\n".join(error_lines))

    if not merged_items:
        raise RuntimeError(
            "No detection results were loaded. "
            "The output file was not created."
        )

    output_path = Path(args.out_file_name).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(merged_items, file, ensure_ascii=False, indent=2)

    print(
        f"Done. Processed {processed_folders} folders, "
        f"read {total_read} items, merged {len(merged_items)} items "
        f"into {output_path}"
    )


if __name__ == "__main__":
    main()
