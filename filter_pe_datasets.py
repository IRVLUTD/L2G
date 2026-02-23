#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean mismatched rgb/mask images.

ROOT_RGB = ../SSD2/High_datasets/objects_create_high_bbox/rgb
ROOT_MSK = ../SSD2/High_datasets/objects_create_high_bbox/mask

For each object folder (e.g. 000001):
- Match files by stem (filename without extension)
- If a stem exists only in rgb or only in mask, delete those files
- Keep only rgb/mask pairs with matching stems on both sides

Usage:
    python clean_mismatched_pairs.py --dry-run   # only print, do NOT delete
    python clean_mismatched_pairs.py             # actually delete

"""

import argparse
from pathlib import Path
from typing import Dict, List, Set

ROOT_RGB = Path("../SSD2/High_datasets/objects_create_high_bbox/rgb")
ROOT_MSK = Path("../SSD2/High_datasets/objects_create_high_bbox/mask")

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images_by_stem(pdir: Path) -> Dict[str, List[Path]]:
    """
    List images under `pdir`, group by stem (filename without extension).
    Returns: {stem: [paths...]}
    """
    out: Dict[str, List[Path]] = {}
    if not pdir.is_dir():
        return out

    for p in sorted(pdir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VALID_EXTS:
            continue
        out.setdefault(p.stem, []).append(p)
    return out


def clean_mismatched_for_object(
    oid: str,
    root_rgb: Path,
    root_msk: Path,
    dry_run: bool = True,
) -> None:
    """
    For a single object id (e.g. '000001'):
    - Find unmatched stems between rgb and mask
    - Delete corresponding files if not dry_run
    """
    rgb_dir = root_rgb / oid
    msk_dir = root_msk / oid

    if not rgb_dir.is_dir() or not msk_dir.is_dir():
        print(f"[WARN] Skip oid={oid}: rgb_dir or msk_dir missing")
        return

    rgb_map = list_images_by_stem(rgb_dir)
    msk_map = list_images_by_stem(msk_dir)

    rgb_stems: Set[str] = set(rgb_map.keys())
    msk_stems: Set[str] = set(msk_map.keys())

    only_rgb = sorted(rgb_stems - msk_stems)
    only_msk = sorted(msk_stems - rgb_stems)
    common   = sorted(rgb_stems & msk_stems)

    n_rgb = sum(len(v) for v in rgb_map.values())
    n_msk = sum(len(v) for v in msk_map.values())

    if not only_rgb and not only_msk:
        print(f"[OK]   oid={oid}: all matched (rgb={n_rgb}, msk={n_msk}, pairs={len(common)})")
        return

    print(f"[CLEAN] oid={oid}: rgb={n_rgb}, msk={n_msk}, "
          f"pairs={len(common)}, rgb_only_stems={len(only_rgb)}, msk_only_stems={len(only_msk)}")

    # Delete rgb-only files
    for stem in only_rgb:
        for p in rgb_map.get(stem, []):
            if dry_run:
                print(f"  [DRY-RUN] Would delete RGB: {p}")
            else:
                print(f"  [DEL] RGB: {p}")
                try:
                    p.unlink()
                except Exception as e:
                    print(f"    [ERROR] failed to delete {p}: {e}")

    # Delete mask-only files
    for stem in only_msk:
        for p in msk_map.get(stem, []):
            if dry_run:
                print(f"  [DRY-RUN] Would delete MSK: {p}")
            else:
                print(f"  [DEL] MSK: {p}")
                try:
                    p.unlink()
                except Exception as e:
                    print(f"    [ERROR] failed to delete {p}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean mismatched rgb/mask images by stem under objects_create_high_bbox."
    )
    parser.add_argument(
        "--root-rgb",
        type=str,
        default=str(ROOT_RGB),
        help="Root directory for RGB images (default: %(default)s)",
    )
    parser.add_argument(
        "--root-msk",
        type=str,
        default=str(ROOT_MSK),
        help="Root directory for mask images (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do NOT actually delete files, only print what would be deleted.",
    )
    args = parser.parse_args()

    root_rgb = Path(args.root_rgb)
    root_msk = Path(args.root_msk)

    if not root_rgb.is_dir():
        raise SystemExit(f"[FATAL] RGB root does not exist or is not a directory: {root_rgb}")
    if not root_msk.is_dir():
        raise SystemExit(f"[FATAL] MSK root does not exist or is not a directory: {root_msk}")

    # Object ids: union of subdirectory names in rgb & mask
    rgb_oids = {d.name for d in root_rgb.iterdir() if d.is_dir()}
    msk_oids = {d.name for d in root_msk.iterdir() if d.is_dir()}
    all_oids = sorted(rgb_oids | msk_oids)

    print(f"[INFO] Found {len(all_oids)} object folders (rgb={len(rgb_oids)}, msk={len(msk_oids)})")
    print(f"[INFO] dry_run={args.dry_run}")

    for oid in all_oids:
        clean_mismatched_for_object(oid, root_rgb, root_msk, dry_run=args.dry_run)

    print("[DONE] Cleaning finished.")


if __name__ == "__main__":
    main()
