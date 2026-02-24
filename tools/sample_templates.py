import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def build_pairs_by_object(root: Path):
    """
    Build paired (rgb, mask) lists per object folder.
    Expected:
      root/rgb/<obj_id>/*
      root/mask/<obj_id>/*.png
    Pairing rule: same stem within the same obj_id folder.
    Returns:
      dict[obj_id] -> list[(rgb_path, mask_path)]
    """
    rgb_root = root / "rgb"
    mask_root = root / "mask"

    if not rgb_root.is_dir():
        raise FileNotFoundError(f"Missing rgb folder: {rgb_root}")
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Missing mask folder: {mask_root}")

    pairs_by_obj = defaultdict(list)
    obj_ids = sorted([p.name for p in rgb_root.iterdir() if p.is_dir()])

    for obj_id in obj_ids:
        rgb_dir = rgb_root / obj_id
        mask_dir = mask_root / obj_id
        if not mask_dir.is_dir():
            continue

        mask_map = {p.stem: p for p in mask_dir.glob("*.png")}

        for rgb_path in rgb_dir.iterdir():
            if not rgb_path.is_file():
                continue
            if rgb_path.suffix.lower() not in IMG_EXTS:
                continue

            m = mask_map.get(rgb_path.stem, None)
            if m is None:
                continue

            pairs_by_obj[obj_id].append((rgb_path, m))

    return pairs_by_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="../data/Templates", help="Path to templates root")
    parser.add_argument("--datasets", default="RoboTools", help="Dataset name prefix")
    parser.add_argument("--n", type=int, default=8, help="Number of samples per object")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--allow_smaller", action="store_true",
                        help="If set, objects with < n pairs will take all available pairs instead of error.")
    args = parser.parse_args()

    root = Path(f"{args.root}/{args.datasets}_all").resolve()
    out_dir = root.parent / f"{args.datasets}_{args.n}"

    rng = random.Random(args.seed)

    pairs_by_obj = build_pairs_by_object(root)
    if len(pairs_by_obj) == 0:
        raise RuntimeError(f"No paired data found under: {root}")

    # For each object folder, sample n pairs independently
    for obj_id in sorted(pairs_by_obj.keys()):
        pairs = pairs_by_obj[obj_id]
        if len(pairs) == 0:
            continue

        if len(pairs) < args.n and not args.allow_smaller:
            raise ValueError(
                f"Object {obj_id}: only {len(pairs)} pairs available, but n={args.n}. "
                f"Use --allow_smaller to bypass."
            )

        chosen = pairs if (len(pairs) < args.n and args.allow_smaller) else rng.sample(pairs, args.n)

        rgb_out = out_dir / "rgb" / obj_id
        mask_out = out_dir / "mask" / obj_id
        rgb_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        # Re-index from 000000.png within each object folder
        for idx, (rgb_path, mask_path) in enumerate(chosen):
            new_name = f"{idx:06d}.png"
            shutil.copy2(rgb_path, rgb_out / new_name)
            shutil.copy2(mask_path, mask_out / new_name)

    print(f"Done. Saved to: {out_dir}")


if __name__ == "__main__":
    main()