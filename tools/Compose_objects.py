#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compose single-object RGB/mask pairs onto backgrounds with 3 scenarios (each ~1/3 prob):
  (1) Only the target object
  (2) Target object + 2~4 other objects whose tight bboxes (from masks) are placed
      randomly around the target tight bbox (around its 4 sides), with:
        - inside_ratio >= 2/5
        - other objects are allowed to overlap each other
        - other objects' tight bboxes do NOT overlap the target tight bbox
  (3) Target object + 2~4 other objects with MUST-overlap, total overlap in (area/6, area/2),
      and 50% chance the target is in front; otherwise target is behind and occluded by a
      random non-empty subset of the other objects.

Common settings:
  - Random scale in [scale_min, scale_max]
  - 50% chance to apply Gaussian blur to RGB only (mask unchanged)
  - 50% chance to apply random rotation (RGB+mask) by a random angle in [0, 360)
  - Allow out-of-bounds placement; ensure target inside-ratio >= 1/3; others >= 2/5 in scenario 2

Inputs:
- objects_all/
    - rgb/000001..000020/000001.png..000024.png
    - mask/000001..000020/000001.png..000024.png  (single object, non-zero = foreground)
- backgrounds: new_datasets_10/  (JPEG images)

Outputs:
- out_root (e.g., objects_create/)
    - rgb/{obj_id}/000000.jpg ...
    - mask/{obj_id}/000000.png ...
- bbox_out_root (e.g., objects_create_bbox/)
    - rgb/{obj_id}/000000.jpg ...   # cropped by target tight bbox in canvas
    - mask/{obj_id}/000000.png ...  # same crop on final target mask
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageFilter

# # --- Pillow resample compatibility (Pillow >=10 vs <10) ---
# try:
#     RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
#     RESAMPLE_NEAREST = Image.Resampling.NEAREST
# except AttributeError:
#     RESAMPLE_LANCZOS = Image.LANCZOS
#     RESAMPLE_NEAREST = Image.NEAREST

# --- Pillow resample compatibility (Pillow >=10 vs <10) ---
try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS
    RESAMPLE_NEAREST = Image.NEAREST
    RESAMPLE_BICUBIC = Image.BICUBIC
# -------------------------
# Basic IO helpers
# -------------------------
def list_backgrounds(bg_dir: Path):
    exts = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    return sorted([p for p in bg_dir.iterdir() if p.suffix in exts], key=lambda p: p.name)


def load_object_pair(root_rgb: Path, root_mask: Path, obj_id: int, frame_idx: int):
    obj_name = f"{obj_id:06d}"
    #frame_name_rgb = f"{frame_idx:06d}.jpg"
    frame_name_rgb = f"{frame_idx:06d}.png"
    frame_name_mask = f"{frame_idx:06d}.png"
    rgb_path = root_rgb / obj_name / frame_name_rgb
    mask_path = root_mask / obj_name / frame_name_mask
    if not rgb_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"Missing rgb/mask: {rgb_path} or {mask_path}")
    rgb = Image.open(rgb_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    mask = (mask > 0).astype(np.uint8) * 255
    mask = Image.fromarray(mask, mode="L")
    return rgb, mask


def resize_pair(rgb: Image.Image, mask: Image.Image, scale: float):
    W, H = rgb.size
    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))
    rgb_r = rgb.resize((new_w, new_h), resample=RESAMPLE_LANCZOS)
    mask_r = mask.resize((new_w, new_h), resample=RESAMPLE_NEAREST)
    m = np.array(mask_r)
    m = (m > 0).astype(np.uint8) * 255
    mask_r = Image.fromarray(m, mode="L")
    return rgb_r, mask_r


def maybe_blur_rgb_only(
    obj_rgb: Image.Image,
    apply_prob: float,
    blur_min: float,
    blur_max: float,
    rng: random.Random,
):
    if rng.random() < apply_prob:
        radius = rng.uniform(blur_min, blur_max)
        return obj_rgb.filter(ImageFilter.GaussianBlur(radius=radius))
    return obj_rgb


def maybe_rotate_pair(
    rgb: Image.Image,
    mask: Image.Image,
    apply_prob: float,
    rng: random.Random,
):
    """
    Randomly rotate RGB+mask by a random angle in [0, 360), `apply_prob` probability.
    Use BICUBIC for RGB and NEAREST for mask (Pillow does not allow LANCZOS in rotate).
    """
    if rng.random() >= apply_prob:
        return rgb, mask

    angle = rng.uniform(0.0, 360.0)


    rgb_rot = rgb.rotate(angle, resample=RESAMPLE_BICUBIC, expand=True)
    mask_rot = mask.rotate(angle, resample=RESAMPLE_NEAREST, expand=True)

    m = np.array(mask_rot)
    m = (m > 0).astype(np.uint8) * 255
    mask_rot = Image.fromarray(m, mode="L")
    return rgb_rot, mask_rot


def full_transform_pair(
    rgb: Image.Image,
    mask: Image.Image,
    scale_min: float,
    scale_max: float,
    blur_prob: float,
    blur_min: float,
    blur_max: float,
    rot_prob: float,
    rng: random.Random,
):
    """
    Apply: scale -> (optional) rotation -> (optional) Gaussian blur on RGB.
    """
    scale = rng.uniform(scale_min, scale_max)
    rgb, mask = resize_pair(rgb, mask, scale)
    rgb, mask = maybe_rotate_pair(rgb, mask, rot_prob, rng)
    rgb = maybe_blur_rgb_only(rgb, blur_prob, blur_min, blur_max, rng)
    return rgb, mask


# -------------------------
# Geometry / mask utilities
# -------------------------
def mask_area(mask_arr: np.ndarray) -> int:
    return int((mask_arr > 0).sum())


def area_of_overlap_on_canvas(
    A_mask: np.ndarray,
    A_xy: Tuple[int, int],
    B_mask: np.ndarray,
    B_xy: Tuple[int, int],
    canvas_wh: Tuple[int, int],
) -> int:
    """Count overlapping foreground pixels of A and B when pasted on a canvas."""
    W, H = canvas_wh
    Ax, Ay = A_xy
    Bx, By = B_xy

    x0 = max(max(0, Ax), max(0, Bx))
    y0 = max(max(0, Ay), max(0, By))
    x1 = min(min(W, Ax + A_mask.shape[1]), min(W, Bx + B_mask.shape[1]))
    y1 = min(min(H, Ay + A_mask.shape[0]), min(H, By + B_mask.shape[0]))
    if x1 <= x0 or y1 <= y0:
        return 0

    A_sub = A_mask[y0 - Ay : y1 - Ay, x0 - Ax : x1 - Ax]
    B_sub = B_mask[y0 - By : y1 - By, x0 - Bx : x1 - Bx]

    return int(((A_sub > 0) & (B_sub > 0)).sum())

def inside_ratio_on_canvas(
    mask_arr: np.ndarray,
    xy: Tuple[int, int],
    canvas_wh: Tuple[int, int],
) -> float:
    """fraction of mask pixels that land inside the canvas when placed at xy"""
    W, H = canvas_wh
    x, y = xy
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + mask_arr.shape[1])
    y1 = min(H, y + mask_arr.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    sub = mask_arr[y0 - y : y1 - y, x0 - x : x1 - x]
    total = mask_area(mask_arr)
    if total == 0:
        return 0.0
    return float((sub > 0).sum()) / float(total)


def sample_position_with_constraint(
    bg_w,
    bg_h,
    obj_w,
    obj_h,
    mask_arr,
    min_inside_ratio=1 / 3,
    max_trials=80,
    rng: random.Random = None,
):
    if rng is None:
        rng = random
    total_fg = mask_area(mask_arr)
    if total_fg == 0:
        return (max(0, (bg_w - obj_w) // 2), max(0, (bg_h - obj_h) // 2))
    for _ in range(max_trials):
        x = rng.randint(-obj_w, bg_w)
        y = rng.randint(-obj_h, bg_h)
        if inside_ratio_on_canvas(mask_arr, (x, y), (bg_w, bg_h)) >= min_inside_ratio:
            return x, y
    return ((bg_w - obj_w) // 2, (bg_h - obj_h) // 2)


def tight_bbox_from_mask(mask_arr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute tight bbox [x0,y0,x1,y1) from mask (mask>0). Return None if empty.
    """
    ys, xs = np.where(mask_arr > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


def sample_adjacent_near_tight_bbox(
    bg_wh: Tuple[int, int],
    tgt_xy: Tuple[int, int],
    tgt_bbox_local: Tuple[int, int, int, int],
    other_mask: np.ndarray,
    other_bbox_local: Tuple[int, int, int, int],
    min_inside_ratio: float,
    gap_thr: int,
    max_trials: int,
    rng: random.Random,
) -> Tuple[int, int]:
    """
    Place 'other' randomly around the target's tight bbox (4 sides), such that:
      - other is near/around the target bbox with a gap in [0, gap_thr]
      - other tight bbox does NOT overlap the target tight bbox
      - inside_ratio_on_canvas(other_mask, (ox,oy)) >= min_inside_ratio
    Overlap between different 'other' objects is allowed; only relation to target bbox
    is constrained here.
    """
    W, H = bg_wh

    tx, ty = tgt_xy
    tbx0, tby0, tbx1, tby1 = tgt_bbox_local
    obx0, oby0, obx1, oby1 = other_bbox_local

    # target global tight bbox
    tgt_gx0 = tx + tbx0
    tgt_gy0 = ty + tby0
    tgt_gx1 = tx + tbx1
    tgt_gy1 = ty + tby1

    o_w = obx1 - obx0
    o_h = oby1 - oby0

    for _ in range(max_trials):
        side = rng.randint(0, 3)  # 0:left, 1:right, 2:top, 3:bottom
        gap = rng.randint(0, max(0, gap_thr))
        

        if side == 0:
            # other on the LEFT side of target bbox, with a small gap
            other_gx1 = tgt_gx0 - gap
            other_gx0 = other_gx1 - o_w
            # vertical overlap with target bbox
            low = tgt_gy0 - (o_h - 1)
            high = tgt_gy1 - 1
            if high < low:
                continue
            other_gy0 = rng.randint(low, high)
            other_gy1 = other_gy0 + o_h

        elif side == 1:
            # other on the RIGHT side
            other_gx0 = tgt_gx1 + gap
            other_gx1 = other_gx0 + o_w
            low = tgt_gy0 - (o_h - 1)
            high = tgt_gy1 - 1
            if high < low:
                continue
            other_gy0 = rng.randint(low, high)
            other_gy1 = other_gy0 + o_h

        elif side == 2:
            # other on the TOP side
            other_gy1 = tgt_gy0 - gap
            other_gy0 = other_gy1 - o_h
            low = tgt_gx0 - (o_w - 1)
            high = tgt_gx1 - 1
            if high < low:
                continue
            other_gx0 = rng.randint(low, high)
            other_gx1 = other_gx0 + o_w

        else:
            # other on the BOTTOM side
            other_gy0 = tgt_gy1 + gap
            other_gy1 = other_gy0 + o_h
            low = tgt_gx0 - (o_w - 1)
            high = tgt_gx1 - 1
            if high < low:
                continue
            other_gx0 = rng.randint(low, high)
            other_gx1 = other_gx0 + o_w

        # bbox non-overlap check 
        inter_w = min(tgt_gx1, other_gx1) - max(tgt_gx0, other_gx0)
        inter_h = min(tgt_gy1, other_gy1) - max(tgt_gy0, other_gy0)
        if inter_w > 0 and inter_h > 0:
            # bboxes overlap -> reject
            continue

        # convert global bbox -> top-left of full other_mask
        ox = other_gx0 - obx0
        oy = other_gy0 - oby0

        # inside ratio constraint
        if inside_ratio_on_canvas(other_mask, (ox, oy), (W, H)) < min_inside_ratio:
            continue

        return ox, oy

    # fallback: simply put it near bottom-right of target bbox
    ox = tgt_gx1 - obx0 + 1
    oy = tgt_gy1 - oby0 + 1
    return ox, oy


def sample_position_with_overlap_range(
    bg_wh: Tuple[int, int],
    tgt_xy: Tuple[int, int],
    tgt_mask: np.ndarray,
    other_mask: np.ndarray,
    min_overlap: int,
    max_overlap: int,
    max_trials: int,
    rng: random.Random,
) -> Tuple[int, int, int]:
    """
    Sample (x,y) for 'other' so that its overlap with target is within [min_overlap, max_overlap].
    Also require >= 1/5 of the other mask inside canvas.
    Returns (x, y, overlap).
    """
    W, H = bg_wh
    min_overlap = max(0, int(min_overlap))
    max_overlap = max(0, int(max_overlap))
    if max_overlap < min_overlap:
        min_overlap, max_overlap = max_overlap, min_overlap

    best_xy = (0, 0)
    best_ov = -1

    for _ in range(max_trials):
        x = rng.randint(-other_mask.shape[1], W)
        y = rng.randint(-other_mask.shape[0], H)

        if inside_ratio_on_canvas(other_mask, (x, y), (W, H)) < (1 / 5):
            continue

        ov = area_of_overlap_on_canvas(tgt_mask, tgt_xy, other_mask, (x, y), (W, H))
        if min_overlap <= ov <= max_overlap:
            return (x, y, ov)

        if ov > best_ov:
            best_xy = (x, y)
            best_ov = ov

    return (best_xy[0], best_xy[1], max(0, best_ov))


# -------------------------
# Compositing helpers
# -------------------------
def paste_with_mask(
    background: Image.Image,
    obj_rgb: Image.Image,
    obj_mask: Image.Image,
    x: int,
    y: int,
):
    bg = background.copy()
    bg_w, bg_h = bg.size
    mask_canvas = Image.new("L", (bg_w, bg_h), 0)
    bg.paste(obj_rgb, (x, y), obj_mask)
    mask_canvas.paste(obj_mask, (x, y))
    return bg, mask_canvas


def composite_layers(
    base_bg: Image.Image,
    ordered_rgbs_and_masks_and_xy: List[Tuple[Image.Image, Image.Image, Tuple[int, int]]],
) -> Tuple[Image.Image, Image.Image]:
    """ordered list is drawn back -> front, returns (composed_rgb, union_mask_of_all_objects)."""
    composed = base_bg.copy()
    bg_w, bg_h = base_bg.size
    union_mask = Image.new("L", (bg_w, bg_h), 0)
    for rgb, msk, (x, y) in ordered_rgbs_and_masks_and_xy:
        composed.paste(rgb, (x, y), msk)
        union_mask.paste(msk, (x, y), msk)
    return composed, union_mask


def apply_front_back_to_target_mask(
    target_mask_canvas: np.ndarray,
    occluder_masks_canvas: List[np.ndarray],
    target_in_front: bool,
) -> np.ndarray:
    """If target_in_front: unchanged; else subtract occluders' union."""
    if target_in_front or not occluder_masks_canvas:
        return target_mask_canvas
    occ_union = np.zeros_like(target_mask_canvas, dtype=np.uint8)
    for m in occluder_masks_canvas:
        occ_union = np.maximum(occ_union, (m > 0).astype(np.uint8) * 255)
    out = (target_mask_canvas > 0).astype(np.uint8) * 255
    out[(occ_union > 0)] = 0
    return out


def crop_by_mask_tight_bbox(
    image: Image.Image,
    mask_arr: np.ndarray,
) -> Optional[Image.Image]:
    """
    Crop image by tight bbox of mask_arr>0. Return None if empty.
    """
    bbox = tight_bbox_from_mask(mask_arr)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    return image.crop((x0, y0, x1, y1))


# -------------------------
# Per-object pipeline with 3 scenarios
# -------------------------
def process_for_object(
    obj_id: int,
    bg_paths,
    rgb_root: Path,
    mask_root: Path,
    out_root: Path,
    bbox_out_root: Path,
    scale_min: float,
    scale_max: float,
    blur_prob: float,
    blur_min: float,
    blur_max: float,
    rot_prob: float,
    all_obj_ids: List[int],
    rng: random.Random,
    max_nums: int,
    num_epochs: int = 1,   
):
    out_rgb_dir = out_root / "rgb" / f"{obj_id:06d}"
    out_msk_dir = out_root / "mask" / f"{obj_id:06d}"
    out_rgb_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)

    bbox_rgb_dir = bbox_out_root / "rgb" / f"{obj_id:06d}"
    bbox_msk_dir = bbox_out_root / "mask" / f"{obj_id:06d}"
    bbox_rgb_dir.mkdir(parents=True, exist_ok=True)
    bbox_msk_dir.mkdir(parents=True, exist_ok=True)

    idx = 0  

    for epoch in range(num_epochs):
        bg_paths_shuffled = list(bg_paths)
        rng.shuffle(bg_paths_shuffled)

        #for bg_path in bg_paths_shuffled:
        max_bg_per_object = int(max_nums)  
        for bg_path in bg_paths_shuffled[:max_bg_per_object]:
            with Image.open(bg_path) as bg_im:
                bg = bg_im.convert("RGB")
            bg_w, bg_h = bg.size
            canvas_wh = (bg_w, bg_h)

            # ---- target placement ----
            frame_idx = rng.randint(1, 8)  # [1,6]
            #frame_idx = rng.randint(0, 99)
            tgt_rgb_raw, tgt_mask_raw = load_object_pair(rgb_root, mask_root, obj_id, frame_idx)
            tgt_rgb, tgt_mask = full_transform_pair(
                tgt_rgb_raw,
                tgt_mask_raw,
                scale_min,
                scale_max,
                blur_prob,
                blur_min,
                blur_max,
                rot_prob,
                rng,
            )
            tgt_arr = np.array(tgt_mask)
            tgt_xy = sample_position_with_constraint(
                bg_w,
                bg_h,
                tgt_rgb.size[0],
                tgt_rgb.size[1],
                tgt_arr,
                min_inside_ratio=1 / 3,
                max_trials=100,
                rng=rng,
            )

            # local tight bbox of target
            tgt_bbox_local = tight_bbox_from_mask(tgt_arr)
            if tgt_bbox_local is None:
                # degenerate, skip this sample
                continue

            scenario_pick = rng.random()

            # --- Make probs 1/3, 1/3, 1/3 ---
            if scenario_pick < 1 / 3:
                # -------- Scenario 1: only target --------
                composed, _ = paste_with_mask(bg, tgt_rgb, tgt_mask, *tgt_xy)
                tgt_canvas = Image.new("L", (bg_w, bg_h), 0)
                tgt_canvas.paste(tgt_mask, tgt_xy)
                tgt_canvas_np_full = (np.array(tgt_canvas) > 0).astype(np.uint8) * 255
                final_target_mask = tgt_canvas_np_full

            elif scenario_pick < 2 / 3:
                # -------- Scenario 2: tight adjacency around target tight bbox, others can overlap each other --------
                k = rng.randint(2, 4)
                #k = rng.randint(1, 2)
                candidate_ids = [i for i in all_obj_ids if i != obj_id]
                other_ids = rng.sample(candidate_ids, k=min(k, len(candidate_ids)))

                placed = [] 
                t_h = tgt_arr.shape[0]
                t_w = tgt_arr.shape[1]
                #gap_thr = max(2, int(0.05 * max(t_w, t_h)))  
                gap_thr = 4  

                for oid in other_ids:
                    #fidx = rng.randint(1, 24)
                    fidx = rng.randint(1, 6)
                    o_rgb_raw, o_msk_raw = load_object_pair(rgb_root, mask_root, oid, fidx)
                    o_rgb, o_msk = full_transform_pair(
                        o_rgb_raw,
                        o_msk_raw,
                        scale_min,
                        scale_max,
                        blur_prob,
                        blur_min,
                        blur_max,
                        rot_prob,
                        rng,
                    )
                    o_arr = np.array(o_msk)
                    o_bbox_local = tight_bbox_from_mask(o_arr)
                    if o_bbox_local is None:
                        continue

                    xy = sample_adjacent_near_tight_bbox(
                        canvas_wh,
                        tgt_xy,
                        tgt_bbox_local,
                        o_arr,
                        o_bbox_local,
                        min_inside_ratio=2 / 5,  
                        gap_thr=gap_thr,
                        max_trials=120,
                        rng=rng,
                    )

                    placed.append((o_rgb, o_arr, xy))

                # draw: others (back) -> target (front)
                draw_list = [
                    (o_rgb, Image.fromarray(o_arr, mode="L"), xy)
                    for (o_rgb, o_arr, xy) in placed
                ]
                draw_list.append((tgt_rgb, tgt_mask, tgt_xy))
                composed, _ = composite_layers(bg, draw_list)

                tgt_canvas = Image.new("L", (bg_w, bg_h), 0)
                tgt_canvas.paste(tgt_mask, tgt_xy)
                tgt_canvas_np_full = (np.array(tgt_canvas) > 0).astype(np.uint8) * 255
                final_target_mask = tgt_canvas_np_full

            else:
                # -------- Scenario 3: MUST overlap; total overlap in (area/6, area/2) --------
                k = rng.randint(2, 4)
                #k = rng.randint(1, 2)
                candidate_ids = [i for i in all_obj_ids if i != obj_id]
                other_ids = rng.sample(candidate_ids, k=min(k, len(candidate_ids)))

                tgt_area = mask_area(tgt_arr)
                if tgt_area == 0:
                    # degenerate
                    continue

                min_overlap_total = int(np.ceil(tgt_area / 6.0))   # strictly > area/6
                max_overlap_total = int(np.floor(tgt_area / 2.0))  # strictly < area/2

                placed = []
                overlap_acc = 0

                # distribute desired per-object overlap as guidance
                if len(other_ids) > 0:
                    per_min = max(0, int(min_overlap_total / len(other_ids) * 0.5))
                    per_max = max(per_min, int(max_overlap_total / len(other_ids) * 1.2))
                else:
                    per_min, per_max = 0, 0

                for oid in other_ids:
                    #fidx = rng.randint(1, 24)
                    fidx = rng.randint(1, 6)
                    o_rgb_raw, o_msk_raw = load_object_pair(rgb_root, mask_root, oid, fidx)
                    o_rgb, o_msk = full_transform_pair(
                        o_rgb_raw,
                        o_msk_raw,
                        scale_min,
                        scale_max,
                        blur_prob,
                        blur_min,
                        blur_max,
                        rot_prob,
                        rng,
                    )
                    o_arr = np.array(o_msk)

                    remain_max = max(0, max_overlap_total - overlap_acc)
                    lo = max(0, min(per_min, remain_max))
                    hi = max(0, min(per_max, remain_max))
                    if hi < lo:
                        lo, hi = 0, remain_max

                    x, y, ov = sample_position_with_overlap_range(
                        (bg_w, bg_h),
                        tgt_xy,
                        tgt_arr,
                        o_arr,
                        lo,
                        hi,
                        max_trials=150,
                        rng=rng,
                    )
                    overlap_acc += ov
                    placed.append((o_rgb, o_arr, (x, y)))

                # fine-tune total overlap into (min, max)
                adjust_trials = 120
                while (
                    adjust_trials > 0
                    and not (min_overlap_total < overlap_acc < max_overlap_total)
                    and len(placed) > 0
                ):
                    idx_pick_inner = rng.randrange(len(placed))
                    o_rgb, o_arr, (x_old, y_old) = placed[idx_pick_inner]
                    old_ov = area_of_overlap_on_canvas(
                        tgt_arr, tgt_xy, o_arr, (x_old, y_old), (bg_w, bg_h)
                    )
                    overlap_acc -= old_ov

                    need_lo = max(0, min_overlap_total - overlap_acc + 1)   # ensure strictly greater than 1/6
                    need_hi = max(0, max_overlap_total - overlap_acc - 1)   # ensure strictly less than 1/2
                    if need_hi < need_lo:
                        need_lo = 0
                        need_hi = max(0, max_overlap_total - overlap_acc)

                    x_new, y_new, new_ov = sample_position_with_overlap_range(
                        (bg_w, bg_h),
                        tgt_xy,
                        tgt_arr,
                        o_arr,
                        need_lo,
                        need_hi,
                        max_trials=180,
                        rng=rng,
                    )
                    placed[idx_pick_inner] = (o_rgb, o_arr, (x_new, y_new))
                    overlap_acc += new_ov
                    adjust_trials -= 1

                # front/back for target
                target_in_front = (rng.random() < 0.5)
                occluder_indices = []
                if not target_in_front and len(placed) > 0:
                    count = rng.randint(1, len(placed))
                    occluder_indices = rng.sample(list(range(len(placed))), count)

                # draw order
                draw_list = []
                if target_in_front:
                    for (o_rgb, o_arr, xy) in placed:
                        draw_list.append((o_rgb, Image.fromarray(o_arr, mode="L"), xy))
                    draw_list.append((tgt_rgb, tgt_mask, tgt_xy))
                else:
                    non_occ, occ = [], []
                    for i, (o_rgb, o_arr, xy) in enumerate(placed):
                        if i in occluder_indices:
                            occ.append((o_rgb, o_arr, xy))
                        else:
                            non_occ.append((o_rgb, o_arr, xy))
                    for (o_rgb, o_arr, xy) in non_occ:
                        draw_list.append((o_rgb, Image.fromarray(o_arr, mode="L"), xy))
                    draw_list.append((tgt_rgb, tgt_mask, tgt_xy))
                    for (o_rgb, o_arr, xy) in occ:
                        draw_list.append((o_rgb, Image.fromarray(o_arr, mode="L"), xy))

                composed, _ = composite_layers(bg, draw_list)

                # final target mask considering occlusion
                tgt_canvas = Image.new("L", (bg_w, bg_h), 0)
                tgt_canvas.paste(tgt_mask, tgt_xy)
                tgt_canvas_np_full = (np.array(tgt_canvas) > 0).astype(np.uint8) * 255

                occluder_masks_canvas = []
                if not target_in_front:
                    for i, (o_rgb, o_arr, xy) in enumerate(placed):
                        if i in occluder_indices:
                            canvas = Image.new("L", (bg_w, bg_h), 0)
                            canvas.paste(Image.fromarray(o_arr, mode="L"), xy)
                            occluder_masks_canvas.append(
                                (np.array(canvas) > 0).astype(np.uint8) * 255
                            )

                final_target_mask = apply_front_back_to_target_mask(
                    tgt_canvas_np_full, occluder_masks_canvas, target_in_front
                )

            # ---- save outputs (per target object id) ----
            out_rgb = out_rgb_dir / f"{idx:06d}.jpg"
            out_msk = out_msk_dir / f"{idx:06d}.png"



            #save
            composed.save(out_rgb, format="JPEG", quality=95, subsampling=0, optimize=True)
            #save


            Image.fromarray(
                (final_target_mask > 0).astype(np.uint8) * 255, mode="L"
            ).save(out_msk)

            # ---- save bbox-cropped target image+mask (by target full mask tight bbox) ----
            tgt_canvas_np_for_bbox = tgt_canvas_np_full  # use original target mask on canvas
            bbox_rgb = crop_by_mask_tight_bbox(composed, tgt_canvas_np_for_bbox)
            bbox_msk = crop_by_mask_tight_bbox(
                Image.fromarray((final_target_mask > 0).astype(np.uint8) * 255, mode="L"),
                (final_target_mask > 0).astype(np.uint8),
            )

            if bbox_rgb is not None:
                bbox_rgb_path = bbox_rgb_dir / f"{idx:06d}.jpg"
                bbox_rgb.save(
                    bbox_rgb_path, format="JPEG", quality=95, subsampling=0, optimize=True
                )
            if bbox_msk is not None:
                bbox_msk_path = bbox_msk_dir / f"{idx:06d}.png"
                bbox_msk.save(bbox_msk_path)

            idx += 1

    print(
        f"[Obj {obj_id:06d}] Wrote {idx} pairs -> "
        f"{out_rgb_dir} / {out_msk_dir} and {bbox_rgb_dir} / {bbox_msk_dir}"
    )

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compose objects with 3 scenarios (1/3 each): "
            "only target / tight adjacent others around target bbox / MUST-overlap with front-back."
        )
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="How many epochs to iterate over all backgrounds for each object.",
    )
    parser.add_argument(
        "--objects-root",
        type=Path,
        default=Path("objects_all"),
        help="Root folder containing rgb/ and mask/",
    )
    parser.add_argument(
        "--backgrounds",
        type=Path,
        default=Path("Backgrounds_2048"),
        help="Folder containing background JPEGs",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root for objects_create (default: sibling of backgrounds)",
    )
    parser.add_argument(
        "--bbox-out-root",
        type=Path,
        default=None,
        help="Output root for bbox crops (default: out-root + '_bbox')",
    )
    parser.add_argument(
        "--start-object-id",
        type=int,
        default=1,
        help="Start object id (inclusive)",
    )
    parser.add_argument(
        "--end-object-id",
        type=int,
        default=20,
        help="End object id (inclusive)",
    )
    parser.add_argument("--scale-min", type=float, default=1.2)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--blur-prob", type=float, default=0.4)
    parser.add_argument("--blur-min", type=float, default=0.8)
    parser.add_argument("--blur-max", type=float, default=2.2)
    parser.add_argument("--max_nums", type=int, default=200)
    parser.add_argument(
        "--rot-prob",
        type=float,
        default=0.5,
        help="Probability to apply random rotation to RGB+mask",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if (
        args.scale_min <= 0
        or args.scale_max <= 0
        or args.scale_min > args.scale_max
    ):
        raise SystemExit("[Error] scale range invalid.")
    if (
        args.blur_min < 0
        or args.blur_max < 0
        or args.blur_min > args.blur_max
    ):
        raise SystemExit("[Error] blur radius range invalid.")
    if not (0.0 <= args.rot_prob <= 1.0):
        raise SystemExit("[Error] rot-prob must be in [0,1].")

    rng = random.Random(args.seed) if args.seed is not None else random

    bg_dir = args.backgrounds.resolve()
    if args.out_root is None:
        #out_root = bg_dir.parent / "HOCAP_objects_create"
        out_root = bg_dir.parent / "HOCAP_create_0117"
    else:
        out_root = args.out_root.resolve()

    if args.bbox_out_root is None:
        bbox_out_root = out_root.parent / f"{out_root.name}_bbox"
    else:
        bbox_out_root = args.bbox_out_root.resolve()

    rgb_root = (args.objects_root / "rgb").resolve()
    mask_root = (args.objects_root / "mask").resolve()
    if not rgb_root.exists() or not mask_root.exists():
        raise SystemExit(
            f"[Error] objects_all/rgb or objects_all/mask not found under {args.objects_root}"
        )

    bg_paths = list_backgrounds(bg_dir)
    if not bg_paths:
        raise SystemExit(f"[Error] No background JPEGs found in {bg_dir}")

    start_id = int(args.start_object_id)
    end_id = int(args.end_object_id)
    if start_id < 1 or end_id < start_id:
        raise SystemExit("[Error] invalid object-id range.")

    all_obj_ids = list(range(start_id, end_id + 1))
    print("all_obj_ids:", all_obj_ids)

    print(f"[Info] Backgrounds: {len(bg_paths)}")
    print(f"[Info] Objects: {start_id:06d}..{end_id:06d}")
    print(
        f"[Info] Scale: [{args.scale_min}, {args.scale_max}]  "
        f"Blur: p={args.blur_prob}, r=[{args.blur_min},{args.blur_max}]  "
        f"Rot: p={args.rot_prob}"
    )
    print(f"[Info] Output root: {out_root}")
    print(f"[Info] BBox Output root: {bbox_out_root}")

    for obj_id in all_obj_ids:
        process_for_object(
            obj_id,
            bg_paths,
            rgb_root,
            mask_root,
            out_root,
            bbox_out_root,
            args.scale_min,
            args.scale_max,
            args.blur_prob,
            args.blur_min,
            args.blur_max,
            args.rot_prob,
            all_obj_ids,
            rng,
            max_nums=args.max_nums,
            num_epochs=args.epochs,
        )


if __name__ == "__main__":
    main()
