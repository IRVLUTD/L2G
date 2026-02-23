# utils/visualization.py

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

from .build import tight_bbox_from_mask

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))  


def _normalize_binary_mask(mask_np: np.ndarray, target_hw: tuple[int, int] | None):
    """
    Normalize mask to shape (H, W) with dtype=uint8 {0,1}.
    Handles common shapes like (1,1,H,W), (1,H,W), (H,W,1), (H,W).
    Optionally resize to target_hw using nearest neighbor.
    """
    m = np.asarray(mask_np)

    # Squeeze leading singleton dims smartly
    if m.ndim == 4 and m.shape[0] == 1 and m.shape[1] == 1:
        m = m[0, 0]                  # (H, W)
    elif m.ndim == 3 and m.shape[0] == 1:
        m = m[0]                     # (H, W) or (H, W, C-1)
    elif m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]                # (H, W)
    elif m.ndim == 2:
        pass                         # already (H, W)
    else:
        raise ValueError(f"Unsupported mask shape: {mask_np.shape}, need (H,W) after squeeze")

    # Binarize to {0,1}
    m = (m.astype(np.float32) > 0.5).astype(np.uint8)

    # Resize to target size if provided and mismatched
    if target_hw is not None:
        Ht, Wt = target_hw
        if m.shape != (Ht, Wt):
            m = cv2.resize(m, (Wt, Ht), interpolation=cv2.INTER_NEAREST)

    return m  # (H, W) uint8 {0,1}

def save_mask_image(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    point_coords: np.ndarray | list | None,  # support (N,2) ndarray or list
    out_root: Path,
    left_index_b: int,         # b: 0-based
    right_path: Path,
    borders: bool = True,
):
    """
    Save the RIGHT image with a semi-transparent SAM2 mask overlay and all prompt points.

    Output path pattern:
        save_results/<b+1:06d>/<right_file_name>
    """
    # Prepare output path
    out_dir = out_root / f"{left_index_b+1:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / right_path.name  # keep the same filename as the right image

    # Figure size
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(image_np)
    ax.set_axis_off()

    # --- Normalize mask to (H, W) and align to image size ---
    H, W = image_np.shape[:2]
    mask = _normalize_binary_mask(mask_np, target_hw=(H, W))  # (H, W) uint8 {0,1}
    #print("normalized mask shape:", mask.shape)

    # --- Overlay as RGBA (only visible where mask==1) ---
    color = np.array([30/255.0, 144/255.0, 1.0, 0.6], dtype=np.float32)  # RGBA DodgerBlue with alpha
    overlay = (mask[..., None].astype(np.float32)) * color[None, None, :]  # (H, W, 4)
    ax.imshow(overlay)  # matplotlib expects HxW or HxWx3/4 (float in [0,1])

    # --- Optional borders using contours ---
    if borders and mask.any():
        # Find contours on 0/255 mask
        mask_u8_255 = (mask * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_u8_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = [cv2.approxPolyDP(c, epsilon=0.01 * cv2.arcLength(c, True), closed=True) for c in cnts]
        for c in cnts:
            c = c.squeeze(1)
            if c.ndim == 2 and c.shape[1] == 2:
                ax.plot(c[:, 0], c[:, 1], linewidth=2, color=(1, 1, 1, 0.7))

    # --- Draw prompt points (x=col, y=row) ---
    pts = None
    if point_coords is not None:
        if isinstance(point_coords, list):
            pcs = []
            for p in point_coords:
                if p is None:
                    continue
                arr = np.asarray(p, dtype=np.float32).reshape(-1, 2)
                if arr.size > 0:
                    pcs.append(arr)
            if len(pcs) > 0:
                pts = np.vstack(pcs)
        else:
            arr = np.asarray(point_coords, dtype=np.float32).reshape(-1, 2)
            if arr.size > 0:
                pts = arr

    if pts is not None and pts.size > 0:
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=36,
            c='green',
            marker='o',
            edgecolors='white',
            linewidths=0.8,
            zorder=10
        )

    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_mask_image_with_bbox(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    point_coords: np.ndarray | list | None,  # support (N,2) ndarray or list
    out_root: Path,
    left_index_b: int,         # b: 0-based
    right_path: Path,
    borders: bool = True,
):
    """
    Save the RIGHT image with a semi-transparent SAM2 mask overlay, all prompt points,
    and a red bbox (tight box from mask).

    Output path pattern:
        save_results/<b+1:06d>/<right_file_name>
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    

    # point_coords = np.array([[1072.9412,  731.25  ],
    #     [1095.5294,  686.25  ],
    #     [1095.5294,  753.75  ],
    #     [1140.7059,  731.25  ],]) 
 

    # Prepare output path
    out_dir = out_root / f"{left_index_b+1:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / right_path.name  # keep the same filename as the right image

    # Figure size
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(image_np)
    ax.set_axis_off()

    # --- Normalize mask to (H, W) and align to image size ---
    H, W = image_np.shape[:2]
    mask = _normalize_binary_mask(mask_np, target_hw=(H, W))  # (H, W) uint8 {0,1}
    #print("normalized mask shape:", mask.shape)

    # --- Overlay as RGBA (only visible where mask==1) ---
    color = np.array([30/255.0, 144/255.0, 1.0, 0.6], dtype=np.float32)  # RGBA DodgerBlue with alpha
    overlay = (mask[..., None].astype(np.float32)) * color[None, None, :]  # (H, W, 4)
    ax.imshow(overlay)  # matplotlib expects HxW or HxWx3/4 (float in [0,1])

    # --- NEW: draw tight bbox (red outline) ---
    bbox = tight_bbox_from_mask(mask.astype(np.uint8))
    if bbox is not None:
        x0, y0, x1, y1 = bbox  # (left, top, right, bottom) ; right/bottom are exclusive
        w, h = max(0, x1 - x0), max(0, y1 - y0)
        if w > 0 and h > 0:
            rect = patches.Rectangle(
                (x0, y0),
                w, h,
                linewidth=2,
                edgecolor=(1.0, 0.0, 0.0, 0.95),  # red with slight alpha
                facecolor="none"
            )
            ax.add_patch(rect)

    # --- Optional borders using contours ---
    if borders and mask.any():
        # Find contours on 0/255 mask
        mask_u8_255 = (mask * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_u8_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = [cv2.approxPolyDP(c, epsilon=0.01 * cv2.arcLength(c, True), closed=True) for c in cnts]
        for c in cnts:
            c = c.squeeze(1)
            if c.ndim == 2 and c.shape[1] == 2:
                ax.plot(c[:, 0], c[:, 1], linewidth=2, color=(1, 1, 1, 0.7))

    # --- Draw prompt points (x=col, y=row) ---
    pts = None
    #print("point_coords:",point_coords)
    if point_coords is not None:
        if isinstance(point_coords, list):
            pcs = []
            for p in point_coords:
                if p is None:
                    continue
                arr = np.asarray(p, dtype=np.float32).reshape(-1, 2)
                if arr.size > 0:
                    pcs.append(arr)
            if len(pcs) > 0:
                pts = np.vstack(pcs)
        else:
            arr = np.asarray(point_coords, dtype=np.float32).reshape(-1, 2)
            if arr.size > 0:
                pts = arr

    if pts is not None and pts.size > 0:
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=36,
            c='red',
            marker='o',
            edgecolors='white',
            linewidths=0.8,
            zorder=10
        )

    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig)



def save_mask_rgb_on_white(image_rgb: np.ndarray, mask: np.ndarray, save_path: str):
    """
    image_rgb: (H, W, 3) uint8
    mask: (H, W) bool or {0,1} or float
    output: background white, masked region keeps original RGB
    """
    assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3
    H, W, _ = image_rgb.shape

    # Normalize mask to boolean
    if mask.dtype != np.bool_:
        mask_bool = mask > 0.5
    else:
        mask_bool = mask

    # White background
    out = np.full((H, W, 3), 255, dtype=np.uint8)
    out[mask_bool] = image_rgb[mask_bool]

    Image.fromarray(out).save(save_path)