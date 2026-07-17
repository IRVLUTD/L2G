# utils/image.py
import numpy as np


def crop_around_point(image_right, point_coords, crop_size=(1536, 2048), image_np=None):
    """
    Crop a fixed-size window around a point, while staying inside image bounds.

    Args:
        image_right: np.ndarray, shape (H, W, 3), original image.
        point_coords: array-like, (2,) or (1, 2) or (N, 2), in (x, y) format in original image.
        crop_size: (crop_h, crop_w), height and width of the crop.
        image_np: optional pre-converted np.ndarray of image_right (H,W,3). Pass this when
            calling repeatedly for many points on the same image, to avoid re-converting
            the (potentially large) PIL image on every call.

    Returns:
        image_scale_4: np.ndarray, cropped image of shape (crop_h, crop_w, 3)
        new_point_coords: np.ndarray, same shape as input point_coords,
                          but in the coordinate system of image_scale_4
        crop_box: (y0, x0, y1, x1) in original image coordinates
    """
    # --- Parse input ---
    img = image_np if image_np is not None else np.array(image_right)
    H, W = img.shape[:2]
    crop_h, crop_w = crop_size

    # Ensure array
    pc = np.asarray(point_coords, dtype=np.float32)
    if pc.ndim == 1:
        # (2,) -> (1,2)
        pc = pc[None, :]

    # Only use the first point to decide crop center
    cx, cy = pc[0]  # (x, y)

    # Clip center to be inside image (just in case)
    cx = float(np.clip(cx, 0, W - 1))
    cy = float(np.clip(cy, 0, H - 1))

    # --- Compute desired top-left before clamping ---
    x0 = int(round(cx - crop_w / 2.0))
    y0 = int(round(cy - crop_h / 2.0))

    # --- Clamp to keep crop fully inside image ---
    # Ensure crop size is not larger than image size
    if crop_w > W or crop_h > H:
        raise ValueError(
            f"Crop size {crop_size} is larger than image size {(H, W)}."
        )

    x0 = max(0, min(x0, W - crop_w))
    y0 = max(0, min(y0, H - crop_h))

    x1 = x0 + crop_w
    y1 = y0 + crop_h

    # --- Crop image ---
    image_scale_4 = img[y0:y1, x0:x1].copy()

    # --- Transform point coords into crop coordinate system ---
    new_pc = pc.copy()
    new_pc[:, 0] = new_pc[:, 0] - x0  # x' = x - x0
    new_pc[:, 1] = new_pc[:, 1] - y0  # y' = y - y0

    crop_box = (y0, x0, y1, x1)
    return image_scale_4, new_pc, crop_box