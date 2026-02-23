# utils/sam_utils.py

import os
import re
import glob
import torch
import torch.nn.functional as F
import numpy as np
import cv2


def load_object_tokens(model_sam2, dirpath: str, device: torch.device,):

    """
    SAM -> SAM* :
    Load per-object object_tokens from directory and merge into SAM model 

    Expected filename format:
        full_mask_tokens_<object_id>.pt

    Each file should contain either:
        - {"tokens": Tensor}  shape (1,K,C) or (K,C)
        - {"token":  Tensor}  shape (C,) or (1,C)

    This function updates model_sam2.sam_mask_decoder.full_mask_tokens in-place.
    """
    model_sam2.eval()
    dec = model_sam2.sam_mask_decoder
    num_objects_, K_, C_ = dec.full_mask_tokens.shape
    dtype_  = dec.full_mask_tokens.dtype 
    buf = torch.zeros((num_objects_, K_, C_), device=device, dtype=dtype_)

    pat = re.compile(r"full_mask_tokens_(\d+)\.pt$")
    files = sorted(glob.glob(os.path.join(dirpath, "full_mask_tokens_*.pt")))
    print(f"found {len(files)} files")

    loaded = 0
    seen_ids = set()   # save 1-based ID

    for f in files:
        m = pat.search(os.path.basename(f))
        if not m:
            print(f"[skip] bad filename: {f}")
            continue

        obj_id = int(m.group(1))          # 1-based
        if not (1 <= obj_id <= num_objects_):
            print(f"[skip] id {obj_id} out of range [1, {num_objects_}]")
            continue
        slot = obj_id - 1                 # to 0-based slot

        payload = torch.load(f, map_location="cpu")

        # "tokens" or "token"
        if "tokens" in payload:
            t = payload["tokens"]                 # (1,K,C) or (K,C)
            if t.dim() == 3 and t.shape[0] == 1:
                t = t.squeeze(0)                  # -> (K,C)
            elif t.dim() != 2:
                print(f"[skip] unsupported 'tokens' shape {tuple(t.shape)} in {f}")
                continue
        elif "token" in payload:
            t = payload["token"]                  # (C,) or (1,C)
            if t.dim() == 1:
                t = t.unsqueeze(0)                # -> (1,C)
            elif t.dim() != 2:
                print(f"[skip] unsupported 'token' shape {tuple(t.shape)} in {f}")
                continue
            t = t.expand(K_, -1)                  # -> (K,C)
        else:
            print(f"[skip] no 'tokens' or 'token' in {f}")
            continue

        t = t.to(device=device, dtype=dtype_)
        buf[slot].copy_(t)

        loaded += 1
        seen_ids.add(obj_id)
        print(f"[ok] loaded object {obj_id:06d}")

    with torch.no_grad():
        dec.full_mask_tokens.copy_(buf)

    missing = sorted(set(range(1, num_objects_ + 1)) - seen_ids)
    print(f"Done. Merged {loaded} objects into {tuple(dec.full_mask_tokens.shape)}")
    if missing:
        print(f"Missing object ids (kept zeros): {missing}")
    else:
        print("All object ids loaded.")


def clean_mask_dilate_seed_intersect(mask_in: np.ndarray,
                                     point_coords: np.ndarray,
                                     radius: int = 6) -> np.ndarray:
    m0 = (mask_in > 0).astype(np.uint8)
    H, W = m0.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    m_dil = cv2.dilate(m0, k, iterations=1)
    if m_dil.max() == 0:
        return np.zeros_like(m0, dtype=np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m_dil, connectivity=8)

    W = labels.shape[1]; H = labels.shape[0]
    candidate_labels = []
    for xy in np.asarray(point_coords, dtype=float):
        x, y = int(round(xy[0])), int(round(xy[1]))
        if 0 <= x < W and 0 <= y < H:
            lab = labels[y, x]
            if lab != 0:
                candidate_labels.append(lab)

    if not candidate_labels:
        keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    else:
        candidate_labels = list(set(candidate_labels))
        keep = max(candidate_labels, key=lambda l: stats[l, cv2.CC_STAT_AREA])

    m_keep_dil = (labels == keep).astype(np.uint8)
    m_final = (m_keep_dil & m0).astype(np.uint8)
    return m_final


def postprocess_mask_preserve_format(mask_in: np.ndarray,
                                     point_coords: np.ndarray,
                                     radius: int = 6) -> np.ndarray:
    """
    Clean noise while preserving the original dtype and value range
    of the input mask (0/1, 0/255, bool, or float).
    """
    orig_dtype = mask_in.dtype
    unique_vals = np.unique(mask_in)
    hi = float(unique_vals.max()) if unique_vals.size else 1.0
    # squeeze to 2D
    m2d = mask_in
    if m2d.ndim == 3 and m2d.shape[0] == 1:
        m2d = m2d[0]
    elif m2d.ndim == 3 and m2d.shape[-1] == 1:
        m2d = m2d[..., 0]
    # clean
    clean01 = clean_mask_dilate_seed_intersect(m2d, point_coords, radius=radius)  # 0/1 uint8
    if np.issubdtype(orig_dtype, np.bool_):
        out = clean01.astype(bool)
    elif np.issubdtype(orig_dtype, np.integer):
        if hi > 1:
            out = (clean01 * int(hi)).astype(orig_dtype)
        else:
            out = clean01.astype(orig_dtype)
    else:
        out = (clean01 * hi).astype(orig_dtype)
    # out = out[None, ...]
    return out


def _mask_np_to_right_patch_bool(mask_np: np.ndarray,
                                 Hr: int, Wr: int,
                                 patch_size: int,
                                 device: torch.device) -> torch.Tensor:
    """
    Convert a binary mask (Horig, Worig) from SAM2 to a boolean mask on the RIGHT patch grid (H2, W2).

    Steps:
      1) Resize to the DINO-resized size (Hr, Wr) with NEAREST (keeps hard edges).
      2) Average-pool with kernel=stride=patch_size to go to (H2, W2).
      3) Threshold at 0.5 to get boolean.
    """
    if mask_np is None:
        return None
    m = torch.from_numpy(mask_np.astype(np.float32))      # (Horig, Worig)
    m = m.unsqueeze(0).unsqueeze(0).to(device)            # (1,1,Horig,Worig)
    m_rs = F.interpolate(m, size=(Hr, Wr), mode="nearest")  # (1,1,Hr,Wr)
    pooled = F.avg_pool2d(m_rs, kernel_size=patch_size, stride=patch_size)  # (1,1,H2,W2)
    right_bool = (pooled.squeeze(0).squeeze(0) > 0.5)     # (H2, W2) bool
    return right_bool


def mean_similarity_for_mask(masks_i: np.ndarray,
                             heatmaps_b: torch.Tensor,
                             left_fg_flat: torch.Tensor,
                             Hr: int, Wr: int,
                             patch_size: int,
                             device: torch.device) -> float:
    """
    Compute the mean similarity between ALL left-foreground patches and ALL right-mask patches.

    Args:
      masks_i      : numpy mask from SAM2, shape (Horig, Worig), values {0,1}.
      heatmaps_b   : (K, H2, W2) tensor, similarities for one left template b.
      left_fg_flat : (K,) boolean tensor, foreground selection for the LEFT object (all patches in its mask).
      Hr, Wr       : height/width of the RIGHT tensor fed into DINO after resize.
      patch_size   : DINO patch size (e.g., 16).
      device       : torch device for temporary tensors.

    Returns:
      Mean similarity (float). Returns -inf if either side is empty.
    """
    if (masks_i is None) or (heatmaps_b is None):
        return float("-inf")

    right_bool = _mask_np_to_right_patch_bool(masks_i, Hr, Wr, patch_size, device)
    if right_bool is None:
        return float("-inf")

    if (not torch.any(left_fg_flat)) or (not torch.any(right_bool)):
        return float("-inf")

    K, H2, W2 = heatmaps_b.shape
    sims_flat = heatmaps_b.view(K, -1)      # (K, H2*W2)
    sel_r = right_bool.view(-1)             # (H2*W2)
    sub = sims_flat[left_fg_flat][:, sel_r] # (K_fg, N_right_mask)
    return float(sub.mean().item())