# utilts.py
import os
from typing import Tuple, List

import torch
import numpy as np
from PIL import Image
from pathlib import Path

from .core.vision_encoder import pe, transforms








# -------------------------------
# Global, lazily initialized model
# -------------------------------
_PE_MODEL = None
_PE_PREPROCESS = None
_PE_DEVICE = None


def _init_pe():
    """Lazily initialize the PE model, preprocess, and device only once."""
    global _PE_MODEL, _PE_PREPROCESS, _PE_DEVICE
    if _PE_MODEL is not None:
        return

    # Select device
    if torch.cuda.is_available():
        _PE_DEVICE = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        _PE_DEVICE = torch.device("mps")
    else:
        _PE_DEVICE = torch.device("cpu")

    # Build the model
    _PE_MODEL = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True).to(_PE_DEVICE).eval()
    _PE_PREPROCESS = transforms.get_image_transform(_PE_MODEL.image_size)


def _enumerate_positions(total: int, win: int, stride: int) -> List[int]:
    """
    Generate start positions for a sliding window along one dimension (W or H).
    Uses stride steps, and if the final step would exceed the boundary, it also
    includes the last valid start so the window touches the boundary.

    Example:
      total=100, win=32, stride=16 -> [0, 16, 32, 52, 68] (where 68 = 100-32)
    """
    if win > total:
        # Window larger than the image; clamp to (0) and caller will crop (shouldn't happen normally).
        return [0]

    starts = []
    pos = 0
    while pos + win <= total:
        starts.append(pos)
        pos += stride

    last_start = total - win
    if starts and starts[-1] != last_start:
        starts.append(last_start)
    elif not starts:
        # Edge case: total == win
        starts = [0]
    return starts


def _load_pil_rgb(path: str) -> Image.Image:
    """Open an image as RGB."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _pil_to_unit_tensor(img_pil: Image.Image, image_size: int) -> torch.Tensor:
    """
    Match _load_img_as_tensor format:
      - RGB
      - resize to (image_size, image_size)
      - scale to [0,1]
      - return torch.float32 tensor of shape (C, H, W)
    """
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype != np.uint8:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype}")
    img_np = (img_np.astype(np.float32) / 255.0)  # [0,1], float32
    img = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()  # (C,H,W), float32
    return img


def pe_get_window(
    window_size: Tuple[int, int],
    window_stride: int,
    object_path: str,
    img_path: str,
    image_size: int,  # NEW: output tensor size to match _load_img_as_tensor
) -> Tuple[float, torch.Tensor, Tuple[int, int]]:
    """
    Slide a window, score with PE, and return:
      best_logit (float),
      best_window_tensor (torch.float32, (C,image_size,image_size), values in [0,1]),
      top_left_xy (x,y).

    - Scoring uses PE preprocess (model-normalized)
    - Returned tensor matches _load_img_as_tensor format
    """
    _init_pe()
    model = _PE_MODEL
    preprocess = _PE_PREPROCESS
    device = _PE_DEVICE

    win_h, win_w = int(window_size[0]), int(window_size[1])
    stride = int(window_stride)

    tmpl_img = _load_pil_rgb(object_path)
    src_img = _load_pil_rgb(img_path)
    W, H = src_img.size

    if win_w <= 0 or win_h <= 0:
        raise ValueError(f"Invalid window_size={window_size}")
    if stride <= 0:
        raise ValueError(f"Invalid window_stride={window_stride}")

    xs = _enumerate_positions(W, win_w, stride)
    ys = _enumerate_positions(H, win_h, stride)

    # Encode template (PE preprocess for scoring)
    with torch.no_grad():
        tmpl_tensor_pe = preprocess(tmpl_img).unsqueeze(0).to(device)
        tmpl_feat = model.encode_image(tmpl_tensor_pe)
        tmpl_feat = tmpl_feat / tmpl_feat.norm(dim=-1, keepdim=True)

    # Build two versions per window:
    #  - PE-preprocessed tensor for scoring (on device)
    #  - Unit-range float32 tensor (C,S,S) for RETURN (on CPU), matching _load_img_as_tensor
    pe_tensors = []
    unit_tensors = []
    coords = []

    for y in ys:
        for x in xs:
            crop = src_img.crop((x, y, x + win_w, y + win_h))

            # for scoring
            pe_tensors.append(preprocess(crop).unsqueeze(0))  # (1,C,S,S)

            # for return: match _load_img_as_tensor
            unit_tensors.append(_pil_to_unit_tensor(crop, image_size))  # (C,S,S), float32, [0,1]

            coords.append((x, y))

    with torch.no_grad():
        pe_batch = torch.cat(pe_tensors, dim=0).to(device)  # (N,C,S,S)
        feats = model.encode_image(pe_batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = (100.0 * (tmpl_feat @ feats.T)).squeeze(0)  # (N,)

    best_idx = int(torch.argmax(logits).item())
    best_logit = float(logits[best_idx].item())
    best_window_tensor = unit_tensors[best_idx]  # (C,image_size,image_size), float32, [0,1]
    top_left_xy = coords[best_idx]

    return best_logit, best_window_tensor, top_left_xy