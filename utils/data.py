# utils/data.py
from PIL import Image
import numpy as np
import torch
import glob, os
import math
from pathlib import Path
from torchvision.transforms.functional import pil_to_tensor

def load_rgb(p: Path) -> Image.Image:
    """Load an image and convert to RGB (3 channels)."""
    return Image.open(p).convert("RGB")

def load_mask(p: Path, match_size_to=None) -> Image.Image:
    """Load a mask; prefer alpha channel if present, otherwise convert to grayscale ('L'). Optionally resize."""
    m = Image.open(p)
    m = (m.getchannel("A") if "A" in m.getbands() else m.convert("L"))
    if match_size_to and m.size != match_size_to:
        m = m.resize(match_size_to, resample=Image.NEAREST)
    return m

def count_object_tokens(dirpath: str) -> int:
    files = glob.glob(os.path.join(dirpath, "full_mask_tokens_*.pt"))
    return len(files)

def stride_filter_mask(
    sel_row: torch.Tensor,
    H1: int,
    W1: int,
    m: int,
    keep_edges: bool = False,
    rows_grid: torch.Tensor | None = None,
    cols_grid: torch.Tensor | None = None,
    Filter: bool = True,
):
    """
    Row-then-column sparsification INSIDE the foreground mask (no union).
    Stride is computed as: stride = max(1, int(sqrt(n_fg / max(int(m), 1)))).

    Steps:
      1) Row pass: for each row, split its foreground indices into ~stride-sized chunks
         and keep the middle index of each chunk.
      2) Column pass: run the same mid-pick over columns, but ONLY on the kept set from (1).
         (This overwrites the row pass; no union.)
      3) Optionally add back outer-border foreground if keep_edges=True.

    Parameters
    ----------
    sel_row : (K,) bool
        Foreground selection on the left patch grid (flattened, row-major).
    H1, W1 : int
        Patch-grid height/width.
    m : int
        Target factor; stride = max(1, floor(sqrt(n_fg / max(1, m)))).
    keep_edges : bool
        If True, add back foreground points on the outer border (after both passes).
    rows_grid, cols_grid : (unused; kept for signature compatibility)
    Filter : bool
        If False, return sel_row unchanged.

    Returns
    -------
    keep_mask : (K,) bool
        Final kept indices after row-then-column sparsification (and optional edges).
    stride_used : int
        The stride actually used.
    """
    device = sel_row.device
    K = H1 * W1
    sel2d = sel_row.view(H1, W1)

    if not Filter:
        return sel_row.clone(), 1

    # Compute stride from the ORIGINAL formula (do not change)
    n_fg = int(sel_row.sum().item())
    if n_fg == 0:
        return torch.zeros(K, dtype=torch.bool, device=device), 1

    stride = max(1, float(math.sqrt(n_fg / float(max(int(m), 1)))))

    # -------- Pass 1: ROW-wise mid-pick --------
    keep_rows = torch.zeros((H1, W1), dtype=torch.bool, device=device)
    for r in range(H1):
        cols = torch.nonzero(sel2d[r], as_tuple=False).squeeze(1)  # foreground cols in this row
        k = int(cols.numel())
        if k == 0:
            continue
        # number of representatives for this row
        n_rep = max(1, round(math.ceil(k / stride)))
        # chunk boundaries over [0, k)
        boundaries = torch.linspace(0, k, steps=n_rep + 1, device=device).floor().long()
        for i in range(n_rep):
            start = int(boundaries[i].item())
            end   = int(boundaries[i + 1].item()) - 1
            if end < start:
                continue
            mid = (start + end) // 2
            c = int(cols[mid].item())
            keep_rows[r, c] = True

    # Safety: intersect with original foreground
    keep_rows &= sel2d

    # -------- Pass 2: COLUMN-wise mid-pick on the kept set (overwrite) --------
    keep_cols = torch.zeros((H1, W1), dtype=torch.bool, device=device)
    for c in range(W1):
        rows = torch.nonzero(keep_rows[:, c], as_tuple=False).squeeze(1)
        k = int(rows.numel())
        if k == 0:
            continue
        n_rep = max(1, math.ceil(k / stride))
        boundaries = torch.linspace(0, k, steps=n_rep + 1, device=device).floor().long()
        for i in range(n_rep):
            start = int(boundaries[i].item())
            end   = int(boundaries[i + 1].item()) - 1
            if end < start:
                continue
            mid = (start + end) // 2
            r = int(rows[mid].item())
            keep_cols[r, c] = True

    keep2d = keep_cols & sel2d

    # -------- Optional: add back outer-border foreground AFTER both passes --------
    if keep_edges:
        edge_mask = torch.zeros_like(sel2d)
        if H1 > 0:
            edge_mask[0, :]  = True
            edge_mask[-1, :] = True
        if W1 > 0:
            edge_mask[:, 0]  = True
            edge_mask[:, -1] = True
        keep2d |= (edge_mask & sel2d)

    # Fallback: if empty but foreground exists, keep the median foreground
    if (not torch.any(keep2d)) and torch.any(sel2d):
        ys, xs = torch.nonzero(sel2d, as_tuple=True)
        mid = ys.numel() // 2
        keep2d[ys[mid], xs[mid]] = True
        # keep stride as-is; or set to 1 if you want to signal fallback

    keep_mask = keep2d.view(-1)
    return keep_mask, stride


def load_left_groups(group_root_rgb: Path, group_root_msk: Path):
    """
    Read the following directory structure:
      group_root_rgb/
        000001/*.png|jpg
        000002/*.png|jpg
        ...
      group_root_msk/
        000001/*.png|jpg
        000002/*.png|jpg
        ...
    Return：
      groups: List[List[Tuple[Path, Path]]]  # Each object corresponds to a list, where each element is a pair (img, mask)
      num_objects
      seq_len: Number of images per object (must be the same)
    """
    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

    def list_images(pdir: Path):
        return sorted([p for p in pdir.iterdir() if p.suffix.lower() in IMG_EXT])

    groups = []
    object_ids = sorted([d.name for d in group_root_rgb.iterdir() if d.is_dir()])
    assert len(object_ids) > 0, f"No object folders found in {group_root_rgb}"
    seq_len_ref = None
    for oid in object_ids:
        rgb_dir = group_root_rgb / oid
        msk_dir = group_root_msk / oid
        assert rgb_dir.is_dir() and msk_dir.is_dir(), f"Missing rgb/mask dir for {oid}"

        rgb_list = list_images(rgb_dir)
        msk_list = list_images(msk_dir)
        assert len(rgb_list) == len(msk_list) and len(rgb_list) > 0, f"{oid}: rgb/mask count mismatch or empty"

        pairs = list(zip(rgb_list, msk_list))
        groups.append(pairs)

        if seq_len_ref is None:
            seq_len_ref = len(pairs)
        else:
            assert len(pairs) == seq_len_ref, "All objects must have the same number of images"

    return groups, len(groups), seq_len_ref


def crops_to_padded_batch(pil_crops, x0y0_list, H_can, W_can, device, dtype=torch.float32):
    """
    Zero-pad multiple cropped patches (already intersected with the image)
    to a unified size (H_can, W_can) without resizing.

    Parameters:
    pil_crops : list[PIL.Image]
        Each element is a valid crop after intersection with the image
        (xs0:xs1, ys0:ys1).

    x0y0_list : list[(x0_int, y0_int, xs0, ys0)]
        x0_int, y0_int are the integer top-left coordinates of the target 3×3 window.
        xs0, ys0 are the top-left coordinates of the actual cropped region
        (after intersection with the image).

    H_can, W_can : int
        Target unified window size (in pixels).

    device : torch.device
        The device where the output tensor will be allocated.

    dtype : torch.dtype
        Output tensor data type (default: float32).

    Returns:
    batch : torch.Tensor
        Shape (N, 3, H_can, W_can), values in [0, 1], located on the specified device.
    """
    N = len(pil_crops)
    batch = torch.zeros((N, 3, H_can, W_can), device=device, dtype=dtype)

    for i, (crop, meta) in enumerate(zip(pil_crops, x0y0_list)):
        x0_int, y0_int, xs0, ys0 = meta
        # Placement offset: position of the actual crop's top-left corner within the target window coordinate system
        dx = int(xs0 - x0_int)
        dy = int(ys0 - y0_int)
        dx = max(0, min(dx, W_can - 1))
        dy = max(0, min(dy, H_can - 1))

        t = pil_to_tensor(crop).to(dtype=dtype, device=device) / 255.0  # (3,h,w)
        h, w = int(t.shape[-2]), int(t.shape[-1])

        hh = min(h, H_can - dy)
        ww = min(w, W_can - dx)
        if hh > 0 and ww > 0:
            batch[i, :, dy:dy+hh, dx:dx+ww] = t[:, 0:hh, 0:ww]

    return batch

