#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a residual MLP adapter on frozen pe_model features with flexible crops.

A: bbox-tight crop around mask (fixed)
B: random choice from up to 4 crop types:
   - square with side ratio n/48 of image (n in {3,5,7}), center sampled within mask
   - bbox-tight crop (same as A)
Large squares are filtered out if area > area_filter_ratio * bbox_area.

Positive: same object id; Negative: different object id.
Loss: NT-Xent (multi-positive friendly), **A-only adapter, A→B**.

Author: you + ChatGPT
"""

import os
import math
import random
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

# 你环境里的 pe_model
from perception_models.core.vision_encoder import pe, transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# === 可视化配置 ===

import matplotlib
# matplotlib.use("Agg") 


import matplotlib.pyplot as plt

VIS_EVERY_STEPS = 10           # 每隔多少个 step 可视化一次
VIS_MAX_SHOW = 8               # 每次最多展示多少对
VIS_SAVE_DIR = Path("debug_vis_01_1116")

# =========================
# Config
# =========================
SEED = 3
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda:0")        
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("[pe_adapter] Using device:", device)

# Data roots (按你的数据布局)
# ROOT_RGB = Path("../SSD2/Dinov3_features/objects_all/rgb")
# ROOT_MSK = Path("../SSD2/Dinov3_features/objects_all/mask")

ROOT_RGB = Path("../SSD2/High_datasets/objects_create_high_bbox/rgb")
ROOT_MSK = Path("../SSD2/High_datasets/objects_create_high_bbox/mask")

# Train hyper-params
EPOCHS = 20
STEPS_PER_EPOCH = 100
BATCH_SIZE = 96                 # pairs per step (A,B)
LR = 5e-4
WEIGHT_DECAY = 0.01
TAU = 0.07                      # temperature for NT-Xent
AREA_FILTER_RATIO = 1.10        # 1.1 × bbox area threshold
N_SQUARES = [3, 5, 7]           # n/48 square side ratios
SQUARE_BASE = "min_side"        # {"min_side","max_side"} for side = (n/48)*base
MIN_FG_PIXELS = 16              # minimal mask fg pixels to accept
SAVE_ROOT = Path("pe_adapter_save_high_1116_01")  # 每个 epoch 存这里/子目录

# Adapter config (Residual MLP / ClipAdapter style)
ADAPTER_HIDDEN_RATIO = 2        # hidden = c // ratio
ADAPTER_ALPHA = 0.2             # residual strength
DROPOUT = 0

# =========================
# IO / Model Utils
# =========================
def build_pe(device: torch.device):
    """Load PE model and its preprocess transform."""
    print("Loading PE model...")
    print("CLIP configs:", pe.CLIP.available_configs())
    model_pe = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True).to(device).eval()
    preprocess_pe = transforms.get_image_transform(model_pe.image_size)
    print("model_pe.image_size:", model_pe.image_size)
    return model_pe, preprocess_pe

def list_images(pdir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in pdir.iterdir() if p.suffix.lower() in exts])

def load_rgb(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")

def load_mask(p: Path, match_size_to: Tuple[int, int]) -> Image.Image:
    m = Image.open(p)
    if 'A' in m.getbands():
        m = m.getchannel('A')
    else:
        m = m.convert('L')
    if m.size != match_size_to:
        m = m.resize(match_size_to, resample=Image.NEAREST)
    return m

def tight_bbox_from_mask(msk: Image.Image, thr: int = 128) -> Optional[Tuple[int, int, int, int]]:
    """Return tight bbox (l,t,r,b) from binary mask; None if no fg."""
    arr = np.array(msk)
    fg = arr > thr
    if not fg.any():
        return None
    ys, xs = np.where(fg)
    l, r = int(xs.min()), int(xs.max() + 1)
    t, b = int(ys.min()), int(ys.max() + 1)
    return (l, t, r, b)

def crop_bbox(img: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    l, t, r, b = bbox
    return img.crop((l, t, r, b))

def sample_fg_point(msk: Image.Image, thr: int = 128) -> Optional[Tuple[int, int]]:
    arr = np.array(msk) > thr
    ys, xs = np.where(arr)
    if ys.size == 0:
        return None
    i = np.random.randint(0, ys.size)
    return int(xs[i]), int(ys[i])  # (cx,cy)

def safe_square_from_center(img_w: int, img_h: int, cx: int, cy: int, side: int) -> Tuple[int, int, int, int]:
    """Return a square bbox (l,t,r,b) clamped inside image."""
    half = side // 2
    l = max(0, cx - half)
    t = max(0, cy - half)
    r = min(img_w, l + side)
    b = min(img_h, t + side)
    # Adjust back if near boundary to keep side as much as possible
    l = max(0, r - side); t = max(0, b - side)
    return (l, t, r, b)

def area(b: Tuple[int, int, int, int]) -> int:
    l, t, r, bm = b
    return max(0, r - l) * max(0, bm - t)

# =========================
# Dataset Index
# =========================
# def build_index(root_rgb: Path, root_msk: Path):
#     obj_ids = sorted([d.name for d in root_rgb.iterdir() if d.is_dir()])
#     assert obj_ids, f"No object folders in {root_rgb}"
#     frames: List[List[Tuple[Path, Path]]] = []
#     for oid in obj_ids:
#         rgb_dir, msk_dir = root_rgb / oid, root_msk / oid
#         assert rgb_dir.is_dir() and msk_dir.is_dir(), f"Missing dirs for {oid}"
#         imgs, msks = list_images(rgb_dir), list_images(msk_dir)
#         assert len(imgs) == len(msks) and len(imgs) > 0, f"{oid}: rgb/mask mismatch or empty"
#         frames.append(list(zip(imgs, msks)))
#     return obj_ids, frames

def build_index(root_rgb: Path, root_msk: Path):
    """
    根据 rgb / mask 目录构建索引：
    - 以子文件夹名作为 object id（例如 000001, 000002, ...）
    - 对每个 object，按文件名 stem 对齐 rgb 和 mask
    - 只保留同时存在 rgb 和 mask 的样本
    - 对于完全没有匹配样本的 object，直接跳过
    """
    all_obj_ids = sorted([d.name for d in root_rgb.iterdir() if d.is_dir()])
    assert all_obj_ids, f"No object folders in {root_rgb}"

    obj_ids: List[str] = []
    frames: List[List[Tuple[Path, Path]]] = []

    for oid in all_obj_ids:
        rgb_dir, msk_dir = root_rgb / oid, root_msk / oid
        # 如果 mask 目录不存在，直接跳过这个 object
        if not (rgb_dir.is_dir() and msk_dir.is_dir()):
            continue

        imgs = list_images(rgb_dir)
        msks = list_images(msk_dir)

        # 用 stem 做键，过滤掉不匹配的
        rgb_stems = {p.stem: p for p in imgs}
        msk_stems = {p.stem: p for p in msks}

        common_stems = sorted(set(rgb_stems.keys()) & set(msk_stems.keys()))
        if not common_stems:
            # 这个 object 一个匹配样本都没有，就跳过
            continue

        # 只保留同时有 rgb 和 mask 的样本，并按 stem 排序，保证顺序稳定
        pairs = [(rgb_stems[s], msk_stems[s]) for s in common_stems]

        obj_ids.append(oid)
        frames.append(pairs)

    # 至少要有一个 object 有有效样本
    assert frames, f"No matching rgb/mask pairs found in {root_rgb} and {root_msk}"

    return obj_ids, frames

OBJ_IDS, OBJ_FRAMES = build_index(ROOT_RGB, ROOT_MSK)
NUM_OBJECTS = len(OBJ_IDS)
SEQ_LEN = len(OBJ_FRAMES[0])
print(f"[pe_adapter] Loaded {NUM_OBJECTS} objects, seq_len={SEQ_LEN} per object")

# =========================
# Model: PE (frozen) + Adapter (trainable)
# =========================
class PEClipAdapter(nn.Module):
    """
    Residual MLP adapter: y = x + alpha * MLP(LN(x))
    """
    def __init__(self, dim: int, hidden_ratio: int = 4, alpha: float = 0.5, dropout: float = 0.0):
        super().__init__()
        hidden = max(1, dim // max(1, hidden_ratio))
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.alpha = alpha
        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.ln(x)
        z = self.fc2(self.drop(self.act(self.fc1(z))))
        return x + self.alpha * z


def encode_pe(model, preprocess, imgs: List[Image.Image]) -> torch.Tensor:
    """Encode a list of PIL images -> (N, C) L2-normalized."""
    batch = torch.cat([preprocess(im).unsqueeze(0) for im in imgs], dim=0).to(device)
    feats = model.encode_image(batch)
    feats = F.normalize(feats, p=2, dim=-1)
    return feats

# =========================
# Visualization
# =========================
def visualize_pairs(A_imgs, B_imgs, obj_pairs, obj_id_list,
                    max_show=8, save_path=None, show=False,
                    tag="vis/pairs", step=0):
    k = min(len(A_imgs), len(B_imgs), len(obj_pairs), max_show)
    if k == 0:
        return

    ncols, nrows = 2, k
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows))
    if k == 1:
        import numpy as np
        axes = np.array([axes])

    for i in range(k):
        oA, oB = obj_pairs[i]
        pos = (oA == oB)
        color = "green" if pos else "red"
        label = "POSITIVE" if pos else "NEGATIVE"

        axA = axes[i, 0]
        axA.imshow(A_imgs[i]); axA.axis("off")
        axA.set_title(f"A | obj={obj_id_list[oA]} (id={oA})", fontsize=12)
        for spine in axA.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(3)

        axB = axes[i, 1]
        axB.imshow(B_imgs[i]); axB.axis("off")
        axB.set_title(f"B | obj={obj_id_list[oB]} (id={oB}) • {label}", fontsize=12, color=color)
        for spine in axB.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(3)

    fig.suptitle("Sampled A/B Crops (Left=A, Right=B)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path is not None:
        VIS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)


# =========================
# Crop Policy for A and B
# =========================
def build_B_choices(img: Image.Image, msk: Image.Image,
                    bbox: Tuple[int, int, int, int],
                    area_filter_ratio: float,
                    n_squares: List[int],
                    base_mode: str = "min_side") -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """
    Return candidate crops for B: list of ("bbox" or "square", bbox_ltrb).
    Filter out square types whose area > area_filter_ratio * bbox_area,
    and also remove larger ones once a threshold is exceeded.
    """
    W, H = img.size
    bbox_area = area(bbox)
    candidates: List[Tuple[str, Tuple[int, int, int, int]]] = []
    candidates.append(("bbox", bbox))  # 始终包含 bbox 紧贴

    base = min(W, H) if base_mode == "min_side" else max(W, H)
    # sorted_ns = sorted(n_squares)
    # allow_larger = True
    # for n in sorted_ns:
    #     side = max(1, int(round(base * (n / 48.0))))
    #     sq_area = side * side
    #     if allow_larger and (sq_area <= area_filter_ratio * bbox_area):
    #         candidates.append((f"square_{n}", (-1, -1, -1, -1)))  # 实际坐标采样时再定
    #     else:
    #         allow_larger = False
    return candidates

def sample_B_bbox_for_square(msk: Image.Image, img_size: Tuple[int, int], side: int) -> Optional[Tuple[int, int, int, int]]:
    """Sample a square whose center is inside mask foreground."""
    W, H = img_size
    pt = sample_fg_point(msk)
    if pt is None:
        return None
    cx, cy = pt
    return safe_square_from_center(W, H, cx, cy, side)

def make_A_B_crops(imgA: Image.Image, mskA: Image.Image,
                   imgB: Image.Image, mskB: Image.Image,
                   area_filter_ratio: float = AREA_FILTER_RATIO,
                   n_squares: List[int] = N_SQUARES,
                   base_mode: str = SQUARE_BASE) -> Optional[Tuple[Image.Image, Image.Image]]:
    """
    Build A (bbox-tight on A) and B (random type among filtered {3,5,7}/bbox) crops.
    Returns (cropA, cropB) or None if failed.
    """
    bboxA = tight_bbox_from_mask(mskA)
    bboxB = tight_bbox_from_mask(mskB)
    if bboxA is None or bboxB is None:
        return None
    # mask foreground sanity
    if (np.array(mskA) > 128).sum() < MIN_FG_PIXELS or (np.array(mskB) > 128).sum() < MIN_FG_PIXELS:
        return None

    # A = bbox-tight on A
    cropA = crop_bbox(imgA, bboxA)

    # B candidates filtered
    choices = build_B_choices(imgB, mskB, bboxB, area_filter_ratio, n_squares, base_mode)
    if len(choices) == 0:
        return None

    # Randomly choose one
    typ, _ = random.choice(choices)
    if typ == "bbox":
        cropB = crop_bbox(imgB, bboxB)
    elif typ.startswith("square_"):
        n = int(typ.split("_")[1])
        base = min(imgB.size) if base_mode == "min_side" else max(imgB.size)
        side = max(1, int(round(base * (n / 48.0))))
        bb2 = sample_B_bbox_for_square(mskB, imgB.size, side)
        if bb2 is None:
            return None
        cropB = crop_bbox(imgB, bb2)
    else:
        return None

    # Simple safety: avoid degenerate crops
    if cropA.size[0] < 2 or cropA.size[1] < 2 or cropB.size[0] < 2 or cropB.size[1] < 2:
        return None

    return cropA, cropB

# =========================
# Batch Sampler (pos:neg = 1:2)
# =========================
def sample_positive_pair():
    """A、B 同一 object。"""
    o = random.randrange(NUM_OBJECTS)
    imgA_p, mskA_p = random.choice(OBJ_FRAMES[o])
    imgB_p, mskB_p = random.choice(OBJ_FRAMES[o])
    return (o, (imgA_p, mskA_p)), (o, (imgB_p, mskB_p))

def sample_negative_pair():
    """A、B 不同 object。"""
    oA, oB = random.sample(range(NUM_OBJECTS), 2)
    imgA_p, mskA_p = random.choice(OBJ_FRAMES[oA])
    imgB_p, mskB_p = random.choice(OBJ_FRAMES[oB])
    return (oA, (imgA_p, mskA_p)), (oB, (imgB_p, mskB_p))

def build_batch_pairs(batch_size: int,
                      pos_ratio: float = 1.0 / 3.0,
                      max_tries_per_batch: int = 2000
                      ) -> Tuple[List[Image.Image], List[Image.Image], List[Tuple[int, int]]]:
    """
    返回 A_imgs, B_imgs, obj_pairs，其中 obj_pairs[i]=(oA, oB)。
    以 pos:neg = 1:2（≈33% 正样本）构建一个 batch。
    """
    target_pos = int(round(batch_size * pos_ratio))
    target_pos = max(1, min(batch_size - 1, target_pos))
    target_neg = batch_size - target_pos

    A_list, B_list, obj_pairs = [], [], []
    n_pos, n_neg = 0, 0
    tries = 0

    while (n_pos < target_pos or n_neg < target_neg) and tries < max_tries_per_batch:
        tries += 1

        if n_pos < target_pos and n_neg < target_neg:
            p_need_pos = (target_pos - n_pos) / max(1, (target_pos - n_pos) + (target_neg - n_neg))
            want_pos = (random.random() < p_need_pos)
        else:
            want_pos = (n_pos < target_pos)

        if want_pos:
            (oA, (imgA_p, mskA_p)), (oB, (imgB_p, mskB_p)) = sample_positive_pair()
        else:
            (oA, (imgA_p, mskA_p)), (oB, (imgB_p, mskB_p)) = sample_negative_pair()

        imgA = load_rgb(imgA_p); mskA = load_mask(mskA_p, imgA.size)
        imgB = load_rgb(imgB_p); mskB = load_mask(mskB_p, imgB.size)
        pair = make_A_B_crops(imgA, mskA, imgB, mskB)
        if pair is None:
            continue

        cropA, cropB = pair
        A_list.append(cropA); B_list.append(cropB); obj_pairs.append((oA, oB))
        if oA == oB:
            n_pos += 1
        else:
            n_neg += 1

    return A_list, B_list, obj_pairs

# =========================
# Loss: NT-Xent with multi-positives (A→B)
# =========================
def nt_xent_multi_pos(zQ: torch.Tensor, zK: torch.Tensor,
                      same_qk: torch.Tensor, tau: float = TAU) -> torch.Tensor:
    """
    zQ, zK: (N, C) L2-normalized
    same_qk: (N, N) bool, same_qk[i,j] True if Q_i and K_j are positives
    For each i, positives are K_j with same_qk[i,j]==True (j!=i)
    Negatives: all others in K
    """
    N = zQ.shape[0]
    sim = zQ @ zK.t()              # (N,N)
    eye = torch.eye(N, dtype=torch.bool, device=sim.device)
    pos_mask = same_qk & (~eye)    # 排除 (i,i) 自对
    sim_scaled = sim / tau

    losses = []
    for i in range(N):
        pos_idx = pos_mask[i].nonzero(as_tuple=False).squeeze(1)
        if pos_idx.numel() == 0:
            continue
        lse_pos = torch.logsumexp(sim_scaled[i, pos_idx], dim=0)
        lse_all = torch.logsumexp(sim_scaled[i, ~eye[i]], dim=0)
        losses.append(-(lse_pos - lse_all))
    if len(losses) == 0:
        return zQ.new_tensor(0.0)
    return torch.stack(losses, dim=0).mean()

# =========================
# Train
# =========================
def main():
    # 1) Load frozen PE model + preprocess
    pe_model, preprocess = build_pe(device)
    pe_model.eval()
    for p in pe_model.parameters():
        p.requires_grad_(False)

    # 2) Probe feature dim with one A-crop
    (o0, (img0_p, msk0_p)), _ = sample_positive_pair()
    img0 = load_rgb(img0_p); msk0 = load_mask(msk0_p, img0.size)
    bb0 = tight_bbox_from_mask(msk0)
    assert bb0 is not None, "No foreground in the very first sampled image/mask; please check data."
    crop0 = crop_bbox(img0, bb0)
    with torch.inference_mode():
        z0 = encode_pe(pe_model, preprocess, [crop0])
    C_DIM = z0.shape[1]
    print(f"[pe_adapter] PE feature dim = {C_DIM}")

    # 3) Build adapter (trainable)
    adapter = PEClipAdapter(
        dim=C_DIM,
        hidden_ratio=ADAPTER_HIDDEN_RATIO,
        alpha=ADAPTER_ALPHA,
        dropout=DROPOUT
    ).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 4) Logger
    run_name = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/pe_adapter_{run_name}")

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        adapter.train()
        running = 0.0
        for step in range(1, STEPS_PER_EPOCH + 1):
            # ---- Build a minibatch with pos:neg=1:2 ----
            A_imgs, B_imgs, obj_pairs = build_batch_pairs(BATCH_SIZE)

            # # 可视化
            # if global_step % VIS_EVERY_STEPS == 0:
            #     save_path = VIS_SAVE_DIR / f"pairs_e{epoch:02d}_s{step:04d}_g{global_step}.png"
            #     visualize_pairs(A_imgs, B_imgs, obj_pairs, OBJ_IDS,
            #         max_show=VIS_MAX_SHOW,
            #         save_path=save_path,
            #         show=False,
            #         tag="vis/pairs",
            #         step=global_step)

            if len(A_imgs) < 2:
                continue  # 至少两对

            with torch.no_grad():
                zA = encode_pe(pe_model, preprocess, A_imgs)   # (N,C)
                zB = encode_pe(pe_model, preprocess, B_imgs)   # (N,C)

            # ---- A-only adapter + L2 ----
            zA_hat = F.normalize(adapter(zA), p=2, dim=-1)    # A 侧适配
            zB_hat = F.normalize(adapter(zB), p=2, dim=-1)             # B 侧适配

            # ---- Build same_obj matrix (N,N) for A→B ----
            N = zA_hat.shape[0]
            a_ids = torch.tensor([ab[0] for ab in obj_pairs], device=device)
            b_ids = torch.tensor([ab[1] for ab in obj_pairs], device=device)
            same_obj = (a_ids.view(-1, 1) == b_ids.view(1, -1))  # (N,N) A_i vs B_j

            # ---- Loss: A→B ----
            loss = nt_xent_multi_pos(zA_hat, zB_hat, same_obj, tau=TAU)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            writer.add_scalar("loss/train", float(loss.item()), global_step)
            writer.add_scalar("counts/batch_size", N, global_step)
            # 统计正负
            if step % 50 == 0:
                pos_cnt = int((a_ids == b_ids).sum().item())
                writer.add_scalar("counts/positives_in_batch", pos_cnt, global_step)

            global_step += 1

            if step % 20 == 0:
                avg = running / 20
                print(f"[Epoch {epoch:02d} Step {step:04d}] loss={avg:.4f}  N={N}")
                running = 0.0

        # ---- save at end of epoch ----
        SAVE_ROOT.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "state_dict": adapter.state_dict(),
            "dim": C_DIM,
            "adapter_type": "pe_residual_mlp_A_only",
            "hidden_ratio": ADAPTER_HIDDEN_RATIO,
            "alpha": ADAPTER_ALPHA,
            "dropout": DROPOUT,
            "tau": TAU,
            "area_filter_ratio": AREA_FILTER_RATIO,
            "n_squares": N_SQUARES,
            "square_base": SQUARE_BASE,
            "seed": SEED,
        }
        out_dir = SAVE_ROOT / f"pe_adapter_epoch_{epoch:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "adapter.pt"
        torch.save(ckpt, out_path)
        print(f"[Epoch {epoch:02d}] saved checkpoint: {out_path}")

    print("Training finished.")
    writer.close()

if __name__ == "__main__":
    main()
