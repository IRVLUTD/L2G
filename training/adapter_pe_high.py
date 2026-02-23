#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a residual MLP adapter on frozen pe_model features with pre-cropped RGB images.

Changes vs original:
- Use DataLoader + IterableDataset to build batches (A,B) with pos:neg ≈ 1:2.
- Each RGB image is already a tight bbox crop, so we no longer do mask-based cropping.
- We still keep the same loss (NT-Xent with multi-positives) and adapter structure.

Author: you + ChatGPT
"""

import os
import math
import random
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image

# Your PE model env
from perception_models.core.vision_encoder import pe, transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

# === Visualization config (kept but not actively used) ===

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIS_EVERY_STEPS = 10           # how many steps between visualizations
VIS_MAX_SHOW = 8               # max pairs to visualize each time
VIS_SAVE_DIR = Path("debug_vis_1116")

# =========================
# Config
# =========================
SEED = 3
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("[pe_adapter] Using device:", device)

# TODO: adjust to your new dataset root if needed
ROOT_RGB = Path("../SSD2/High_datasets/RoboTools_create_bbox/rgb")
ROOT_MSK = Path("../SSD2/High_datasets/RoboTools_create_bbox/mask")  # kept for index-building, but masks are not used in training

# Train hyper-params
EPOCHS = 20
STEPS_PER_EPOCH = 100
BATCH_SIZE = 96                 # pairs per step (A,B)
LR = 5e-4
WEIGHT_DECAY = 0.01
TAU = 0.07                      # temperature for NT-Xent
AREA_FILTER_RATIO = 1.10        # kept for record (not used now)
N_SQUARES = [3, 5, 7]           # kept for record (not used now)
SQUARE_BASE = "min_side"        # kept for record (not used now)
MIN_FG_PIXELS = 16              # kept for record (not used now)
SAVE_ROOT = Path("pe_adapter_save_0110_RoboTools")

# Adapter config (Residual MLP / ClipAdapter style)
ADAPTER_HIDDEN_RATIO = 2        # hidden = c // ratio
ADAPTER_ALPHA = 0.2             # residual strength
DROPOUT = 0

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

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
    """List image files with valid extensions sorted by name."""
    return sorted([p for p in pdir.iterdir() if p.suffix.lower() in VALID_EXTS])


def load_rgb(p: Path) -> Image.Image:
    """Load an RGB image."""
    return Image.open(p).convert("RGB")


# ========== robust index builder (RGB + mask matched by stem, but training only uses RGB) ==========
def build_index(root_rgb: Path, root_msk: Path):
    """
    Build index of objects and frames:
    - object id is each sub-directory name under root_rgb (and/or root_msk).
    - for each object, we align rgb and mask images by filename stem.
    - only keep samples that have both rgb and mask. Masks are not used later, but this
      ensures your dataset is consistent and avoids accidental junk files.
    """
    all_obj_ids = sorted([d.name for d in root_rgb.iterdir() if d.is_dir()])
    assert all_obj_ids, f"No object folders in {root_rgb}"

    obj_ids: List[str] = []
    frames: List[List[Tuple[Path, Path]]] = []

    for oid in all_obj_ids:
        rgb_dir, msk_dir = root_rgb / oid, root_msk / oid
        if not (rgb_dir.is_dir() and msk_dir.is_dir()):
            # Skip objects missing either rgb or mask folder
            continue

        imgs = list_images(rgb_dir)
        msks = list_images(msk_dir)

        # Align by filename stem
        rgb_stems: Dict[str, Path] = {p.stem: p for p in imgs}
        msk_stems: Dict[str, Path] = {p.stem: p for p in msks}

        common_stems = sorted(set(rgb_stems.keys()) & set(msk_stems.keys()))
        if not common_stems:
            # No valid pairs for this object, skip it
            continue

        pairs = [(rgb_stems[s], msk_stems[s]) for s in common_stems]

        obj_ids.append(oid)
        frames.append(pairs)

    assert frames, f"No matching rgb/mask pairs found in {root_rgb} and {root_msk}"

    return obj_ids, frames


OBJ_IDS, OBJ_FRAMES = build_index(ROOT_RGB, ROOT_MSK)
NUM_OBJECTS = len(OBJ_IDS)
SEQ_LEN = len(OBJ_FRAMES[0])
print(f"[pe_adapter] Loaded {NUM_OBJECTS} objects, seq_len={SEQ_LEN} per object (using RGB as tight crops)")

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
        # initialization
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
# Visualization (kept for future use)
# =========================
def visualize_pairs(A_imgs, B_imgs, obj_pairs, obj_id_list,
                    max_show=8, save_path=None, show=False,
                    tag="vis/pairs", step=0):
    """Visualize A/B pairs for debugging."""
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
        axA.imshow(A_imgs[i])
        axA.axis("off")
        axA.set_title(f"A | obj={obj_id_list[oA]} (id={oA})", fontsize=12)
        for spine in axA.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

        axB = axes[i, 1]
        axB.imshow(B_imgs[i])
        axB.axis("off")
        axB.set_title(f"B | obj={obj_id_list[oB]} (id={oB}) • {label}", fontsize=12, color=color)
        for spine in axB.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    fig.suptitle("Sampled A/B Crops (Left=A, Right=B)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path is not None:
        VIS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)


# =========================
# IterableDataset for pairs (A,B) with pos:neg ≈ 1:2
# =========================
class PairIterableDataset(IterableDataset):
    """
    Iterable dataset that yields (imgA, imgB, oA, oB) pairs.

    - Objects are indexed by 0..NUM_OBJECTS-1.
    - For positive pairs, oA == oB, and two frames are randomly chosen from that object.
    - For negative pairs, oA != oB, and frames are sampled from two different objects.
    - RGB images are already tight bbox crops, so we just load them and use directly.
    """
    def __init__(self, obj_frames, pos_ratio: float = 1.0 / 3.0, seed: int = 3):
        super().__init__()
        self.obj_frames = obj_frames
        self.num_objects = len(obj_frames)
        self.pos_ratio = float(pos_ratio)
        self.base_seed = int(seed)

    def _sample_positive_pair(self, rng: random.Random):
        """Sample (o, frameA, frameB) from same object."""
        o = rng.randrange(self.num_objects)
        imgA_p, _ = rng.choice(self.obj_frames[o])
        imgB_p, _ = rng.choice(self.obj_frames[o])
        return o, imgA_p, imgB_p

    def _sample_negative_pair(self, rng: random.Random):
        """Sample (oA, frameA) and (oB, frameB) from different objects."""
        oA, oB = rng.sample(range(self.num_objects), 2)
        imgA_p, _ = rng.choice(self.obj_frames[oA])
        imgB_p, _ = rng.choice(self.obj_frames[oB])
        return oA, imgA_p, oB, imgB_p

    def __iter__(self):
        """
        Each worker gets its own RNG seeded by base_seed + worker_id,
        then generates an infinite stream of pairs.
        """
        worker_info = get_worker_info()
        if worker_info is None:
            rng = random.Random(self.base_seed)
        else:
            rng = random.Random(self.base_seed + worker_info.id)

        while True:
            # Decide whether to generate a positive or negative pair
            if rng.random() < self.pos_ratio:
                # Positive pair: same object
                o, imgA_p, imgB_p = self._sample_positive_pair(rng)
                imgA = load_rgb(imgA_p)
                imgB = load_rgb(imgB_p)
                yield imgA, imgB, o, o
            else:
                # Negative pair: different objects
                oA, imgA_p, oB, imgB_p = self._sample_negative_pair(rng)
                imgA = load_rgb(imgA_p)
                imgB = load_rgb(imgB_p)
                yield imgA, imgB, oA, oB


def pair_collate_fn(batch):
    """
    Collate function for DataLoader.

    Input:
        batch: list of tuples (imgA, imgB, oA, oB), length = batch_size

    Output:
        A_imgs: list of PIL Images
        B_imgs: list of PIL Images
        a_ids:  1D LongTensor of object indices for A
        b_ids:  1D LongTensor of object indices for B
    """
    A_imgs = [x[0] for x in batch]
    B_imgs = [x[1] for x in batch]
    a_ids = torch.tensor([x[2] for x in batch], dtype=torch.long)
    b_ids = torch.tensor([x[3] for x in batch], dtype=torch.long)
    return A_imgs, B_imgs, a_ids, b_ids


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
    pos_mask = same_qk & (~eye)    # exclude self-pairs
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

    # 2) Probe feature dim with one random RGB crop
    #    (we pick one positive pair and use its A image)
    (o0, (img0_p, _)), _ = (
        (0, OBJ_FRAMES[0][0]),
        (0, OBJ_FRAMES[0][0]),
    ) if NUM_OBJECTS > 0 else None

    img0 = load_rgb(img0_p)
    with torch.inference_mode():
        z0 = encode_pe(pe_model, preprocess, [img0])
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

    # 5) Build iterable dataset + dataloader
    dataset = PairIterableDataset(OBJ_FRAMES, pos_ratio=1.0 / 3.0, seed=SEED)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,          # adjust according to your CPU
        pin_memory=True,
        collate_fn=pair_collate_fn
    )

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        adapter.train()
        running = 0.0

        # We iterate over the (potentially infinite) loader, but
        # manually stop after STEPS_PER_EPOCH steps per epoch.
        for step, batch in enumerate(loader, start=1):
            if step > STEPS_PER_EPOCH:
                break

            A_imgs, B_imgs, a_ids, b_ids = batch
            # Move ids to device
            a_ids = a_ids.to(device)
            b_ids = b_ids.to(device)

            if len(A_imgs) < 2:
                # At least 2 samples are required for contrastive loss
                continue

            # ---- Encode with frozen PE (no grad) ----
            with torch.no_grad():
                zA = encode_pe(pe_model, preprocess, A_imgs)   # (N,C)
                zB = encode_pe(pe_model, preprocess, B_imgs)   # (N,C)

            # ---- Adapter on A and B, + L2 normalize ----
            zA_hat = F.normalize(adapter(zA), p=2, dim=-1)
            zB_hat = F.normalize(adapter(zB), p=2, dim=-1)

            # ---- Build same_obj matrix (N,N) for A→B ----
            N = zA_hat.shape[0]
            same_obj = (a_ids.view(-1, 1) == b_ids.view(1, -1))  # (N,N)

            # ---- Loss: A→B ----
            loss = nt_xent_multi_pos(zA_hat, zB_hat, same_obj, tau=TAU)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            writer.add_scalar("loss/train", float(loss.item()), global_step)
            writer.add_scalar("counts/batch_size", N, global_step)

            # For logging: count positives in this batch
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
            "adapter_type": "pe_residual_mlp_A_B",  # both A and B use adapter here
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
