import os
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize as tv_resize, to_tensor as tv_to_tensor
from torch.utils.tensorboard import SummaryWriter
import time

# =========================
# Config
# =========================
SEED = 3
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = (
    torch.device("cuda:1") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
print(f"Using device: {device}")

# Paths: adapt to your layout
ROOT_RGB = Path("../SSD2/Dinov3_features/objects_all/rgb")
ROOT_MSK = Path("../SSD2/Dinov3_features/objects_all/mask")

# DINO & input sizes
PATCH_SIZE = 16
IMAGE_SIZE = 768
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

# Feature backbone (local DINOv3 hub)
REPO_DIR = '../SSD2/dinov3'
MODEL_NAME = 'dinov3_vitl16'
MODEL_TO_NUM_LAYERS = {
    'dinov3_vits16': 12,
    'dinov3_vits16plus': 12,
    'dinov3_vitb16': 12,
    'dinov3_vitl16': 24,
    'dinov3_vith16plus': 32,
    'dinov3_vit7b16': 40,
}

# Training hyperparams (you can tweak)
EPOCHS = 12
STEPS_PER_EPOCH = 1500
LR = 5e-4
WEIGHT_DECAY = 0.01
TAU = 0.07
POS_NEG_RATIO = 5.0
HARD_NEG_FRAC = 0.6
MASK_FG_THRESHOLD = 0.7
M_TARGET = 10

# =========================
# Adapter Config
# =========================
# 选这个来启用 3x3 token 自注意力适配器
ADAPTER_TYPE = "selfattn_token"   # "residual" | "weight" | "qk" | "selfattn" | "selfattn_token"

SELFAT_NUM_HEADS = 4
SELFAT_D_HEAD    = 16
SELFAT_DROPOUT   = 0.10
SELFAT_MODE      = "residual"   # 仅 SelfAttnChannelAdapter 用
SELFAT_ALPHA     = 0.2          # 仅 SelfAttnChannelAdapter 用

WEIGHT_ADAPT_REDUCTION = 4
WEIGHT_ADAPT_SCALAR = 5.0

# QK（保留以避免使用该分支时报未定义）
QK_NUM_HEADS = 4
QK_D_HEAD = 16
QK_DROPOUT = 0.1
QK_ALLOW_AMPLIFY = False
QK_BETA = 0.5

# Directory to save per-epoch adapters
ADAPTER_SAVE_DIR = Path(f"{ADAPTER_TYPE}_adapter_save")

# =========================
# IO & utils
# =========================

def list_images(pdir: Path) -> List[Path]:
    exts = {'.jpg','.jpeg','.png','.bmp'}
    return sorted([p for p in pdir.iterdir() if p.suffix.lower() in exts])


def load_left_groups(group_root_rgb: Path, group_root_msk: Path):
    """
    Return:
      groups: List[List[Tuple[Path, Path]]], len == num_objects
               each inner list is aligned (img_path, mask_path)
      num_objects, seq_len
    """
    object_ids = sorted([d.name for d in group_root_rgb.iterdir() if d.is_dir()])
    assert len(object_ids) > 0, f"No object folders found in {group_root_rgb}"

    groups: List[List[Tuple[Path, Path]]] = []
    seq_len_ref = None
    for oid in object_ids:
        rgb_dir = group_root_rgb / oid
        msk_dir = group_root_msk / oid
        assert rgb_dir.is_dir() and msk_dir.is_dir(), f"Missing rgb/mask dir for {oid}"
        rgb_list = list_images(rgb_dir)
        msk_list = list_images(msk_dir)
        assert len(rgb_list) == len(msk_list) and len(rgb_list) > 0, f"{oid}: rgb/mask mismatch or empty"
        pairs = list(zip(rgb_list, msk_list))
        groups.append(pairs)
        if seq_len_ref is None:
            seq_len_ref = len(pairs)
        else:
            assert len(pairs) == seq_len_ref, "All objects must have the same number of images"
    return groups, len(groups), seq_len_ref


def load_rgb(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")


def load_mask(p: Path, match_size_to: Tuple[int,int]) -> Image.Image:
    m = Image.open(p)
    if 'A' in m.getbands():
        m = m.getchannel('A')
    else:
        m = m.convert('L')
    if m.size != match_size_to:
        m = m.resize(match_size_to, resample=Image.NEAREST)
    return m


def resize_to_grid(img: Image.Image, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE) -> torch.Tensor:
    w, h = img.size
    h_p = image_size // patch_size
    w_p = int((w * image_size) / (h * patch_size))
    out_h, out_w = h_p * patch_size, w_p * patch_size
    arr = tv_to_tensor(tv_resize(img, [out_h, out_w]))  # [0,1]
    return arr

# =========================
# Sparsify (row then col) inside foreground mask
# =========================

def stride_filter_mask(
    sel_row: torch.Tensor,
    H1: int,
    W1: int,
    m: int,
) -> torch.Tensor:
    """Return keep_mask (K,) bool that sparsifies inside foreground.
    """
    device = sel_row.device
    K = H1 * W1
    sel2d = sel_row.view(H1, W1)
    n_fg = int(sel_row.sum().item())
    if n_fg == 0:
        return torch.zeros(K, dtype=torch.bool, device=device)
    stride = max(1, int(math.sqrt(n_fg / float(max(int(m), 1)))))

    keep_rows = torch.zeros((H1, W1), dtype=torch.bool, device=device)
    for r in range(H1):
        cols = torch.nonzero(sel2d[r], as_tuple=False).squeeze(1)
        k = int(cols.numel())
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
            c = int(cols[mid].item())
            keep_rows[r, c] = True
    keep_rows &= sel2d

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
    if (not torch.any(keep2d)) and torch.any(sel2d):
        ys, xs = torch.nonzero(sel2d, as_tuple=True)
        mid = ys.numel() // 2
        keep2d[ys[mid], xs[mid]] = True
    return keep2d.view(-1)

# =========================
# Backbone & Adapter
# =========================

print("Loading DINOv3 (local hub)...")
model_dino = torch.hub.load(REPO_DIR, model=MODEL_NAME, source='local',
                            weights=f"{REPO_DIR}/dinov3/checkpointer/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
model_dino.to(device).eval()

NUM_LAYERS = MODEL_TO_NUM_LAYERS[MODEL_NAME]

@torch.inference_mode()
def extract_patch_tokens(img: Image.Image, msk: Image.Image) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Return:
      feats: (K, C) float32 on device (L2 normalized per-channel later)
      sel:   (K,) bool foreground selection (after pooling)
      H1, W1: patch grid size
    """
    img_t = resize_to_grid(img).unsqueeze(0).to(device)
    msk_t = resize_to_grid(msk).unsqueeze(0).to(device)

    img_n = (img_t - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)

    out = model_dino.get_intermediate_layers(
        img_n,
        n=[NUM_LAYERS - 1],
        reshape=True,
        norm=True,
        return_class_token=True,
    )
    feat, _cls = out[0]  # (B,C,H1,W1), (B,C)
    feat = feat[0].detach()  # (C,H1,W1)

    # foreground selection on patch grid
    m_avg = F.avg_pool2d(msk_t, kernel_size=PATCH_SIZE, stride=PATCH_SIZE).squeeze(0).squeeze(0)  # (H1,W1)
    H1, W1 = m_avg.shape
    sel = (m_avg > MASK_FG_THRESHOLD).view(-1)  # (K,)

    feat = F.normalize(feat, p=2, dim=0)  # channel-wise
    C = feat.shape[0]
    feats = feat.permute(1,2,0).reshape(-1, C)  # (K,C)
    return feats, sel, H1, W1















class ResidualAdapter(nn.Module):
    def __init__(self, dim: int, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=False),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)

    def forward(self, x):
        return x + self.alpha * self.net(x)

class WeightAdapter(nn.Module):
    def __init__(self, c_in: int, reduction: int = 4, scalar: float = 10.0):
        super().__init__()
        hidden = max(1, c_in // reduction)
        self.scalar = float(scalar)
        self.fc = nn.Sequential(
            nn.Linear(c_in, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c_in, bias=False),
            nn.ReLU(inplace=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.scalar * x
        g = torch.sigmoid(self.fc(z))
        y = g * z
        return y

class QKChannelAdapter(nn.Module):
    def __init__(self, c_in: int, num_heads: int = 4, d_head: int = 16,
                 dropout: float = 0.0, allow_amplify: bool = False, beta: float = 0.5):
        super().__init__()
        self.c, self.h, self.d = c_in, num_heads, d_head
        self.scale = 1.0 / math.sqrt(d_head)
        self.Wk = nn.Linear(c_in, c_in * num_heads * d_head, bias=False)
        self.Wv = nn.Linear(c_in, c_in * num_heads * d_head, bias=False)
        self.q = nn.Parameter(torch.randn(num_heads, d_head))
        self.Wo = nn.Linear(num_heads * c_in, c_in, bias=True)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(c_in)
        self.allow_amplify = allow_amplify
        self.beta = beta
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
        nn.init.normal_(self.q, std=0.02)

    def forward(self, x):             # x: (N, C)
        x_in = self.ln(x)
        N, C, H, D = x_in.shape[0], self.c, self.h, self.d
        K = self.Wk(x_in).view(N, H, C, D)              # (N,H,C,D)
        q = self.q.view(1, H, 1, D)                     # (1,H,1,D)
        attn_logits = (K * q).sum(-1) * self.scale      # (N,H,C)
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.drop(attn)
        gate_logits = self.Wo(attn.reshape(N, H * C))   # (N,C)
        if self.allow_amplify:
            g = 1.0 + self.beta * torch.sigmoid(gate_logits)
        else:
            g = torch.sigmoid(gate_logits)
        return g * x

class SelfAttnChannelAdapter(nn.Module):
    def __init__(self, c_in: int, num_heads: int = 4, d_head: int = 16,
                 dropout: float = 0.1, mode: str = "gate", alpha: float = 0.2):
        super().__init__()
        self.c, self.h, self.d = c_in, num_heads, d_head
        self.scale = 1.0 / math.sqrt(d_head)
        d_model = num_heads * d_head
        self.ln = nn.LayerNorm(c_in)
        self.Wq = nn.Linear(c_in, d_model, bias=False)
        self.Wk = nn.Linear(c_in, d_model, bias=False)
        self.Wv = nn.Linear(c_in, d_model, bias=False)
        self.proj = nn.Linear(d_model, c_in, bias=True)
        self.drop = nn.Dropout(dropout)
        assert mode in ("gate", "residual")
        self.mode = mode
        self.alpha = alpha
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)

    def forward(self, x):   # x: (N, C)
        N, C = x.shape
        x_n = self.ln(x)
        q = self.Wq(x_n).view(N, C, self.h, self.d).transpose(1, 2)  # (N,H,C,D)
        k = self.Wk(x_n).view(N, C, self.h, self.d).transpose(1, 2)  # (N,H,C,D)
        v = self.Wv(x_n).view(N, C, self.h, self.d).transpose(1, 2)  # (N,H,C,D)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale     # (N,H,C,C)
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)                # (N,H,C,D)
        out = out.transpose(1, 2).contiguous().view(N, C, self.h * self.d)
        out = self.proj(out)                       # (N, C)
        if self.mode == "gate":
            g = torch.sigmoid(out)
            y = g * x
        else:
            y = x + self.alpha * out
        return y

# ------------ 新增：3x3 token 自注意力 + token mean -------------
class SelfAttnTokenAdapter(nn.Module):
    """
    输入: x_tokens (N, 9, C) —— 每个样本的3x3邻域9个token
    在 token 维做 MHSA，最后对 token 维求平均，输出 (N, C)
    """
    def __init__(self, c_in: int, num_heads: int = 4, d_head: int = 16, dropout: float = 0.1):
        super().__init__()
        self.h, self.d = num_heads, d_head
        d_model = num_heads * d_head
        self.scale = 1.0 / math.sqrt(d_head)
        self.ln   = nn.LayerNorm(c_in)
        self.Wq   = nn.Linear(c_in, d_model, bias=False)
        self.Wk   = nn.Linear(c_in, d_model, bias=False)
        self.Wv   = nn.Linear(c_in, d_model, bias=False)
        self.proj = nn.Linear(d_model, c_in, bias=True)
        self.drop = nn.Dropout(dropout)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)

    def forward(self, x):                 # x: (N, 9, C)
        N, T, C = x.shape
        x_n = self.ln(x)                  # (N,9,C)
        q = self.Wq(x_n).view(N, T, self.h, self.d).transpose(1, 2)  # (N,H,9,D)
        k = self.Wk(x_n).view(N, T, self.h, self.d).transpose(1, 2)  # (N,H,9,D)
        v = self.Wv(x_n).view(N, T, self.h, self.d).transpose(1, 2)  # (N,H,9,D)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale     # (N,H,9,9)
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out  = torch.matmul(attn, v)                                  # (N,H,9,D)
        out  = out.transpose(1, 2).contiguous().view(N, T, self.h * self.d)  # (N,9,H*D)
        out  = self.proj(out)                                         # (N,9,C)
        y = out.mean(dim=1)                                           # (N,C)
        return y

# ---- 新增：把 K×C 和 H,W 还原后取 3x3 邻域tokens ----
OFFS_3x3 = [(-1,-1),(-1,0),(-1,1),
            ( 0,-1),( 0,0),( 0,1),
            ( 1,-1),( 1,0),( 1,1)]

def build_idx_and_tokens_3x3(
    feats_flat: torch.Tensor,   # (K, C)
    H: int, W: int,
    keep_mask: torch.Tensor,    # (K,) bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = feats_flat.device
    C = feats_flat.shape[1]
    idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)   # (N,)
    if idx.numel() == 0:
        return torch.empty(0, 2, dtype=torch.long, device=device), torch.empty(0, 9, C, device=device)

    r = (idx // W)
    c = (idx %  W)
    centers_rc = torch.stack([r, c], dim=1)                     # (N,2)

    grid = feats_flat.view(H, W, C)

    tokens_list = []
    for dr, dc in OFFS_3x3:
        rr = (r + dr).clamp(0, H-1)
        cc = (c + dc).clamp(0, W-1)
        tok = grid[rr, cc]                                      # (N, C)
        tokens_list.append(tok)
    tokens_9 = torch.stack(tokens_list, dim=1)                  # (N, 9, C)
    return centers_rc, tokens_9
# --------------------------------------------------------------

# =========================
# Data indexing & sampling
# =========================

groups, NUM_OBJECTS, SEQ_LEN = load_left_groups(ROOT_RGB, ROOT_MSK)
print(f"Loaded {NUM_OBJECTS} objects, seq_len per object = {SEQ_LEN}")

# Build a flat index per object for convenience
obj_frames: List[List[Tuple[Path, Path]]] = groups

# =========================
# Loss: MIL-NCE (bag-positive)
# =========================

def mil_nce_bag_loss(
    anchors: torch.Tensor,   # (Na,C), L2-normalized
    pos_bag: torch.Tensor,   # (Np,C), L2-normalized
    neg_pool: torch.Tensor,  # (Nn,C), L2-normalized
    tau: float = TAU,
    neg_ratio: float = POS_NEG_RATIO,
    hard_frac: float = HARD_NEG_FRAC,
) -> torch.Tensor:
    assert anchors.ndim == 2 and pos_bag.ndim == 2
    Na, C = anchors.shape
    Np = pos_bag.shape[0]
    if Na == 0 or Np == 0:
        return anchors.new_tensor(0.0)

    sim_ap = anchors @ pos_bag.t()               # (Na,Np)
    if neg_pool is None or neg_pool.numel() == 0:
        logits_pos = sim_ap / tau
        lse_pos = torch.logsumexp(logits_pos, dim=1)
        lse_all = lse_pos
        loss = -(lse_pos - lse_all).mean()
        return loss

    sim_an = anchors @ neg_pool.t()              # (Na,Nn)
    Nn = neg_pool.shape[0]

    Bn = int(math.ceil(neg_ratio * Np))
    Bn = min(Bn, Nn) if Nn > 0 else 0
    if Bn == 0:
        logits_pos = sim_ap / tau
        lse_pos = torch.logsumexp(logits_pos, dim=1)
        loss = -(lse_pos - lse_pos).mean()
        return loss

    k_hard = int(math.floor(hard_frac * Bn))
    k_rand = Bn - k_hard

    losses = []
    for i in range(Na):
        logits_pos_i = sim_ap[i] / tau
        if Nn > 0:
            sim_an_i = sim_an[i]
            if k_hard > 0:
                hard_idx = torch.topk(sim_an_i, k=min(k_hard, Nn), largest=True, sorted=False).indices
            else:
                hard_idx = torch.empty(0, dtype=torch.long, device=anchors.device)
            if k_rand > 0 and (Nn - hard_idx.numel()) > 0:
                mask = torch.ones(Nn, dtype=torch.bool, device=anchors.device)
                mask[hard_idx] = False
                rest_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                if rest_idx.numel() <= k_rand:
                    rand_idx = rest_idx
                else:
                    perm = torch.randperm(rest_idx.numel(), device=anchors.device)[:k_rand]
                    rand_idx = rest_idx[perm]
                neg_idx = torch.cat([hard_idx, rand_idx], dim=0)
            else:
                neg_idx = hard_idx
            logits_neg_i = (sim_an_i[neg_idx] / tau) if neg_idx.numel() > 0 else None
        else:
            logits_neg_i = None

        if logits_neg_i is None or logits_neg_i.numel() == 0:
            lse_pos_i = torch.logsumexp(logits_pos_i, dim=0)
            lse_all_i = lse_pos_i
        else:
            lse_pos_i = torch.logsumexp(logits_pos_i, dim=0)
            lse_neg_i = torch.logsumexp(logits_neg_i, dim=0)
            m = torch.maximum(lse_pos_i, lse_neg_i)
            lse_all_i = m + torch.log(torch.exp(lse_pos_i - m) + torch.exp(lse_neg_i - m))
        losses.append(-(lse_pos_i - lse_all_i))

    loss = torch.stack(losses).mean()
    return loss

# =========================
# Build Adapter after probing C
# =========================
probe_img, probe_msk = obj_frames[0][0]
feats_probe, sel_probe, H1_probe, W1_probe = extract_patch_tokens(load_rgb(probe_img), load_mask(probe_msk, load_rgb(probe_img).size))
C_DIM = feats_probe.shape[1]
print(f"Feature dim (C) = {C_DIM}")

if ADAPTER_TYPE == "residual":
    adapter = ResidualAdapter(C_DIM, alpha=0.2).to(device)
elif ADAPTER_TYPE == "weight":
    adapter = WeightAdapter(C_DIM, reduction=WEIGHT_ADAPT_REDUCTION,
                            scalar=WEIGHT_ADAPT_SCALAR).to(device)
elif ADAPTER_TYPE == "qk":
    adapter = QKChannelAdapter(
        C_DIM, num_heads=QK_NUM_HEADS, d_head=QK_D_HEAD,
        dropout=QK_DROPOUT, allow_amplify=QK_ALLOW_AMPLIFY, beta=QK_BETA
    ).to(device)
elif ADAPTER_TYPE == "selfattn":
    adapter = SelfAttnChannelAdapter(
        C_DIM, num_heads=SELFAT_NUM_HEADS, d_head=SELFAT_D_HEAD,
        dropout=SELFAT_DROPOUT, mode=SELFAT_MODE, alpha=SELFAT_ALPHA
    ).to(device)
elif ADAPTER_TYPE == "selfattn_token":
    adapter = SelfAttnTokenAdapter(
        C_DIM, num_heads=SELFAT_NUM_HEADS, d_head=SELFAT_D_HEAD, dropout=SELFAT_DROPOUT
    ).to(device)
else:
    raise ValueError(f"Unknown ADAPTER_TYPE: {ADAPTER_TYPE}")

opt = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# TensorBoard writer
run_name = time.strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=f"runs/adapter_{ADAPTER_TYPE}_{run_name}")

# =========================
# Training loop (pure training — no validation/ckpt here)
# =========================

def sample_positive_pair() -> Tuple[int, Tuple[Path,Path], Tuple[Path,Path]]:
    o = random.randrange(NUM_OBJECTS)
    frames = obj_frames[o]
    i, j = random.sample(range(len(frames)), 2)
    return o, frames[i], frames[j]

def sample_negative_images(exclude_obj: int, total_imgs: int = 20) -> List[Tuple[Path,Path]]:
    """从除 exclude_obj 之外的对象中选 10–19 个对象，总计采样 total_imgs=20 张负图。"""
    other_ids = [x for x in range(NUM_OBJECTS) if x != exclude_obj]
    if not other_ids:
        return []

    # 选 10–19 个对象（上限不能超过剩余对象数）
    k_obj = random.randint(10, min(19, len(other_ids)))
    random.shuffle(other_ids)
    sel_objs = other_ids[:k_obj]

    # 目标总图数 = 20，尽量均匀分配到每个对象；先各取 1，再均匀补齐
    out: List[Tuple[Path,Path]] = []

    # 第一次轮询：每个对象先取 1 张
    for oid in sel_objs:
        frames = obj_frames[oid]
        t = random.randrange(len(frames))
        out.append(frames[t])

    # 还差多少
    remaining = max(0, total_imgs - len(out))

    # 第二次轮询：均匀补齐到 total_imgs
    # 每个对象再取 ceil(remaining/k_obj) 张，注意越界保护
    per_extra = math.ceil(remaining / k_obj) if k_obj > 0 else 0
    if remaining > 0 and per_extra > 0:
        for oid in sel_objs:
            if remaining <= 0:
                break
            frames = obj_frames[oid]
            k_take = min(per_extra, remaining, len(frames))  # 防止越界
            # 随机不重复采样 k_take 张（若 frames 很少，用 random.sample 的 k 不能超过 len）
            idxs = random.sample(range(len(frames)), k=k_take)
            out.extend([frames[i] for i in idxs])
            remaining -= k_take

    # 若仍不足（极端小数据情况），从其它对象补齐
    while len(out) < total_imgs and len(other_ids) > 0:
        oid = random.choice(other_ids)
        frames = obj_frames[oid]
        t = random.randrange(len(frames))
        out.append(frames[t])

    # 截断到 total_imgs
    return out[:total_imgs]

def select_foreground_sparse(feats: torch.Tensor, sel: torch.Tensor, H1: int, W1: int) -> torch.Tensor:
    keep = stride_filter_mask(sel, H1, W1, m=M_TARGET)
    idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
    return feats[idx]

global_step = 0
for epoch in range(1, EPOCHS + 1):
    adapter.train()
    running = 0.0

    for step in range(1, STEPS_PER_EPOCH + 1):
        # ---- sample positive pair ----
        o_pos, (img_a_p, msk_a_p), (img_b_p, msk_b_p) = sample_positive_pair()
        img_a = load_rgb(img_a_p)
        img_b = load_rgb(img_b_p)
        msk_a = load_mask(msk_a_p, img_a.size)
        msk_b = load_mask(msk_b_p, img_b.size)

        # ---- extract tokens ----
        with torch.inference_mode():
            fa, sela, H1a, W1a = extract_patch_tokens(img_a, msk_a)  # (Ka_all,C)
            fb, selb, H1b, W1b = extract_patch_tokens(img_b, msk_b)

        # ---- branch: adapter inputs ----
        if ADAPTER_TYPE == "selfattn_token":
            # 3x3 邻域组包
            keep_a = stride_filter_mask(sela, H1a, W1a, m=M_TARGET)
            keep_b = stride_filter_mask(selb, H1b, W1b, m=M_TARGET)
            _, Za_9 = build_idx_and_tokens_3x3(fa, H1a, W1a, keep_a)  # (Na,9,C)
            _, Zp_9 = build_idx_and_tokens_3x3(fb, H1b, W1b, keep_b)  # (Np,9,C)
            if Za_9.numel() == 0 or Zp_9.numel() == 0:
                continue
        else:
            # 其它 adapter 走原来的稀疏前景采样
            Za = select_foreground_sparse(fa, sela, H1a, W1a)  # (Ka,C)
            Zp = select_foreground_sparse(fb, selb, H1b, W1b)  # (Kb,C)
            if Za.numel() == 0 or Zp.numel() == 0:
                continue

        # ---- negatives ----
        neg_imgs = sample_negative_images(exclude_obj=o_pos, total_imgs=10)
        if ADAPTER_TYPE == "selfattn_token":
            Zneg_9_list = []
            with torch.inference_mode():
                for (img_n_p, msk_n_p) in neg_imgs:
                    img_n = load_rgb(img_n_p)
                    msk_n = load_mask(msk_n_p, img_n.size)
                    fn, seln, H1n, W1n = extract_patch_tokens(img_n, msk_n)
                    keep_n = stride_filter_mask(seln, H1n, W1n, m=M_TARGET)
                    _, Zn_9 = build_idx_and_tokens_3x3(fn, H1n, W1n, keep_n)
                    if Zn_9.numel() > 0:
                        Zneg_9_list.append(Zn_9)
            Zneg_9 = torch.cat(Zneg_9_list, dim=0) if len(Zneg_9_list) > 0 else torch.empty(0, 9, C_DIM, device=device)
        else:
            Zneg_list = []
            with torch.inference_mode():
                for (img_n_p, msk_n_p) in neg_imgs:
                    img_n = load_rgb(img_n_p)
                    msk_n = load_mask(msk_n_p, img_n.size)
                    fn, seln, H1n, W1n = extract_patch_tokens(img_n, msk_n)
                    Zn = select_foreground_sparse(fn, seln, H1n, W1n)
                    if Zn.numel() > 0:
                        Zneg_list.append(Zn)
            Zneg = torch.cat(Zneg_list, dim=0) if len(Zneg_list) > 0 else torch.empty(0, C_DIM, device=device)

        # ---- adapter forward + L2 ----
        if ADAPTER_TYPE == "selfattn_token":
            Za_after = F.normalize(adapter(Za_9), p=2, dim=1)     # (Na,C)
            Zp_after = F.normalize(adapter(Zp_9), p=2, dim=1)     # (Np,C)
            Zneg_after = F.normalize(adapter(Zneg_9), p=2, dim=1) if Zneg_9.numel() > 0 else torch.empty(0, C_DIM, device=device)
        else:
            Za_after = F.normalize(adapter(Za), p=2, dim=1)
            Zp_after = F.normalize(adapter(Zp), p=2, dim=1)
            Zneg_after = F.normalize(adapter(Zneg), p=2, dim=1) if Zneg.numel() > 0 else Zneg

        # ---- loss ----
        loss = mil_nce_bag_loss(Za_after, Zp_after, Zneg_after, tau=TAU, neg_ratio=POS_NEG_RATIO, hard_frac=HARD_NEG_FRAC)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # TensorBoard logging
        writer.add_scalar("loss/train", float(loss.item()), global_step)
        writer.add_scalar("counts/anchors", Za_after.shape[0], global_step)
        writer.add_scalar("counts/pos_bag", Zp_after.shape[0], global_step)
        writer.add_scalar("counts/neg_pool", Zneg_after.shape[0] if Zneg_after.numel()>0 else 0, global_step)

        global_step += 1
        running += float(loss.item())
        if step % 20 == 0:
            print(f"[Epoch {epoch:02d} Step {step:03d}] loss={running/20:.4f} | Ka={Za_after.shape[0]} Kp={Zp_after.shape[0]} Kn={Zneg_after.shape[0] if Zneg_after.numel()>0 else 0}")
            running = 0.0

        # —— 每个 epoch 结束保存一次 —— 放在 for step 循环的最后
        if step == STEPS_PER_EPOCH:
            ADAPTER_SAVE_DIR.mkdir(parents=True, exist_ok=True)

            ckpt = {
                "epoch": epoch,
                "state_dict": adapter.state_dict(),
                "adapter_type": ADAPTER_TYPE,
                "dim": C_DIM,
                "tau": TAU,
                "m_target": M_TARGET,
                "mask_fg_threshold": MASK_FG_THRESHOLD,
                "model_name": MODEL_NAME,
                "image_size": IMAGE_SIZE,
                "patch_size": PATCH_SIZE,
                "mean": IMAGENET_MEAN.tolist(),
                "std": IMAGENET_STD.tolist(),
            }

            if ADAPTER_TYPE == "residual":
                ckpt.update({
                    "alpha": float(getattr(adapter, "alpha", 0.2)),
                })
            elif ADAPTER_TYPE == "qk":
                ckpt.update({
                    "qk_num_heads": getattr(adapter, "h", QK_NUM_HEADS),
                    "qk_d_head":    getattr(adapter, "d", QK_D_HEAD),
                    "qk_dropout":   float(getattr(adapter, "drop").p) if hasattr(adapter, "drop") else QK_DROPOUT,
                    "qk_allow_amplify": bool(getattr(adapter, "allow_amplify", QK_ALLOW_AMPLIFY)),
                    "qk_beta":      float(getattr(adapter, "beta", QK_BETA)),
                })
            elif ADAPTER_TYPE == "selfattn":
                ckpt.update({
                    "selfattn_num_heads": SELFAT_NUM_HEADS,
                    "selfattn_d_head":    SELFAT_D_HEAD,
                    "selfattn_dropout":   SELFAT_DROPOUT,
                    "selfattn_mode":      SELFAT_MODE,
                    "selfattn_alpha":     SELFAT_ALPHA,
                })
            elif ADAPTER_TYPE == "selfattn_token":
                ckpt.update({
                    "selfattn_token_num_heads": SELFAT_NUM_HEADS,
                    "selfattn_token_d_head":    SELFAT_D_HEAD,
                    "selfattn_token_dropout":   SELFAT_DROPOUT,
                    "token_window":             3,      # 3x3
                    "token_count":              9,
                })

            ckpt_path = ADAPTER_SAVE_DIR / f"adapter_epoch{epoch:03d}_{ADAPTER_TYPE}_{run_name}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"[Epoch {epoch:02d}] checkpoint saved to: {ckpt_path}")

print("Training finished.")
writer.close()
