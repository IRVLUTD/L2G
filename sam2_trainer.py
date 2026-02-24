

import os
import glob
import random
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sam2.utils.transforms import SAM2Transforms  # NEW

from sam2.build_sam import build_sam2   

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-7):
    """Combination of BCE and Dice losses."""
    bce = F.binary_cross_entropy_with_logits(logits, target.float())
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    dice = 1 - (2 * inter + eps) / union
    return bce + dice.mean()

def bce_dice_iou_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-7):
    """
    Hybrid loss = BCE + Dice + IoU
    - logits: (B,1,H,W) raw scores
    - target: (B,1,H,W) {0,1}
    """
    # --- BCE ---
    bce = F.binary_cross_entropy_with_logits(logits, target.float())

    # --- probs / sums ---
    probs = torch.sigmoid(logits)
    dims = (2, 3)
    inter = (probs * target).sum(dim=dims)
    sum_p = probs.sum(dim=dims)
    sum_t = target.sum(dim=dims)

    # --- Dice loss ---
    dice = 1.0 - (2.0 * inter + eps) / (sum_p + sum_t + eps)   # per-sample

    # --- IoU (Jaccard) loss ---
    iou = 1.0 - (inter + eps) / (sum_p + sum_t - inter + eps)  # per-sample

    # 组合：BCE + mean(Dice) + mean(IoU)
    return bce + 0.5*dice.mean() + iou.mean()

def pick_random_interior_point(mask: np.ndarray) -> Tuple[int, int]:
    """Pick a random positive pixel not on the mask boundary."""
    assert mask.ndim == 2
    H, W = mask.shape
    import cv2

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode((mask * 255).astype(np.uint8), kernel, iterations=1) > 0
    ys, xs = np.where(eroded)
    if len(xs) == 0:  # fallback
        ys, xs = np.where(mask > 0)
    idx = np.random.randint(len(xs))
    return int(xs[idx]), int(ys[idx])  # (x, y)

# -----------------------------
# Dataset
# -----------------------------

@dataclass
class TrainConfig:
    data_root: str
    object_id: str
    model_cfg: str
    checkpoint: str
    save_dir: str = "./ckpts_full_mask_token"
    epochs: int = 100
    batch_size: int = 1
    lr: float = 1e-2
    num_workers: int = 2
    seed: int = 42
    device: str = "cuda:0"

class OnePointObjectDataset(Dataset):
    """Dataset of template images and masks for one object."""

    def __init__(self, data_root: str, object_id: str):
        self.rgb_dir = os.path.join(data_root, "HOCAP_create_0117", "rgb", object_id)
        self.mask_dir = os.path.join(data_root, "HOCAP_create_0117", "mask", object_id)
        self.samples = []
        # for ext in [".jpg", ".png", ".jpeg"]:
        #     for path in sorted(glob.glob(os.path.join(self.rgb_dir, f"*{ext}"))):
        #         name = os.path.splitext(os.path.basename(path))[0]
        #         mask_path = os.path.join(self.mask_dir, f"{name}.png")
        #         if os.path.exists(mask_path):
        #             self.samples.append((path, mask_path))

        for ext in [".jpg", ".png", ".jpeg"]:
            for path in sorted(glob.glob(os.path.join(self.rgb_dir, f"*{ext}"))):
                name = os.path.splitext(os.path.basename(path))[0]
                mask_path = os.path.join(self.mask_dir, f"{name}.png")
                if not os.path.exists(mask_path):
                    continue
                m = np.array(Image.open(mask_path).convert("L"))
                m = (m > 0).astype(np.uint8)
                if m.sum() == 0:
                    continue
                self.samples.append((path, mask_path))
        if not self.samples:
            raise RuntimeError(f"No samples found for {object_id}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 0).astype(np.uint8)
        x, y = pick_random_interior_point(mask)
        return {"image": img, "mask": mask, "point_xy": np.array([x, y], np.float32)}


# -----------------------------
# Training
# -----------------------------

def main(cfg: TrainConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ✅ Build model using local config + checkpoint
    model = build_sam2(cfg.model_cfg, cfg.checkpoint, device=device)
    model.eval()
    model.to(device)

    print("model.image_size:",model.image_size)
    transforms = SAM2Transforms(
        resolution=model.image_size,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
    )

    # ✅ Check that full_mask_token exists (you already modified MaskDecoder)
    if not hasattr(model.sam_mask_decoder, "full_mask_tokens"):
        raise RuntimeError("MaskDecoder missing full_mask_tokens. Please add it first.")

    # ✅ Freeze all params except the token
    for p in model.parameters():
        p.requires_grad = False
    model.sam_mask_decoder.full_mask_tokens.requires_grad = True

    # Data
    dataset = OnePointObjectDataset(cfg.data_root, cfg.object_id)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    optimizer = torch.optim.AdamW([model.sam_mask_decoder.full_mask_tokens], lr=cfg.lr)
    scaler = torch.amp.GradScaler(device.type)

    best_loss = float("inf")
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        step_ = 0
        for batch in loader:
            image = batch["image"][0]
            mask = batch["mask"][0]
            point = batch["point_xy"][0]
       

            # Forward image encoder

            # A) Prepocess image（predictor.set_image()）
            # ✅ ensure numpy HWC uint8 / HW uint8
            if isinstance(image, torch.Tensor): image = image.cpu().numpy()
            if isinstance(mask, torch.Tensor):  mask  = mask.cpu().numpy()
            if image.ndim == 3 and image.shape[0] in (3,4) and image.shape[-1] not in (3,4):
                image = np.transpose(image, (1,2,0))   # CHW -> HWC
            image = image.astype(np.uint8, copy=False)
            mask  = (mask > 0).astype(np.uint8, copy=False)

            if mask.sum() == 0
                # print(f"[skip] empty mask sample at step {step_}")
                continue



            input_image = transforms(image)       # (3,H',W'), 已做 resize/normalize/pad
            input_image = input_image.unsqueeze(0).to(device)  # (1,3,H',W')

            backbone_out = model.forward_image(input_image)

            fmaps = backbone_out["backbone_fpn"][-model.num_feature_levels:]
            feat  = fmaps[-1]
            high_res_features = None
            if model.use_high_res_features_in_sam:
                high_res_features = [backbone_out["backbone_fpn"][0], backbone_out["backbone_fpn"][1]]

    
            # H, W = mask.shape

            # orig_h, orig_w = H, W

            # point_coords = torch.tensor([[[float(point[0]), float(point[1])]]], device=device)
            # point_coords = transforms.transform_coords(point_coords, normalize=True, orig_hw=(orig_h, orig_w))
            # point_coords[..., 0].clamp_(0, model.image_size - 1)
            # point_coords[..., 1].clamp_(0, model.image_size - 1)
            # point_labels = torch.tensor([[1]], dtype=torch.int32, device=device)
       

   
            H, W = mask.shape
            orig_h, orig_w = H, W

     
            r = random.random()
            k = 1 if r < 0.2 else (2 if r < 0.4 else (3 if r < 0.6 else 4))
            #k = 1 if r < 0.5 else (2 if r < 0.75 else 3)

            pts = [pick_random_interior_point(mask) for _ in range(k)]      # [(x,y), ...]
            point_coords = torch.tensor([[[float(x), float(y)] for (x, y) in pts]], device=device)  # [1,k,2]
            point_coords = transforms.transform_coords(point_coords, normalize=True, orig_hw=(orig_h, orig_w))
            #print("before point_coords:",point_coords)
            point_coords[..., 0].clamp_(0, model.image_size - 1)
            point_coords[..., 1].clamp_(0, model.image_size - 1)
            #print("after point_coords:",point_coords.shape)
            point_labels = torch.ones((1, k), dtype=torch.int32, device=device)  

      

            # ---- prompt & decoder ----
            sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
                points=(point_coords, point_labels), boxes=None, masks=None,
            )
            image_pe = model.sam_prompt_encoder.get_dense_pe()

            masks_pred, iou_pred, _, _ = model.sam_mask_decoder(
                image_embeddings=feat,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )
            logits = masks_pred[:, :1]  # [B,1,h,w]
            if (logits.shape[-2] != orig_h) or (logits.shape[-1] != orig_w):
                logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

            # ---- GT & loss ----
            gt = torch.from_numpy(mask[None, None]).float().to(device)
            main_loss = bce_dice_iou_loss(logits, gt)

            # ---- Orthogonality regularizer for K trainable tokens ----
            # Put this immediately after main_loss and before backward().
            lambda_ortho = 1e-3  # you can expose this via argparse, e.g. --lambda_ortho
            if hasattr(model.sam_mask_decoder, "full_mask_tokens"):
                T = model.sam_mask_decoder.full_mask_tokens  # [1,K,C] or [K,C] (depending on your init)
                if T.ndim == 3:  # [1,K,C] -> [K,C]
                    T = T.squeeze(0)
                K = T.shape[0]
                if K > 1:  # ortho makes sense only when K>=2
                    TT = F.normalize(T, p=2, dim=-1)              # row-wise L2 normalize
                    I = torch.eye(K, device=TT.device, dtype=TT.dtype)
                    ortho = ((TT @ TT.t()) - I).pow(2).mean()     # ||TT TT^T - I||_F^2 / (K^2)
                    #loss = main_loss + lambda_ortho * ortho
                    loss = main_loss
                else:
                    loss = main_loss
            else:
                # fallback for single-token version (no orthogonality needed)
                loss = main_loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            step_ = step_ +1
            if step_ % 10 == 0:
                print(f"[Epoch {epoch+1}/{cfg.epochs}] step={step_} loss={loss.item():.4f}")


        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch+1}/{cfg.epochs}] Loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "object_id": cfg.object_id,
                    "tokens": model.sam_mask_decoder.full_mask_tokens.detach().cpu(),  # [1,5,C]
                },
                os.path.join(cfg.save_dir, f"full_mask_tokens_{cfg.object_id}.pt"),
            )
            print(f"✅ Saved tokens for {cfg.object_id} (best loss {best_loss:.4f})")
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "object_id": cfg.object_id,
                    "tokens": model.sam_mask_decoder.full_mask_tokens.detach().cpu(),
                },
                os.path.join(cfg.save_dir, f"full_mask_tokens_{cfg.object_id}_{epoch+1}.pt"),
            )
            print(f"💾 Saved tokens at epoch {epoch+1} → _{epoch+1}.pt")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--object_id", required=True, type=str)
    ap.add_argument("--model_cfg", required=True, type=str)
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--save_dir", default="./ckpts_full_mask_token_HOCAP_0117", type=str)
    ap.add_argument("--epochs", default=10, type=int)
    ap.add_argument("--lr", default=1e-2, type=float)
    ap.add_argument("--batch_size", default=1, type=int)
    ap.add_argument("--device", default="cuda:0", type=str, help="device, e.g. 'cuda:0' or 'cuda:1'")  # ✅ NEW
    args = ap.parse_args()
    cfg = TrainConfig(
        data_root=args.data_root,
        object_id=args.object_id,
        model_cfg=args.model_cfg,
        checkpoint=args.checkpoint,
        save_dir=args.save_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,   # ✅ NEW
    )
    main(cfg)
