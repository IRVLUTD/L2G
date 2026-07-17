# utils/build.py

import torch
import numpy as np
from PIL import Image

from perception_models.core.vision_encoder import pe, transforms
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .data import load_rgb, load_mask, stride_filter_mask


def build_pe(device):
    """Load PE model and its preprocess transform."""
    print("Loading PE model...")
    print("CLIP configs:", pe.CLIP.available_configs())
    model_pe = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True).to(device).eval()
    preprocess_pe = transforms.get_image_transform(model_pe.image_size)
    print("model_pe.image_size:",model_pe.image_size)
    return model_pe, preprocess_pe


def resize_transform(img: Image.Image, image_size=None, patch_size=None):
    """Resize image to (IMAGE_SIZE, proportional width) snapped to patch multiples; return tensor in [0,1]."""
    w, h = img.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(img, (h_patches * patch_size, w_patches * patch_size)))


def tight_bbox_from_mask(mask_bin: np.ndarray):
    """Return PIL-style bbox (left, top, right, bottom) from a binary HxW mask. None if empty."""
    assert mask_bin.ndim == 2
    ys, xs = np.where(mask_bin > 0)
    if ys.size == 0:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1  # exclusive for PIL
    return (x0, y0, x1, y1)


def tight_bbox_from_mask_scale(
    mask_bin: np.ndarray,
    crop_box,
):
    """
    Return PIL-style bbox (left, top, right, bottom) from a binary HxW mask.
    If crop_box is provided, interpret mask_bin as a crop of the original image,
    and return the bbox in the *original* image coordinates.

    Args:
        mask_bin: np.ndarray, shape (Hc, Wc), binary mask in the crop coordinate.
        crop_box: (y0, x0, y1, x1) of the crop in the original image. If None,
                  assume mask_bin is already in the original image coordinates.

    Returns:
        (left, top, right, bottom) in original image coordinates, or None if empty.
    """
    assert mask_bin.ndim == 2
    ys, xs = np.where(mask_bin > 0)
    if ys.size == 0:
        return None

    # bbox in *crop* coordinates
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1  # exclusive for PIL

    if crop_box is not None:
        cy0, cx0, cy1, cx1 = crop_box  # order (y0, x0, y1, x1)
        
        x0 += cx0
        x1 += cx0
        y0 += cy0
        y1 += cy0
    return (x0, y0, x1, y1)


def pe_feature_from_pil(img_pil: Image.Image, model_pe, preprocess_pe, pe_adapter, device) -> torch.Tensor:
    """Encode a PIL image with PE and return an L2-normalized feature vector (D,)."""
    with torch.inference_mode():
        t = preprocess_pe(img_pil.convert("RGB")).unsqueeze(0).to(device)
        f = model_pe.encode_image(t)
        f = f / f.norm(dim=-1, keepdim=True)

        f =  F.normalize(pe_adapter(f), p=2, dim=-1) 
    return f.squeeze(0)


def dino_cls_from_pil(crop, model, mean, std, cfg: dict, device):
    PATCH_SIZE = cfg["data"]["patch_size"]
    IMAGE_SIZE = cfg["data"]["image_size"]
    MODEL_NAME = cfg["dino"]["model_name"]
    MODEL_TO_NUM_LAYERS = cfg["dino"]["model_to_num_layers"]

    crop_norm = (resize_transform(crop, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE).to(device) - mean) / std
    
    with torch.inference_mode():
        out = model.get_intermediate_layers(
            crop_norm,
            #n=range(MODEL_TO_NUM_LAYERS[MODEL_NAME]),
            n= [MODEL_TO_NUM_LAYERS[MODEL_NAME] - 1],
            reshape=True,
            norm=True,
            return_class_token=True,
        )
        feat_, cls_ = out[0]
        cls_ = cls_.detach()
        cls_ = F.normalize(cls_, p=2, dim=1)
    cls_ = cls_.to(device).squeeze(0) 
    return cls_


def precompute_left_pe_features(left_images, left_masks, model_pe, preprocess_pe, pe_adapter, device):
    """
    For each left object: crop by its mask bbox and compute PE feature.
    Returns a list of length B; items are torch.Tensor(D,) or None if mask empty.
    """
    feats = []
    for im, m in zip(left_images, left_masks):
        m_np = np.array(m)
        if m_np.ndim == 3:                       # RGBA or multi-channel mask
            if m_np.shape[2] == 4:
                m_np = m_np[..., 3]              # prefer alpha channel if present
            else:
                m_np = m_np[..., -1]
        m_bin = (m_np > 0).astype(np.uint8)
        bbox = tight_bbox_from_mask(m_bin)
        if bbox is None:
            feats.append(None)
            continue
        crop = im.crop(bbox).convert("RGB")
        feats.append(pe_feature_from_pil(crop, model_pe, preprocess_pe, pe_adapter, device))
    return feats



def build_left_batch_all(groups, SEQ_LEN, device, IMAGE_SIZE, PATCH_SIZE, mean, std, model, cfg, MASK_FG_THRESHOLD, model_pe, preprocess_pe, pe_adapter):
    MODEL_NAME = cfg["dino"]["model_name"]
    MODEL_TO_NUM_LAYERS = cfg["dino"]["model_to_num_layers"]


    left_pairs_all = {}
    left_images_all = {}
    left_masks_all = {}
    sel_left_all = {}
    left_vecs_all = {}
    left_cls_all = {}
    H1_all = {}
    W1_all = {}
    left_pe_feats_all = {}
    left_fg_mean_all = {}
    for t in range(SEQ_LEN):
        left_pairs_t = [(img_p, msk_p) for (img_p, msk_p) in [g[t] for g in groups]]
        left_images_t = [load_rgb(p) for p, _ in left_pairs_t]
        left_masks_t  = [load_mask(m, match_size_to=left_images_t[i].size) for i, (_, m) in enumerate(left_pairs_t)]

        left_pe_feats_t = precompute_left_pe_features(left_images_t, left_masks_t, model_pe, preprocess_pe, pe_adapter, device)
        #left_pe_feats_t = None  # only use DINO 

        left_imgs_resized  = torch.stack([resize_transform(im, image_size=768, patch_size=PATCH_SIZE)  for im in left_images_t], dim=0).to(device)
        left_masks_resized = torch.stack([resize_transform(msk, image_size=768, patch_size=PATCH_SIZE) for msk in left_masks_t],  dim=0).to(device)

        B, _, Hl, Wl = left_imgs_resized.shape
        left_norm = (left_imgs_resized - mean) / std
        mask_q = F.avg_pool2d(left_masks_resized, kernel_size=PATCH_SIZE, stride=PATCH_SIZE).squeeze(1)  # (B,H1,W1)
        H1, W1 = mask_q.shape[-2], mask_q.shape[-1]
        sel_left_t = (mask_q > MASK_FG_THRESHOLD).view(B, -1)  # (B,K)
        with torch.inference_mode():
            out_l = model.get_intermediate_layers(
                left_norm,
                #n=range(MODEL_TO_NUM_LAYERS[MODEL_NAME]),
                n= [MODEL_TO_NUM_LAYERS[MODEL_NAME] - 1],
                reshape=True,
                norm=True,
                return_class_token=True,
            )
            feat_l, cls_l = out_l[0]
            #feat_l = out_l[-1].detach()  # (B, C, H1, W1)
            feat_l = feat_l.detach()  # (B, C, H1, W1)
            print(f"##### Template {t+1} ##### The shape of feat_l:",feat_l.shape)
            #print("##############The shape of cls_l:",cls_l.shape)
        feat_l = F.normalize(feat_l, p=2, dim=1)
        C = feat_l.shape[1]
        left_vecs_t = feat_l.permute(0,2,3,1).reshape(B, -1, C)  # (B,K,C)
        left_cls_t = precompute_left_dino_features(left_images_t, left_masks_t, model, mean, std, cfg, device)
        with torch.inference_mode():
            left_vecs_t = F.normalize(left_vecs_t, p=2, dim=-1)
        fg_means_t = []
        for b in range(B):
            sel_b = sel_left_t[b]          # (K,) bool
            vecs_b = left_vecs_t[b]        # (K,C)
            if sel_b.any():
                mean_b = vecs_b[sel_b].mean(dim=0)          # (C,)
                mean_b = F.normalize(mean_b, p=2, dim=0) 
            else:
                mean_b = torch.zeros(C, device=device, dtype=vecs_b.dtype)
            fg_means_t.append(mean_b)
        left_fg_mean_all[t] = fg_means_t

        left_pairs_all[t] = left_pairs_t
        left_images_all[t] = left_images_t
        left_masks_all[t] = left_masks_t
        sel_left_all[t] = sel_left_t
        left_vecs_all[t] = left_vecs_t
        left_cls_all[t] = left_cls_t
        H1_all[t] = H1
        W1_all[t] = W1
        left_pe_feats_all[t] = left_pe_feats_t

        del left_imgs_resized, left_masks_resized, left_norm, mask_q
        del feat_l, cls_l, out_l

    return left_pairs_all, left_images_all, left_masks_all, sel_left_all, left_vecs_all, H1_all, W1_all, left_cls_all, left_pe_feats_all, left_fg_mean_all


def precompute_valid_mask_all(sel_left_all, H1_all, W1_all, M_TARGET, SEQ_LEN):
    """
    Precompute the geometric stride-filter mask ('valid_mask') per (template, object) once.

    stride_filter_mask only depends on the template-side mask selection (sel_left_all)
    and the template grid size (H1_all/W1_all) plus M_TARGET, all of which are fixed
    before the query-image loop starts and never change per right/query image. Calling
    it once here instead of per query image avoids redoing an O(H1+W1) Python loop
    (with per-element GPU syncs) for every template/object on every single query image.
    """
    valid_mask_all = {}
    for t in range(SEQ_LEN):
        sel_left_t = sel_left_all[t]      # (B, K_t)
        H1, W1 = H1_all[t], W1_all[t]
        B = sel_left_t.shape[0]
        valid_mask = torch.zeros_like(sel_left_t, dtype=torch.bool)
        for b in range(B):
            keep_mask_b, _stride_b = stride_filter_mask(
                sel_row=sel_left_t[b],
                H1=H1, W1=W1, m=M_TARGET,
                keep_edges=False,
                Filter=True,
            )
            valid_mask[b] = keep_mask_b
        valid_mask_all[t] = valid_mask
    return valid_mask_all


def precompute_left_dino_features(left_images, left_masks, model, mean, std, cfg, device):
    MODEL_NAME = cfg["dino"]["model_name"]
    MODEL_TO_NUM_LAYERS = cfg["dino"]["model_to_num_layers"]
    PATCH_SIZE = cfg["data"]["patch_size"]
    IMAGE_SIZE = cfg["data"]["image_size"]

    cls_l_all = []
    for im, m in zip(left_images, left_masks):
        m_np = np.array(m)
        if m_np.ndim == 3:                       # RGBA or multi-channel mask
            if m_np.shape[2] == 4:
                m_np = m_np[..., 3]              # prefer alpha channel if present
            else:
                m_np = m_np[..., -1]
        m_bin = (m_np > 0).astype(np.uint8)
        bbox = tight_bbox_from_mask(m_bin)
        crop = im.crop(bbox)
        left_norm = (resize_transform(crop, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE).to(device) - mean) / std
        with torch.inference_mode():
            out_l = model.get_intermediate_layers(
                left_norm,
                #n=range(MODEL_TO_NUM_LAYERS[MODEL_NAME]),
                n= [MODEL_TO_NUM_LAYERS[MODEL_NAME] - 1],
                reshape=True,
                norm=True,
                return_class_token=True,
            )
            feat_l, cls_l = out_l[0]
            cls_l = cls_l.detach()
            cls_l = F.normalize(cls_l, p=2, dim=1)
            
        cls_l = cls_l.to(device).squeeze(0) 
        cls_l_all.append(cls_l)
    return cls_l_all
