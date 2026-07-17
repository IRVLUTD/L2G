import os, re, glob
import urllib
from pathlib import Path
import cv2
import math
import json


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
from PIL import Image

from perception_models.core.vision_encoder import pe, transforms

import numpy as np
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
from PIL import Image


import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from tqdm import tqdm
import argparse

from utils.data import count_object_tokens
from utils.sam_utils import load_object_tokens, postprocess_mask_preserve_format
from utils.data import load_rgb, load_mask, stride_filter_mask, load_left_groups
from utils.build import tight_bbox_from_mask, tight_bbox_from_mask_scale
from utils.build import precompute_left_pe_features, pe_feature_from_pil, build_pe, build_left_batch_all, precompute_valid_mask_all
from utils.visualization import save_mask_image_with_bbox
from utils.image import crop_around_point
from adapter import load_pe_adapter
from Candidate import process_one_right_image_all


np.random.seed(3)
parser = argparse.ArgumentParser()
parser.add_argument("--scene-start", type=int, default=1)
parser.add_argument("--scene-end",   type=int, default=1)
parser.add_argument("--config", type=str, default="RoboTools.yaml")
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="Device to use (e.g., cuda:0, cuda:1, cpu)"
)
args = parser.parse_args()

# -------------------------
# Device selection
# -------------------------
if "cuda" in args.device and not torch.cuda.is_available():
    print("CUDA not available, fallback to CPU.")
    device = torch.device("cpu")
else:
    device = torch.device(args.device)

if device.type == "cuda":
    print(f"using device: {device}")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

cfg = load_config(os.path.join("L2G_configs",args.config))

# ---- DATA ----
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
DATASET_NAME = cfg["data"]["datasets"]
PATCH_SIZE = cfg["data"]["patch_size"]
IMAGE_SIZE = cfg["data"]["image_size"]
IMAGENET_MEAN = tuple(cfg["data"]["imagenet_mean"])
IMAGENET_STD  = tuple(cfg["data"]["imagenet_std"])
MASK_FG_THRESHOLD = cfg["data"]["mask_fg_threshold"]
M_TARGET = cfg["data"]["m_target"]
N_OBJECTS = cfg["data"]["n_objects"]
TEMPLATE_IMAGES_PATH = cfg["data"]["template_images_path"]
QUERY_PATH = cfg["data"]["query_path"]

# ---- SAM ----
checkpoint = cfg["sam"]["checkpoint"]
model_cfg  = cfg["sam"]["model_cfg"]
USE_AUGMENTED_PREDICTOR = cfg["sam"]["use_augmented_predictor"]
TOKEN_DIR = cfg["sam"]["token_dir"]

# ---- DINO ----
REPO_DIR = cfg["dino"]["repo_dir"]
MODEL_NAME = cfg["dino"]["model_name"]
MODEL_TO_NUM_LAYERS = cfg["dino"]["model_to_num_layers"]
MODEL_PATH = cfg["dino"]["checkpoint"]

# ---- ADAPTER ----
USE_PE_ADAPTER = cfg["adapter"]["use_pe_adapter"]
PE_ADAPTER_CKPT = cfg["adapter"]["pe_adapter_ckpt"]
USE_WHICH_FEATURE = cfg["adapter"]["use_which_feature"]

# ---- POST PROCESSING ----
USE_LOCAL_VIEW = cfg["post_processing"]["use_local_view"]
USE_FPS_FILTER = cfg["post_processing"]["use_fps_filter"]
SAVE_IMAGES = cfg["post_processing"]["save_images"]
LOCAL_VIEW_BATCH_SIZE = cfg["post_processing"].get("local_view_batch_size", 8)

model  = torch.hub.load(REPO_DIR, model=MODEL_NAME, source='local', weights=MODEL_PATH)
# model = torch.hub.load(
#     "facebookresearch/dinov2",
#     "dinov2_vitl14"
# )
model.to(device)
model.eval()  
model_sam2 = build_sam2(model_cfg, checkpoint, device=device, n_objects=N_OBJECTS)
model_sam2.eval()

if USE_AUGMENTED_PREDICTOR:
    print("Injecting object tokens (Augmented SAM)...")
    num_object_tokens = count_object_tokens(TOKEN_DIR)
    print(f"Detected {num_object_tokens} object_tokens.")
    load_object_tokens(
        model_sam2,
        TOKEN_DIR,
        device,
    )
else:
    print("Using Base SAM")

# ---- build predictor ----
predictor = SAM2ImagePredictor(model_sam2)

# Loading adapter
if USE_PE_ADAPTER:
    print("Loading adapter")
    pe_adapter = load_pe_adapter(PE_ADAPTER_CKPT, device)
else:
    pe_adapter = torch.nn.Identity()
    print(f"No adapter is using")
print(f"Using {USE_WHICH_FEATURE}")

# Loading template images
left_groups, NUM_OBJECTS, SEQ_LEN = load_left_groups(
    Path(os.path.join(TEMPLATE_IMAGES_PATH, "rgb")),
    Path(os.path.join(TEMPLATE_IMAGES_PATH, "mask"))
)


def resize_transform(img: Image.Image, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
    """Resize image to (IMAGE_SIZE, proportional width) snapped to patch multiples; return tensor in [0,1]."""
    w, h = img.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(img, (h_patches * patch_size, w_patches * patch_size)))


model_pe, preprocess_pe = build_pe(device)
model_pe.eval()

mean = torch.tensor(IMAGENET_MEAN, device=device).view(1,3,1,1)
std  = torch.tensor(IMAGENET_STD , device=device).view(1,3,1,1)

(left_pairs_all, left_images_all, left_masks_all,
sel_left_all, left_vecs_all, H1_all, W1_all, left_cls_all, left_pe_feats_all,left_fg_mean_all) = build_left_batch_all(
    groups=left_groups,
    SEQ_LEN=SEQ_LEN,
    device=device,
    IMAGE_SIZE=IMAGE_SIZE,
    PATCH_SIZE=PATCH_SIZE,
    mean=mean,
    std=std,
    model=model,
    cfg=cfg,
    MASK_FG_THRESHOLD=MASK_FG_THRESHOLD,
    model_pe=model_pe,
    preprocess_pe=preprocess_pe,
    pe_adapter=pe_adapter,
)

# valid_mask only depends on the template side (sel_left_all/H1_all/W1_all/M_TARGET), which
# never changes across query images or scenes, so it's computed once here instead of being
# redone inside process_one_right_image_all for every single query image.
valid_mask_all = precompute_valid_mask_all(sel_left_all, H1_all, W1_all, M_TARGET, SEQ_LEN)

for scene_id in range(args.scene_start, args.scene_end + 1):
    print(f"\n===== Processing SCENE : {scene_id} =====")
     # === NEW: scene ===
    scene_results = []
    OUT_ROOT = Path(f"Output/{DATASET_NAME}/{scene_id:06d}/pred_results_USE_PE_ADAPTER={USE_PE_ADAPTER}_USE_AUGMENTED_SAM={USE_AUGMENTED_PREDICTOR}")
    #image_right_dir = Path(f"../SSD2/RoboTools/test/{scene_id:06d}/rgb")
    image_right_dir = Path(QUERY_PATH) / f"{scene_id:06d}" / "rgb"
    right_image_list = sorted([p for p in image_right_dir.iterdir() if p.suffix.lower() in IMG_EXT])
    assert len(right_image_list) > 0, f"No right images found in {image_right_dir}"
    for right_path in right_image_list:
        print(f"\n===== Processing Query image: {right_path.name} =====")
        # ---- Load RIGHT image and compute per-axis scale back to original pixels ----
        image_right = load_rgb(right_path)
        right_img_np = np.array(image_right)
        #print("$$$$$$$$$$$$$$$$$$$$$$$$$$right_img_np.size:",right_img_np.shape)
        if not USE_LOCAL_VIEW:
            predictor.set_image(right_img_np)
        image_right_name = right_path.name
        right_resized = resize_transform(image_right).to(device)  # (3,Hr,Wr)
        Hr, Wr = right_resized.shape[-2], right_resized.shape[-1]
        print(f"hr={Hr}, Wr={Wr}")
        scale_right_x = image_right.width  / Wr
        scale_right_y = image_right.height / Hr
        right_norm = (right_resized.unsqueeze(0) - mean) / std
        print("right_norm.shape:",right_norm.shape)
        # ---- RIGHT features ----
        with torch.inference_mode():
            # with torch.autocast(device_type=device.type, dtype=torch.float32 if device.type=="cuda" else torch.float16):
            out_r = model.get_intermediate_layers(
                right_norm,
                n=range(MODEL_TO_NUM_LAYERS[MODEL_NAME]),
                reshape=True,
                norm=True
            )
            feat_r = out_r[-1].squeeze(0).detach()  # (C, H2, W2)
        feat_r = F.normalize(feat_r, p=2, dim=0)        # (C, H2, W2)
        C, H2, W2 = feat_r.shape
        all_points_per_object = {b: [] for b in range(NUM_OBJECTS)}
        all_scores_per_object = {b: [] for b in range(NUM_OBJECTS)}

        all_points_per_object, all_scores_per_object = process_one_right_image_all(
            image_right=image_right,
            image_right_name=image_right_name,
            scale_right_x=scale_right_x,
            scale_right_y=scale_right_y,
            W2=W2,
            feat_r=feat_r,
            predictor=predictor,
            device=device,
            PATCH_SIZE=PATCH_SIZE,
            IMAGE_SIZE=IMAGE_SIZE,
            mean=mean,
            std=std,
            left_pairs_all=left_pairs_all,
            sel_left_all=sel_left_all,
            left_vecs_all=left_vecs_all,
            H1_all=H1_all,
            W1_all=W1_all,
            M_TARGET=M_TARGET,
            S1_THR=0,
            model=model,
            model_pe=model_pe,
            preprocess_pe=preprocess_pe,
            pe_adapter=pe_adapter,
            left_cls_all = left_cls_all,
            left_pe_feats_all=left_pe_feats_all,
            left_fg_mean_all = left_fg_mean_all,
            cfg=cfg,
            valid_mask_all=valid_mask_all,
            image_right_np=right_img_np,
            TOP_DELTA=0.01,
            dedup_rounding=1,
        )
        
        # ========= Post-processing =========
        avg_score_all = {}
        for b in list(all_scores_per_object.keys()):
            scores = all_scores_per_object[b]           # np.ndarray or list
            pts    = all_points_per_object[b]           # np.ndarray or list

            pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)      # (n,2)
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)   # (n,)
            if scores.size == 0:
                avg_score_all[b] = 0.0
                all_points_per_object[b] = np.empty((0, 2), dtype=np.float32)
                all_scores_per_object[b] = np.empty((0,), dtype=np.float32)
                continue
            avg_score = float(scores.mean())
            avg_score_all[b] = avg_score
            if avg_score < 0:
                all_points_per_object[b] = np.empty((0, 2), dtype=np.float32)
                all_scores_per_object[b] = np.empty((0,), dtype=np.float32)
                continue
            keep = scores >= 0
            if not np.any(keep):
                all_points_per_object[b] = np.empty((0, 2), dtype=np.float32)
                all_scores_per_object[b] = np.empty((0,), dtype=np.float32)
                continue
            P = pts[keep]       # (m,2)
            S = scores[keep]    # (m,)
            m = P.shape[0]     

            # Deduplication (preserving the first occurrence of each point)
            uniq_P, uniq_idx = np.unique(P, axis=0, return_index=True)
            order = np.sort(uniq_idx)              # Preserve the order of first occurrence
            P = P[order]
            S = S[order]
            m = P.shape[0]
            if m == 0:
                all_points_per_object[b] = np.empty((0, 2), dtype=np.float32)
                all_scores_per_object[b] = np.empty((0,), dtype=np.float32)
                continue
            # Filtering and Farthest Point Sampling (FPS) 
            if USE_FPS_FILTER: 
                k_half = (m + 1) // 2     
                top_idx = np.argsort(-S)[:k_half]  
                P_top = P[top_idx]
                S_top = S[top_idx]
                MAX_DIST = 330.0
                #print(f"{b+1}_S_top:",S_top)
                seed = int(np.argmax(S_top))
                selected = [seed]
                min_dists = np.linalg.norm(P_top - P_top[seed], axis=1)   # (k_half,)
                min_dists[seed] = -np.inf

                while len(selected) < 4:
                    nxt = int(np.argmax(min_dists))
                    cur_dist = min_dists[nxt]
                    if not np.isfinite(cur_dist):
                        break
                    if cur_dist > MAX_DIST:
                        min_dists[nxt] = -np.inf   
                        continue
                    selected.append(nxt)
                    d_new = np.linalg.norm(P_top - P_top[nxt], axis=1)
                    min_dists = np.minimum(min_dists, d_new)
                    min_dists[selected] = -np.inf
                sel_idx = np.array(selected, dtype=np.int64)
                P_sel = P_top[sel_idx]
                S_sel = S_top[sel_idx]
                # P_sel = P
                # S_sel = S

                all_points_per_object[b] = P_sel.astype(np.float32, copy=False)   # (k',2)
                all_scores_per_object[b] = S_sel.astype(np.float32, copy=False)   # (k',)
            else:
                k_n = int(min(1, m))
                top_n = np.argsort(-S)[:k_n]  
                P_sel = P[top_n]
                S_sel = S[top_n]
    
                all_points_per_object[b] = P_sel.astype(np.float32, copy=False)   # (k',2)
                all_scores_per_object[b] = S_sel.astype(np.float32, copy=False)   # (k',)

        #using the init_state_
        final_worklist = []
        for b, pts in all_points_per_object.items():
            if len(pts) == 0:
                continue
            point_coords = np.array(pts, dtype=np.float32)              # (N,2)
            point_coords = np.unique(point_coords, axis=0)
            point_labels = np.ones((len(point_coords),), dtype=np.int32)
            final_worklist.append((b, point_coords, point_labels))

        def _emit_result(b, point_coords, mask_to_save, ltbr):
            left_feat_pe_final = left_pe_feats_all[0][b]
            left_feat_pe_final = left_feat_pe_final.to(device)
            if ltbr is None:
                return
            crop_right_final = image_right.crop(ltbr).convert("RGB")
            right_feat_pe_final = pe_feature_from_pil(crop_right_final, model_pe, preprocess_pe, pe_adapter, device)
            score_final = float(torch.dot(left_feat_pe_final, right_feat_pe_final).item())
            x0, y0, x1, y1 = map(int, ltbr)
            coco_bbox = [x0, y0, int(x1 - x0), int(y1 - y0)]  # COCO [x,y,w,h]
            scene_results.append({
                "file_name": f"{scene_id:06d}/rgb/{right_path.name}",
                "category_id": int(b + 1),
                "bbox": coco_bbox,
                "score": score_final,
                "image_width": int(image_right.width),
                "image_height": int(image_right.height)
            })

        if USE_LOCAL_VIEW:
            # Batch the SAM2 image-encoder pass across objects (same fixed crop size per
            # object -> uniform shape, same as Tier2's candidate-scoring batching in
            # Candidate.py) instead of calling set_image/predict once per object. The mask
            # decoder is still invoked once per object inside predict_batch, with that
            # object's own Object_id, so per-object augmented-SAM masks are unchanged --
            # only the encoder invocation is batched.
            for chunk_start in range(0, len(final_worklist), LOCAL_VIEW_BATCH_SIZE):
                chunk = final_worklist[chunk_start:chunk_start + LOCAL_VIEW_BATCH_SIZE]

                crops = []
                point_coords_batch = []
                point_labels_batch = []
                crop_boxes = []
                for (b, point_coords, point_labels) in chunk:
                    image_scale_4, point_coords_used, crop_box = crop_around_point(
                        image_right,
                        point_coords,
                        image_np=right_img_np,
                    )
                    crops.append(image_scale_4)
                    point_coords_batch.append(point_coords_used)
                    point_labels_batch.append(point_labels)
                    crop_boxes.append(crop_box)

                predictor.set_image_batch(crops)
                predict_batch_kwargs = dict(
                    point_coords_batch=point_coords_batch,
                    point_labels_batch=point_labels_batch,
                    multimask_output=False,
                )
                if USE_AUGMENTED_PREDICTOR:
                    predict_batch_kwargs["Object_id_batch"] = [b + 1 for (b, _, _) in chunk]

                masks_batch, scores_batch, logits_batch = predictor.predict_batch(**predict_batch_kwargs)

                for idx, (b, point_coords, point_labels) in enumerate(chunk):
                    masks_final = masks_batch[idx]
                    mask_to_save = masks_final[0] if masks_final.ndim == 3 else masks_final
                    mask_to_save = postprocess_mask_preserve_format(mask_to_save, point_coords, radius=6)
                    if SAVE_IMAGES:
                        save_mask_image_with_bbox(
                            image_np=np.array(crops[idx]),
                            mask_np=mask_to_save,
                            point_coords=point_coords_batch[idx],
                            out_root=OUT_ROOT,
                            left_index_b=b,
                            right_path=right_path,
                            borders=False
                        )
                    _m2d = (mask_to_save > 0).astype(np.uint8)
                    _ltbr = tight_bbox_from_mask_scale(_m2d, crop_box=crop_boxes[idx])
                    _emit_result(b, point_coords, mask_to_save, _ltbr)
        else:
            for (b, point_coords, point_labels) in final_worklist:
                predict_kwargs = dict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False,
                )
                if USE_AUGMENTED_PREDICTOR:
                    predict_kwargs["Object_id"] = b + 1

                masks_final, scores_final, logits_final = predictor.predict(**predict_kwargs)

                mask_to_save = masks_final[0] if masks_final.ndim == 3 else masks_final
                mask_to_save = postprocess_mask_preserve_format(mask_to_save, point_coords, radius=6)
                if SAVE_IMAGES:
                    save_mask_image_with_bbox(
                        image_np=right_img_np,
                        mask_np=mask_to_save,
                        point_coords=point_coords,
                        out_root=OUT_ROOT,
                        left_index_b=b,
                        right_path=right_path,
                        borders=False
                    )
                _m2d = (mask_to_save > 0).astype(np.uint8)
                _ltbr = tight_bbox_from_mask(_m2d)
                _emit_result(b, point_coords, mask_to_save, _ltbr)
    out_json_path = Path(f"Output/{DATASET_NAME}/{scene_id:06d}/pred_results_USE_PE_ADAPTER={USE_PE_ADAPTER}_USE_AUGMENTED_SAM={USE_AUGMENTED_PREDICTOR}.json")
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(scene_results, f, ensure_ascii=False)
