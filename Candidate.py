# adapter.py
# Select candidate points and apply filtering

import torch
import numpy as np

from utils.data import stride_filter_mask
from utils.build import pe_feature_from_pil, dino_cls_from_pil
from utils.image import crop_around_point
from utils.build import tight_bbox_from_mask, tight_bbox_from_mask_scale

def process_one_right_image_all(
    image_right,
    image_right_name,
    scale_right_x,
    scale_right_y,
    W2,
    feat_r,                         # (C,H2,W2), recommended to be externally F.normalize(feat_r, p=2, dim=0)
    predictor,                      # SAM2 predictor (right image already set_image)
    device: torch.device,
    PATCH_SIZE: int,
    IMAGE_SIZE: int,
    mean, std,
    # multi-template inputs
    left_pairs_all,                 # list[ list[... per object ...] ], len = SEQ_LEN
    sel_left_all,                   # list[Tensor(B, K_t) bool]
    left_vecs_all,                  # list[Tensor(B, K_t, C)]
    H1_all, W1_all,                 # list[int], grid size of each template
    M_TARGET: int,
    S1_THR: float,
    model,
    model_pe,
    preprocess_pe,
    pe_adapter,
    left_cls_all,
    left_pe_feats_all,              # list[ list[Tensor(C,)|None] ], outer len=SEQ_LEN, inner len=B
    left_fg_mean_all,
    cfg: dict,                      # config
    TOP_DELTA: float = 0.015,
    dedup_rounding: int = 1,        # pixel granularity for cross-template right-image deduplication (1=integer pixel)
):
    """
    Merged t-loop version (without changing your scoring/filtering logic):
      1) For each template, compute heatmaps -> 'good' right-image candidate points (pixel coordinates), only geometric filtering (stride_filter + S1_THR)
      2) At the per-object level, deduplicate right-image candidate points proposed by all templates at pixel level (run SAM2+PE only once per unique point)
      3) For each unique right-image point:
           - Run SAM2 to get mask, crop ROI, extract right-image PE feature (only once)
           - For all templates that proposed this point, compute dot product with their left-image PE features and write back to their candidate pools
      4) For each template/object, select from candidate pool using TOP_DELTA
         best_points_this_round[t][b], best_scores_this_round[t][b]
      5) Aggregate these best_* into all_points_per_object / all_scores_per_object

    This reuses downstream computation for right-image points while strictly preserving the original TOP_DELTA selection logic.
    """

    USE_WHICH_FEATURE = cfg["adapter"]["use_which_feature"]
    USE_LOCAL_VIEW = cfg["post_processing"]["use_local_view"]

    print("feat_r.shape:",feat_r.shape)


    SEQ_LEN = len(left_vecs_all)
    assert SEQ_LEN == len(sel_left_all) == len(H1_all) == len(W1_all) == len(left_pairs_all) == len(left_pe_feats_all)
    B = sel_left_all[0].shape[0]

    # ------ Log Name -------
    names_by_b = []
    for b in range(B):
        nm = getattr(left_pairs_all[0][b][0], "name", f"obj_{b+1:02d}") if left_pairs_all and left_pairs_all[0] else f"obj_{b+1:02d}"
        names_by_b.append(nm)

    # ------ Candidate pool per template/object (later filtered by TOP_DELTA) ------
    # cand_pool[b][t] = list of (pt_xy(np.float32[2],), score(float))
    cand_pool = [ {t: [] for t in range(SEQ_LEN)} for _ in range(B) ]

    # ------ To avoid repeated expensive computation: deduplicate right-image points across templates ------
    # per object:
    #   unique_pts[b]: dict[key=(x_int,y_int)] -> {"pt": np.array([x,y],float32), "ts": set(template_ids)}
    unique_pts = [ dict() for _ in range(B) ]

    # 1) Iterate templates: dense similarity + geometric filtering; collect right-image candidate points (geometry only, no scoring)
    for t in range(SEQ_LEN):
        sel_left_t  = sel_left_all[t]         # (B, K_t)
        left_vecs_t = left_vecs_all[t]        # (B, K_t, C)
        H1, W1      = H1_all[t], W1_all[t]
        K_t = H1 * W1

        # dense similarity
        # print("$$$$$$$$$$$$left_vecs_t.shape:",left_vecs_t.shape)
        # print("$$$$$$$$$$$$feat_r.shape:",feat_r.shape)
        heatmaps = torch.einsum("bkc,chw->bkhw", left_vecs_t, feat_r)  # (B, K_t, H2, W2)
        flat = heatmaps.view(B, K_t, -1)
        s1, j_best = flat.max(dim=2)            # (B,K_t), (B,K_t)
        y2 = j_best // W2
        x2 = j_best %  W2
        # print("y2,shape:",y2.shape)
        # print(f"y2:{y2},x2:{x2}")
        locs_2d_right = (torch.stack((y2, x2), dim=-1) + 0.5) * PATCH_SIZE  # (B,K_t,2)

        # stride sparsification + threshold
        rows_grid = torch.arange(H1, device=sel_left_t.device).unsqueeze(1).expand(H1, W1).reshape(-1)
        cols_grid = torch.arange(W1, device=sel_left_t.device).unsqueeze(0).expand(H1, W1).reshape(-1)
        valid_mask = torch.zeros_like(sel_left_t, dtype=torch.bool)

        for b in range(B):
            keep_mask_b, stride_b = stride_filter_mask(
                sel_row=sel_left_t[b],
                H1=H1, W1=W1, m=M_TARGET,
                keep_edges=False,
                rows_grid=rows_grid, cols_grid=cols_grid,
                Filter=True
            )
            valid_mask[b] = keep_mask_b

        good = valid_mask & (s1 > S1_THR)  # (B,K_t)

        # Collect candidate points into unique_pts (store points only, no scoring)
        for b in range(B):
            gl = good[b]
            if not bool(gl.any()):
                continue
            #print("locs_2d_right[b, gl]:",locs_2d_right[b, gl])
            locs_right_b = torch.unique(locs_2d_right[b, gl], dim=0)    # deduplicate in patch space
            #print("locs_right_b:",locs_right_b)
            locs_right_np = locs_right_b.detach().cpu().numpy()
            xy_right_px = np.stack([
                (locs_right_np[:, 1] * scale_right_x),  # x
                (locs_right_np[:, 0] * scale_right_y),  # y
            ], axis=1)                                   # (M,2)
            #print("xy_right_px:",xy_right_px)
            # Deduplicate at pixel level and record source templates
            for pt in xy_right_px:
                x_f, y_f = float(pt[0]), float(pt[1])
                if dedup_rounding > 1:
                    x_k = int(round(x_f / dedup_rounding) * dedup_rounding)
                    y_k = int(round(y_f / dedup_rounding) * dedup_rounding)
                else:
                    x_k = int(round(x_f))
                    y_k = int(round(y_f))
                key = (x_k, y_k)

                bucket = unique_pts[b].get(key)
                if bucket is None:
                    unique_pts[b][key] = {"pt": np.array([x_f, y_f], dtype=np.float32),
                                          "ts": set([t])}
                else:
                    bucket["ts"].add(t)

        # clean
        del heatmaps, flat, s1, j_best, y2, x2, locs_2d_right, valid_mask, good
        if device.type == "cuda":
            torch.cuda.empty_cache()
    """
    2)  Run SAM2 + right-image PE only once for each unique right-image point,
     then distribute the computed score to the candidate pools of the templates that proposed it
    """
    for b in range(B):
        cand_dict = unique_pts[b]
        if not cand_dict:
            print(f"[Right={image_right_name}] [{b+1}/{B}] {names_by_b[b]}: no candidates after merge.")
            continue

        print(f"[Right={image_right_name}] [{b+1}/{B}] {names_by_b[b]}: unique candidates = {len(cand_dict)}")

        for key, info in cand_dict.items():
            pt_xy = info["pt"]                 # np.array([x,y], float32)
            src_ts = sorted(list(info["ts"]))  # Templates that generate / support this candidate point

            # SAM with one point
            point_coords = pt_xy.reshape(1, 2).astype(np.float32)  # (1,2)
            point_labels = np.array([1], dtype=np.int32)
            #print("point_coords:",point_coords)

            if USE_LOCAL_VIEW:
                image_scale_4, point_coords_scaled, crop_box = crop_around_point(
                    image_right,
                    point_coords,
                )
                predictor.set_image(image_scale_4)
                masks_i, scores_i, logits_i = predictor.predict(
                    point_coords=point_coords_scaled,
                    point_labels=point_labels,
                    multimask_output=False,
                )
            else:
                masks_i, scores_i, logits_i = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False,
                )
       
            if masks_i is None or len(masks_i) == 0:
                continue

            mask_np_right = masks_i[0] if masks_i.ndim == 3 else masks_i
            mask_np_right = (mask_np_right > 0).astype(np.uint8)
            if USE_LOCAL_VIEW:
                bbox_right = tight_bbox_from_mask_scale(mask_np_right,crop_box=crop_box)
            else:
                bbox_right = tight_bbox_from_mask(mask_np_right)
            if bbox_right is None:
                continue


            crop_right = image_right.crop(bbox_right).convert("RGB")

            # Right image features (normalized)
            if USE_WHICH_FEATURE == "pe_feature":
                right_feat_pe = pe_feature_from_pil(crop_right, model_pe, preprocess_pe, pe_adapter, device)
                for t in src_ts:
                    left_feat_pe = left_pe_feats_all[t][b]
                    if left_feat_pe is None:
                        continue
                    score = float(torch.dot(left_feat_pe, right_feat_pe).item())
                    cand_pool[b][t].append( (pt_xy.copy(), score) )
            elif USE_WHICH_FEATURE == "dino_cls":
                right_dino_cls = dino_cls_from_pil(crop_right, model, mean, std, cfg, device)

                for t in src_ts:
                    left_dino_cls = left_cls_all[t][b]
                    if left_dino_cls is None:
                        continue
                    score = float(torch.dot(left_dino_cls, right_dino_cls).item())
                    cand_pool[b][t].append( (pt_xy.copy(), score) )
         
            # clean
            del masks_i, logits_i
        if device.type == "cuda":
            torch.cuda.empty_cache()



    all_points_per_object: dict[int, list[np.ndarray]] = {b: [] for b in range(B)}
    all_scores_per_object: dict[int, list[float]]      = {b: [] for b in range(B)}

    for t in range(SEQ_LEN):
        for b in range(B):
            pairs = cand_pool[b][t]   # list of (pt, score)
            if not pairs:
                continue


            best_score = max(s for (_, s) in pairs)
            kept = [(pt, s) for (pt, s) in pairs if s >= best_score - TOP_DELTA]
            #kept = [(pt, s) for (pt, s) in pairs if s >= Bset_scores_all_SEQ[b] - TOP_DELTA]

            pts_np = np.stack([pt for (pt, _) in kept], axis=0).astype(np.float32) if kept else None
            sco_ls = [s for (_, s) in kept]

            if pts_np is not None and len(sco_ls) > 0:
                for (pt, s) in kept:
                    all_points_per_object[b].append(np.asarray(pt, dtype=np.float32))
                    all_scores_per_object[b].append(float(s))

            # print(f"[Right={image_right_name}] [t={t} b={b+1}] {names_by_b[b]}: "
            #       f"best_score={best_score:.6f}, kept={len(sco_ls)} within Δ={TOP_DELTA}")
    # for b in range(B):
    #     print(f"all_points_per_object length for {b}:", len(all_points_per_object[b]))

    return all_points_per_object, all_scores_per_object




def process_one_right_image_all_withDense(
    image_right,
    image_right_name,
    scale_right_x,
    scale_right_y,
    W2,
    feat_r,                         # (C,H2,W2), recommended: F.normalize(feat_r, p=2, dim=0)
    predictor,                      # SAM2 predictor (right image already set_image)
    device: torch.device,
    PATCH_SIZE: int,
    IMAGE_SIZE: int,
    mean, std,
    # multi-template inputs
    left_pairs_all,                 # list[list[... per object ...]], len = SEQ_LEN
    sel_left_all,                   # list[Tensor(B, K_t) bool]
    left_vecs_all,                  # list[Tensor(B, K_t, C)]
    H1_all, W1_all,                 # list[int]
    M_TARGET: int,
    S1_THR: float,
    model,
    model_pe,
    preprocess_pe,
    pe_adapter,
    left_cls_all,
    left_pe_feats_all,              # list[list[Tensor(C,)|None]], outer len=SEQ_LEN, inner len=B
    left_fg_mean_all,
    cfg: dict,
    TOP_DELTA: float = 0.015,
    dedup_rounding: int = 1,        # pixel-level rounding for right-point de-dup across templates
):
    """
    Behavior:
      - For each template t: dense match left_vecs_t against feat_r to get best right locations + s1 scores.
      - If USE_WHICH_FEATURE == "only_Dense":
          * Skip SAM2 + ROI feature similarity.
          * Use dense score s1[b,k] directly as the score for the matched right pixel.
          * De-dup within (b,t) so each right pixel appears once, using aggregation over multiple left tokens.
      - Else:
          * Collect right candidates (geometry filtered) into unique_pts[b] across templates.
          * For each unique right point: run SAM2 once, crop ROI, extract right feature (PE or DINO CLS),
            then distribute scores back to templates that proposed this point.
      - Finally: per (b,t), keep points with score >= best_score - TOP_DELTA and aggregate to outputs.

    Notes:
      - This function expects global USE_WHICH_FEATURE to exist, as in your original code.
      - You can choose how to aggregate multiple left tokens hitting the same right pixel in only_Dense mode
        by setting DENSE_DEDUP_AGG below.
    """

    USE_WHICH_FEATURE = cfg["adapter"]["use_which_feature"]


    print("feat_r.shape:", feat_r.shape)

    # -----------------------------
    # only_Dense de-dup aggregation:
    #   "max"  : keep max s1 for the right pixel
    #   "sum"  : sum s1 across all left tokens that hit the same right pixel
    #   "mean" : average s1 across hits
    # -----------------------------
    DENSE_DEDUP_AGG = "max"  # change to "sum" or "mean" if you want

    SEQ_LEN = len(left_vecs_all)
    assert SEQ_LEN == len(sel_left_all) == len(H1_all) == len(W1_all) == len(left_pairs_all) == len(left_pe_feats_all)
    B = sel_left_all[0].shape[0]

    # ---- logging names ----
    names_by_b = []
    for b in range(B):
        nm = getattr(left_pairs_all[0][b][0], "name", f"obj_{b+1:02d}") if left_pairs_all and left_pairs_all[0] else f"obj_{b+1:02d}"
        names_by_b.append(nm)

    # ---- candidate pool per object, per template ----
    # cand_pool[b][t] = list of (pt_xy(np.float32[2]), score(float))
    cand_pool = [{t: [] for t in range(SEQ_LEN)} for _ in range(B)]

    # ---- for SAM2+feature reuse (non-only_Dense paths) ----
    unique_pts = [dict() for _ in range(B)]

    # =========================================================
    # 1) Iterate templates: dense match -> geometry filter -> candidates
    # =========================================================
    for t in range(SEQ_LEN):
        sel_left_t = sel_left_all[t]      # (B, K_t)
        left_vecs_t = left_vecs_all[t]    # (B, K_t, C)
        H1, W1 = H1_all[t], W1_all[t]
        K_t = H1 * W1

        # ---- dense similarity ----
        heatmaps = torch.einsum("bkc,chw->bkhw", left_vecs_t, feat_r)  # (B,K_t,H2,W2)
        flat = heatmaps.view(B, K_t, -1)
        s1, j_best = flat.max(dim=2)  # (B,K_t), (B,K_t)

        y2 = j_best // W2
        x2 = j_best % W2
        locs_2d_right = (torch.stack((y2, x2), dim=-1) + 0.5) * PATCH_SIZE  # (B,K_t,2) in (y,x) coords

        # ---- stride filter + threshold ----
        rows_grid = torch.arange(H1, device=sel_left_t.device).unsqueeze(1).expand(H1, W1).reshape(-1)
        cols_grid = torch.arange(W1, device=sel_left_t.device).unsqueeze(0).expand(H1, W1).reshape(-1)
        valid_mask = torch.zeros_like(sel_left_t, dtype=torch.bool)

        for b in range(B):
            keep_mask_b, _stride_b = stride_filter_mask(
                sel_row=sel_left_t[b],
                H1=H1, W1=W1, m=M_TARGET,
                keep_edges=False,
                rows_grid=rows_grid, cols_grid=cols_grid,
                Filter=True
            )
            valid_mask[b] = keep_mask_b

        good = valid_mask & (s1 > S1_THR)  # (B,K_t)

        # =====================================================
        # only_Dense: directly write (right_point, dense_score) into cand_pool
        # =====================================================
        if USE_WHICH_FEATURE == "only_Dense":
            for b in range(B):
                gl = good[b]
                if not bool(gl.any()):
                    continue

                k_idx = torch.nonzero(gl, as_tuple=False).squeeze(1)  # (M,)

                locs_sel = locs_2d_right[b, k_idx]  # (M,2) (y,x)
                s_sel = s1[b, k_idx]                # (M,)

                locs_np = locs_sel.detach().cpu().numpy()
                #s_np = s_sel.detach().cpu().numpy().astype(np.float32)
                s_np = s_sel.detach().float().cpu().numpy()

                # De-dup by rounded pixel key; aggregate multiple hits per key
                # store:
                #   for "max": key -> (pt_xy, best_score)
                #   for "sum"/"mean": key -> (pt_xy, sum_score, count)
                best_by_key = {}

                for i in range(locs_np.shape[0]):
                    y_f = float(locs_np[i, 0] * scale_right_y)
                    x_f = float(locs_np[i, 1] * scale_right_x)
                    score = float(s_np[i])

                    if dedup_rounding > 1:
                        x_k = int(round(x_f / dedup_rounding) * dedup_rounding)
                        y_k = int(round(y_f / dedup_rounding) * dedup_rounding)
                    else:
                        x_k = int(round(x_f))
                        y_k = int(round(y_f))
                    key = (x_k, y_k)

                    if DENSE_DEDUP_AGG == "max":
                        prev = best_by_key.get(key)
                        if (prev is None) or (score > prev[1]):
                            best_by_key[key] = (np.array([x_f, y_f], dtype=np.float32), score)
                    else:
                        prev = best_by_key.get(key)
                        if prev is None:
                            best_by_key[key] = (np.array([x_f, y_f], dtype=np.float32), score, 1)
                        else:
                            pt_xy_prev, sum_score, cnt = prev
                            best_by_key[key] = (pt_xy_prev, sum_score + score, cnt + 1)

                # Write aggregated results into cand_pool
                if DENSE_DEDUP_AGG == "max":
                    for _key, (pt_xy, sc) in best_by_key.items():
                        cand_pool[b][t].append((pt_xy, float(sc)))
                elif DENSE_DEDUP_AGG == "sum":
                    for _key, (pt_xy, sum_sc, cnt) in best_by_key.items():
                        cand_pool[b][t].append((pt_xy, float(sum_sc)))
                elif DENSE_DEDUP_AGG == "mean":
                    for _key, (pt_xy, sum_sc, cnt) in best_by_key.items():
                        cand_pool[b][t].append((pt_xy, float(sum_sc) / max(1, int(cnt))))
                else:
                    raise ValueError(f"Unknown DENSE_DEDUP_AGG={DENSE_DEDUP_AGG}")

            # cleanup + next template
            del heatmaps, flat, s1, j_best, y2, x2, locs_2d_right, valid_mask, good
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        # =====================================================
        # Original path: collect candidate points into unique_pts (no scoring yet)
        # =====================================================
        for b in range(B):
            gl = good[b]
            if not bool(gl.any()):
                continue

            # De-dup in patch space
            locs_right_b = torch.unique(locs_2d_right[b, gl], dim=0)
            locs_right_np = locs_right_b.detach().cpu().numpy()

            # Convert to right-image pixel coords (x,y)
            xy_right_px = np.stack([
                (locs_right_np[:, 1] * scale_right_x),
                (locs_right_np[:, 0] * scale_right_y),
            ], axis=1)

            for pt in xy_right_px:
                x_f, y_f = float(pt[0]), float(pt[1])

                if dedup_rounding > 1:
                    x_k = int(round(x_f / dedup_rounding) * dedup_rounding)
                    y_k = int(round(y_f / dedup_rounding) * dedup_rounding)
                else:
                    x_k = int(round(x_f))
                    y_k = int(round(y_f))
                key = (x_k, y_k)

                bucket = unique_pts[b].get(key)
                if bucket is None:
                    unique_pts[b][key] = {"pt": np.array([x_f, y_f], dtype=np.float32), "ts": set([t])}
                else:
                    bucket["ts"].add(t)

        del heatmaps, flat, s1, j_best, y2, x2, locs_2d_right, valid_mask, good
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # =========================================================
    # 2) If not only_Dense: run SAM2+ROI feature once per unique right point, distribute scores to cand_pool
    # =========================================================
    if USE_WHICH_FEATURE != "only_Dense":
        for b in range(B):
            cand_dict = unique_pts[b]
            if not cand_dict:
                print(f"[Right={image_right_name}] [{b+1}/{B}] {names_by_b[b]}: no candidates after merge.")
                continue

            print(f"[Right={image_right_name}] [{b+1}/{B}] {names_by_b[b]}: unique candidates = {len(cand_dict)}")

            for _key, info in cand_dict.items():
                pt_xy = info["pt"]                 # np.array([x,y], float32)
                src_ts = sorted(list(info["ts"]))  # templates that proposed this point

                # SAM2 single-point prompt
                point_coords = pt_xy.reshape(1, 2).astype(np.float32)
                point_labels = np.array([1], dtype=np.int32)

                masks_i, scores_i, logits_i = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False,
                )
                if masks_i is None or len(masks_i) == 0:
                    continue

                mask_np_right = masks_i[0] if masks_i.ndim == 3 else masks_i
                mask_np_right = (mask_np_right > 0).astype(np.uint8)

                bbox_right = tight_bbox_from_mask(mask_np_right)
                if bbox_right is None:
                    continue

                crop_right = image_right.crop(bbox_right).convert("RGB")

                if USE_WHICH_FEATURE == "pe_feature":
                    right_feat_pe = pe_feature_from_pil(crop_right, model_pe, preprocess_pe, pe_adapter, device)
                    for t in src_ts:
                        left_feat_pe = left_pe_feats_all[t][b]
                        if left_feat_pe is None:
                            continue
                        score = float(torch.dot(left_feat_pe, right_feat_pe).item())
                        cand_pool[b][t].append((pt_xy.copy(), score))

                elif USE_WHICH_FEATURE == "dino_cls":
                    right_dino_cls = dino_cls_from_pil(crop_right, model, mean, std, cfg, device)
                    for t in src_ts:
                        left_dino_cls = left_cls_all[t][b]
                        if left_dino_cls is None:
                            continue
                        score = float(torch.dot(left_dino_cls, right_dino_cls).item())
                        cand_pool[b][t].append((pt_xy.copy(), score))

                else:
                    raise ValueError(f"Unknown USE_WHICH_FEATURE={USE_WHICH_FEATURE}")

                del masks_i, logits_i

            if device.type == "cuda":
                torch.cuda.empty_cache()

    # =========================================================
    # 3) Keep points within TOP_DELTA per (b,t), then aggregate across templates
    # =========================================================
    all_points_per_object: dict[int, list[np.ndarray]] = {b: [] for b in range(B)}
    all_scores_per_object: dict[int, list[float]] = {b: [] for b in range(B)}

    for t in range(SEQ_LEN):
        for b in range(B):
            pairs = cand_pool[b][t]
            if not pairs:
                continue

            best_score = max(s for (_, s) in pairs)
            kept = [(pt, s) for (pt, s) in pairs if s >= best_score - TOP_DELTA]

            if kept:
                for (pt, s) in kept:
                    all_points_per_object[b].append(np.asarray(pt, dtype=np.float32))
                    all_scores_per_object[b].append(float(s))

            # print(
            #     f"[Right={image_right_name}] [t={t} b={b+1}] {names_by_b[b]}: "
            #     f"best_score={best_score:.6f}, kept={len(kept)} within Δ={TOP_DELTA}"
            # )

    # for b in range(B):
    #     print(f"all_points_per_object length for {b}:", len(all_points_per_object[b]))

    return all_points_per_object, all_scores_per_object

