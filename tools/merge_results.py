import json
from pathlib import Path


def merge_coco_gt(gt_path_1, gt_path_2, out_path):
    """
    Merge two COCO GT jsons into one.

    Assumptions:
    - images in gt1 and gt2 are disjoint (no shared physical images).
    - We keep all IDs from gt1 as-is.
    - All image_id and annotation id from gt2 are shifted by an offset so there is no collision.

    Returns:
        img_id_offset: the value added to image_id from gt2
    """
    gt_path_1 = Path(gt_path_1)
    gt_path_2 = Path(gt_path_2)
    out_path = Path(out_path)

    with gt_path_1.open('r') as f:
        gt1 = json.load(f)
    with gt_path_2.open('r') as f:
        gt2 = json.load(f)

    # If there is no image in gt1, start from 0, otherwise from max_id + 1
    if len(gt1["images"]) > 0:
        max_img_id_1 = max(img["id"] for img in gt1["images"])
    else:
        max_img_id_1 = -1
    img_id_offset = max_img_id_1 + 1

    if len(gt1["annotations"]) > 0:
        max_ann_id_1 = max(ann["id"] for ann in gt1["annotations"])
    else:
        max_ann_id_1 = -1
    ann_id_offset = max_ann_id_1 + 1

    # Start merged skeleton mostly from gt1
    merged = {
        "info": gt1.get("info", {}),
        "licenses": gt1.get("licenses", []),
        "images": [],
        "annotations": [],
        "categories": gt1.get("categories", []),
    }

    # 1) images: keep gt1 ids, shift gt2 ids
    merged_images = []
    # gt1 images: unchanged
    for img in gt1["images"]:
        merged_images.append(img)

    # gt2 images: id += img_id_offset
    for img in gt2["images"]:
        new_img = dict(img)
        new_img["id"] = img["id"] + img_id_offset
        merged_images.append(new_img)

    merged["images"] = merged_images

    # 2) annotations: keep gt1, shift gt2 (both ann id and image_id)
    merged_anns = []
    # gt1 annotations: unchanged
    for ann in gt1["annotations"]:
        merged_anns.append(ann)

    # gt2 annotations: id += ann_id_offset, image_id += img_id_offset
    for ann in gt2["annotations"]:
        new_ann = dict(ann)
        new_ann["id"] = ann["id"] + ann_id_offset
        new_ann["image_id"] = ann["image_id"] + img_id_offset
        merged_anns.append(new_ann)

    merged["annotations"] = merged_anns

    # Save merged GT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        json.dump(merged, f)
    print(f"Merged GT saved to: {out_path}")
    print(f"Image ID offset used for second GT: {img_id_offset}")

    return img_id_offset


def merge_coco_dt(dt_path_1, dt_path_2, out_path, img_id_offset_for_dt2):
    """
    Merge two COCO detection result jsons (list of dicts).

    - dt1 is kept as-is.
    - dt2's image_id are shifted by img_id_offset_for_dt2 so they match the merged GT.
    """
    dt_path_1 = Path(dt_path_1)
    dt_path_2 = Path(dt_path_2)
    out_path = Path(out_path)

    with dt_path_1.open('r') as f:
        dt1 = json.load(f)
    with dt_path_2.open('r') as f:
        dt2 = json.load(f)

    merged_dt = []
    # dt1: unchanged
    for det in dt1:
        merged_dt.append(det)

    # dt2: shift image_id
    for det in dt2:
        new_det = dict(det)
        new_det["image_id"] = det["image_id"] + img_id_offset_for_dt2
        merged_dt.append(new_det)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        json.dump(merged_dt, f)
    print(f"Merged detections saved to: {out_path}")


if __name__ == "__main__":
    # Base directory (modify if your path is different)
    base = Path("../RoboTools/test/000027")
    base_results = Path("../RoboTools/test/high_results")

    # ---- GT paths ----
    gt_hard = base / "instances_test_4_hard.json"   # scenes 31-40
    gt_easy = base / "instances_test_4_easy.json"   # scenes 41-52
    gt_merged = base / "instances_test_4_mergeall.json"

    # ---- detection result paths ----
    dt_31_40 = base_results / "merged_coco_results_31-40_0117_00.json"
    dt_41_52 = base_results / "merged_coco_results_41-52_0117_00.json"
    dt_merged = base_results / "merged_coco_results_31-52_0117_00.json"

    # 1) merge GT; we keep "hard" IDs as-is, shift "easy"
    img_id_offset = merge_coco_gt(gt_hard, gt_easy, gt_merged)

    # 2) merge detections; we keep 31-40 as-is, shift 41-52 by the same offset
    merge_coco_dt(dt_31_40, dt_41_52, dt_merged, img_id_offset)

    print("All done.")