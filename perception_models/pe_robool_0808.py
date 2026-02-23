import argparse
import os
import json
import re
import torch
from PIL import Image
import numpy as np
from pathlib import Path

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

# Scene -> list of valid object IDs (1-based, matching template file stems like '000007')
valid_object_in_scenes = {
    "000001":[11,13,15,17,18,20], "000002":[11,13,15,17,18,20], "000003":[8,9,10,12,14,16,19],
    "000004":[1,2,3,4,5,6,7], "000005":[7,14,15,17,18,20], "000006":[1,3,8,9,12],
    "000007":[8,9,10,14,16,19], "000008":[1,2,3,4,5,6,7], "000009":[6,9,13,15,17],
    "000010":[7,8,13,14], "000011":[3,4,8,9,10,19], "000012":[8,9,10,12,14,16,19],
    "000013":[11,13,15,17,18,20], "000014":[1,2,3,4,5,6,7], "000015":[11,13,15,17,18,20],
    "000016":[11,13,15,17,18,20], "000017":[8,9,10,12,14,16,19], "000018":[1,2,3,4,5,6,7],
    "000019":[1,7,8,11,18], "000020":[10,11,12,14,16], "000021":[8,9,10,12,14,16,19],
    "000022":[1,2,3,4,5,6,7], "000023":[4,5,10,15], "000024":[5,6,9,13,15,17,20]
}

def load_model():
    print("Loading model...")
    print("CLIP configs:", pe.CLIP.available_configs())
    model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True).cuda()
    preprocess = transforms.get_image_transform(model.image_size)
    return model, preprocess

def load_templates(template_dir, preprocess):
    """
    Load all templates (JPEG/PNG) and return a list of (filename, PIL.Image).
    """
    template_images = []
    for fname in sorted(os.listdir(template_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img = Image.open(os.path.join(template_dir, fname)).convert('RGB')
        template_images.append((fname, img))
    print(f"Loaded {len(template_images)} template images.")
    return template_images

def compute_features(model, images, preprocess):
    """
    Encode all template images via the vision encoder to normalized feature vectors.
    """
    tensors = []
    for _, img in images:
        tensors.append(preprocess(img).unsqueeze(0))
    batch = torch.cat(tensors, dim=0).cuda()
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

def parse_object_id_from_template_name(template_name):
    """
    Parse object_id from template filename stem, e.g. '000007.jpg' -> 7.
    Falls back to None if not numeric.
    """
    stem = Path(template_name).stem
    m = re.search(r'\d+', stem)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None

def process_scene_image(scene_image_path, num_rows, num_cols,
                        model, preprocess, template_images, template_features, output_dir):
    """
    - Split scene image into patches (regular grid + half-offset grid).
    - For each VALID object template for this scene, find best-matching patch.
    - Save the best patch as '<object_id_padded>.jpg' into subfolder '<rows>_<cols>_<object_id_padded>'.
    - Return a dict { object_id(int): {"x": int, "y": int} } recording top-left coordinates of the chosen patch.
    """
    scene_img = Image.open(scene_image_path).convert('RGB')
    W, H = scene_img.size
    patch_w, patch_h = W // num_cols, H // num_rows

    # Identify scene_id from path (e.g., '000007')
    scene_id = Path(scene_image_path).parent.name
    valid_template_ids = set(valid_object_in_scenes.get(scene_id, []))

    # Build patch list with their top-left coordinates
    patches = []       # list of (PIL.Image, x, y)
    patch_tensors = [] # list of (1, C, H, W) tensors

    # Regular grid patches
    # Candidate offsets: (0,0) plus half-cell shifts if patch size allows
    offsets = [(0, 0)]
    if patch_w >= 2:
        offsets.append((patch_w // 2, 0))
    if patch_h >= 2:
        offsets.append((0, patch_h // 2))
    if patch_w >= 2 and patch_h >= 2:
        offsets.append((patch_w // 2, patch_h // 2))

    # For each offset, slide with step = patch size, clamp to bounds
    for off_x, off_y in offsets:
        # Ensure starting positions are valid
        start_y = max(0, off_y)
        start_x = max(0, off_x)

        # Slide window without exceeding the image boundary
        for upper in range(start_y, H - patch_h + 1, patch_h):
            for left in range(start_x, W - patch_w + 1, patch_w):
                right = left + patch_w
                lower = upper + patch_h

                # Safety check (should already be guaranteed by the ranges)
                if right > W or lower > H:
                    continue

                patch = scene_img.crop((left, upper, right, lower))
                patches.append((patch, int(left), int(upper)))
                patch_tensors.append(preprocess(patch).unsqueeze(0))

    all_patches_batch = torch.cat(patch_tensors, dim=0).cuda()
    with torch.no_grad():
        patch_features = model.encode_image(all_patches_batch)
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)

    # For this scene, collect coordinates chosen per valid object_id
    chosen_coords = {}  # object_id -> {"x": int, "y": int}

    # Iterate over all templates, but only keep those whose object_id is valid for this scene
    for idx, (template_name, _) in enumerate(template_images):
        object_id = parse_object_id_from_template_name(template_name)
        if object_id is None or object_id not in valid_template_ids:
            continue

        template_feat = template_features[idx:idx+1]
        # Similarity against all patches
        logits = (100.0 * template_feat @ patch_features.T).softmax(dim=-1).cpu().numpy()[0]
        best_idx = int(np.argmax(logits))
        best_patch, px, py = patches[best_idx]

        # Folder per object within this grid: '<rows>_<cols>_<object_id_padded>'
        object_id_padded = f"{object_id:06d}"
        save_folder = output_dir / f"{num_rows}_{num_cols}_{object_id_padded}"
        save_folder.mkdir(parents=True, exist_ok=True)

        # Save best patch as '000007.jpg' (no coordinates in the filename)
        save_path = save_folder / f"000007.jpg"
        best_patch.convert('RGB').save(save_path, format='JPEG')
        print(f"Saved: {save_path} (patch_top_left=({px},{py}))")

        # Record chosen coordinates for JSON summary
        chosen_coords[object_id] = {"x": int(px), "y": int(py)}

    return chosen_coords

def merge_summary(summary_path: Path, scene_id: str, valid_ids, grid_key: str, coords_map):
    """
    Merge the per-run coordinates into a single global JSON:
      summary[scene_id] = {
        "valid_object_ids": [...],
        "objects": {
          "<object_id>": { "<grid_key>": {"x": int, "y": int} }
        }
      }
    If summary exists, load and update; else create new.
    """
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {}

    scene_entry = summary.setdefault(scene_id, {"valid_object_ids": [], "objects": {}})

    # Keep valid_object_ids unique and sorted for stability
    merged_valid = sorted(set(scene_entry.get("valid_object_ids", [])) | set(valid_ids))
    scene_entry["valid_object_ids"] = merged_valid

    # Insert/merge object coords under this grid_key
    objects_entry = scene_entry.setdefault("objects", {})
    for oid, xy in coords_map.items():
        oid_str = str(oid)
        obj_grids = objects_entry.setdefault(oid_str, {})
        obj_grids[grid_key] = {"x": int(xy["x"]), "y": int(xy["y"])}

    summary[scene_id] = scene_entry

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_rows', type=int, required=True, help='Number of rows for grid split')
    parser.add_argument('--num_cols', type=int, required=True, help='Number of columns for grid split')
    parser.add_argument('--scene_dir', type=str, default='../RoboTools_init_frames/slide_window_0808',
                        help='Directory containing scene folders (each with the scene image, e.g., 00000.png)')
    parser.add_argument('--output_dir', type=str, default='../RoboTools_init_frames/slide_window_0808',
                        help='Root directory where cropped patches will be saved (per scene folder)')
    parser.add_argument('--template_dir', type=str, default='../RoboTools_init_frames/template_images',
                        help='Directory with template images')
    parser.add_argument('--summary_json', type=str, default='../RoboTools_init_frames/slide_window_0808/patch_summary.json',
                        help='Path for the global JSON summary to write/merge')
    args = parser.parse_args()

    # Load model + templates
    model, preprocess = load_model()
    template_images = load_templates(args.template_dir, preprocess)
    template_features = compute_features(model, template_images, preprocess)

    scene_root = Path(args.scene_dir)
    out_root = Path(args.output_dir)
    summary_path = Path(args.summary_json)
    grid_key = f"{args.num_rows}_{args.num_cols}"

    # Process all scenes found under scene_dir
    for scene_folder in sorted(scene_root.iterdir()):
        if not scene_folder.is_dir():
            continue

        # Expect a single image named '00000.png' inside each scene folder (as in your original code)
        scene_image_path = scene_folder / '00000.png'
        if not scene_image_path.exists():
            print(f"Missing image in {scene_folder}, skipping.")
            continue

        scene_id = scene_folder.name
        valid_ids = valid_object_in_scenes.get(scene_id, [])
        print(f"\nProcessing scene: {scene_folder} | grid={grid_key} | valid_objects={valid_ids}")

        # Run matching + save crops + collect coordinates
        coords_map = process_scene_image(
            scene_image_path,
            args.num_rows,
            args.num_cols,
            model,
            preprocess,
            template_images,
            template_features,
            scene_folder  # save under the same scene folder
        )

        # Merge this scene’s results into the global JSON summary
        merge_summary(summary_path, scene_id, valid_ids, grid_key, coords_map)

    print(f"\nDone. Global JSON summary written to: {summary_path}")

if __name__ == '__main__':
    main()
