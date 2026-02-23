import argparse
import os
import torch
from PIL import Image
import numpy as np
from pathlib import Path

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

def load_model():
    print("Loading model...")
    print("CLIP configs:", pe.CLIP.available_configs())
    model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True).cuda()
    preprocess = transforms.get_image_transform(model.image_size)
    return model, preprocess

def load_templates(template_dir, preprocess):
    template_features = []
    template_filenames = sorted(os.listdir(template_dir))
    template_images = []
    
    for fname in template_filenames:
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img = Image.open(os.path.join(template_dir, fname)).convert('RGB')
        template_images.append((fname, img))

    print(f"Loaded {len(template_images)} template images.")
    return template_images

def compute_features(model, images, preprocess):
    tensors = []
    for _, img in images:
        tensors.append(preprocess(img).unsqueeze(0))
    batch = torch.cat(tensors, dim=0).cuda()
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

def process_scene_image(scene_image_path, num_rows, num_cols, model, preprocess, template_images, template_features, output_dir):
    scene_img = Image.open(scene_image_path).convert('RGB')
    W, H = scene_img.size
    patch_w, patch_h = W // num_cols, H // num_rows

    # Split scene into patches
    patches = []
    patch_tensors = []
    for row in range(num_rows):
        for col in range(num_cols):
            left = col * patch_w
            upper = row * patch_h
            right = left + patch_w
            lower = upper + patch_h
            patch = scene_img.crop((left, upper, right, lower))
            patches.append(patch)
            patch_tensor = preprocess(patch).unsqueeze(0)
            patch_tensors.append(patch_tensor)

    all_patches_batch = torch.cat(patch_tensors, dim=0).cuda()

    with torch.no_grad():
        patch_features = model.encode_image(all_patches_batch)
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)

    # For each template: find best matching patch
    for idx, (template_name, _) in enumerate(template_images):
        template_feat = template_features[idx:idx+1]
        logits = (100.0 * template_feat @ patch_features.T).softmax(dim=-1).cpu().numpy()[0]
        
        best_idx = np.argmax(logits)
        best_patch = patches[best_idx]

        # Extract template ID (e.g. 000001 from 000001.jpg)
        template_base = Path(template_name).stem

        # Create target folder inside scene folder
        save_folder = output_dir / f"{num_rows}_{num_cols}_{template_base}"
        save_folder.mkdir(parents=True, exist_ok=True)

        # Save patch as 000001.jpg (converted to JPEG)
        save_path = save_folder / f"000001.jpg"
        best_patch.convert('RGB').save(save_path, format='JPEG')
        print(f"Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_rows', type=int, required=True, help='Number of rows to split')
    parser.add_argument('--num_cols', type=int, required=True, help='Number of columns to split')
    parser.add_argument('--scene_dir', type=str, default='../RoboTools_init_frames/test', help='Directory with scene folders')
    parser.add_argument('--template_dir', type=str, default='../RoboTools_init_frames/template_images', help='Directory with template images')
    args = parser.parse_args()

    # Load model
    model, preprocess = load_model()

    # Load template images
    template_images = load_templates(args.template_dir, preprocess)
    template_features = compute_features(model, template_images, preprocess)

    # Process all scenes
    scene_root = Path(args.scene_dir)
    for scene_folder in sorted(scene_root.iterdir()):
        if not scene_folder.is_dir():
            continue
        scene_image_path = scene_folder / '00000.png'
        if not scene_image_path.exists():
            print(f"Missing image in {scene_folder}, skipping.")
            continue

        print(f"\nProcessing scene: {scene_folder}")
        process_scene_image(
            scene_image_path,
            args.num_rows,
            args.num_cols,
            model,
            preprocess,
            template_images,
            template_features,
            scene_folder
        )

if __name__ == '__main__':
    main()
