import torch
from PIL import Image
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

print("CLIP configs:", pe.CLIP.available_configs())


model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)
model = model.cuda()


preprocess = transforms.get_image_transform(model.image_size)


template_img = Image.open("000001.png")
template_tensor = preprocess(template_img).unsqueeze(0).cuda()

with torch.no_grad():
    template_features = model.encode_image(template_tensor)
    template_features /= template_features.norm(dim=-1, keepdim=True)
    print("template_features.shape:", template_features.shape)


test_img = Image.open("00002.jpg")


W, H = test_img.size
num_rows, num_cols = 3, 3
patch_w, patch_h = W // num_cols, H // num_rows

patch_tensors = []
original_patches = []

for row in range(num_rows):
    for col in range(num_cols):
        left = col * patch_w
        upper = row * patch_h
        right = left + patch_w
        lower = upper + patch_h

        patch = test_img.crop((left, upper, right, lower))

        original_patches.append(patch)

        patch_tensor = preprocess(patch).unsqueeze(0)  # Shape [1, 3, H, W]
        patch_tensors.append(patch_tensor)

#  → [16, 3, H, W]
all_patches_batch = torch.cat(patch_tensors, dim=0).cuda()

print("all_patches_batch.shape:", all_patches_batch.shape)  # Expect [16, 3, H, W]


with torch.no_grad():
    patch_features = model.encode_image(all_patches_batch)
    print(model.encode_image)
    patch_features /= patch_features.norm(dim=-1, keepdim=True)
    print("patch_features.shape:", patch_features.shape)  # [16, feature_dim]


    logits = (100.0 * template_features @ patch_features.T).softmax(dim=-1).cpu().numpy()[0]
    logits_value = (100.0 * template_features @ patch_features.T)
    print("Patch similarities (probs):", logits)
    print("logits_value:", logits_value)

   
    for idx, prob in enumerate(logits):
        row_idx = idx // num_cols
        col_idx = idx % num_cols
        print(f"Patch {idx} (Row {row_idx}, Col {col_idx}): Probability = {prob:.4f}")


import numpy as np
import matplotlib.pyplot as plt

# Top-3 
top3_indices = np.argsort(logits)[-3:][::-1]
print("\nTop 3 highest-probability patches:")
for rank, idx in enumerate(top3_indices):
    row_idx = idx // num_cols
    col_idx = idx % num_cols
    print(f"Rank {rank+1}: Patch {idx} (Row {row_idx}, Col {col_idx}) with Probability = {logits[idx]:.4f}")

# Show Top-3 patches
plt.figure(figsize=(12, 4))
for i, idx in enumerate(top3_indices):
    plt.subplot(1, 3, i+1)
    plt.imshow(original_patches[idx])
    plt.title(f"Patch {idx}\nProb: {logits[idx]:.4f}")
    plt.axis('off')

    save_path = f"000001_object_33/top{rank+1}_patch_{idx}.jpg"
    original_patches[idx].save(save_path, format='JPEG')
    print(f"Saved: {save_path}")

plt.tight_layout()
plt.show()