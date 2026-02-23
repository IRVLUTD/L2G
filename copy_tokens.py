import os
import shutil
from pathlib import Path

# Source and target folders (you can modify these paths)
SRC_DIR = Path("ckpts_full_mask_token_high_1113")
DST_DIR = Path("ckpts_full_mask_token_high_best_final")

# Create target folder if it does not exist
DST_DIR.mkdir(parents=True, exist_ok=True)

for i in range(1, 101):  # 1 to 100
    file_name = f"full_mask_tokens_{i:06d}.pt"
    src_path = SRC_DIR / file_name
    dst_path = DST_DIR / file_name

    # Only copy if source file exists
    if src_path.is_file():
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {src_path} -> {dst_path}")
    else:
        print(f"Skip (not found): {src_path}")
