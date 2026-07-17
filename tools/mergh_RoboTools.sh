#!/usr/bin/env bash

set -e

# RoboTools has a single split ("all", 24 scenes). Unlike High_Resolution, merge.py
# discovers scene folders itself (any 000NNN/ under BASE_DIR that has PREDICTION_FILE_NAME),
# so it merges however many scenes are actually present -- no scene-count/range to configure.
BASE_DIR="../Output/RoboTools"
GT_FILE="../eval_results/Ground_Truth/RoboTools/scene_gt_coco_all.json"
PREDICTION_FILE_NAME="pred_results_USE_PE_ADAPTER=True_USE_AUGMENTED_SAM=True.json"
MERGED_OUTPUT="../eval_results/Results_COCO/RoboTools/L2G_all.json"

echo "Merging RoboTools scene predictions..."

python merge.py \
  --gt "$GT_FILE" \
  --root "$BASE_DIR" \
  --out "$MERGED_OUTPUT" \
  --pred-filename "$PREDICTION_FILE_NAME"

echo "Merge completed successfully. Starting evaluation..."

python eval_results.py --dataset RoboTools
