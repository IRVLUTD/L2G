#!/usr/bin/env bash

set -e

python utils/merge.py --gt ../data/Ground_Truth/RoboTools/scene_gt_coco_all.json \
  --root ../Output/RoboTools \
  --out  ../Output/RoboTools/merged_coco.json \
  --pred-filename pred_results_0216_USE_PE_ADAPTER=True_USE_AUGMENTED_SAM=True_test_v2.json \
  --scenes 1



