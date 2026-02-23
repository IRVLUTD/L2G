#!/usr/bin/env bash

set -e

for i in $(seq -f "%06g" 31 40); do
    python convert_to_coco_results.py \
      --input ${i}/pred_results_0120_sam_FPS_USE_ADAPTER=False_USE_PE_ADAPTER=False_USE_WHICH_FEATURE=pe_feature_00.json \
      --output ${i}/coco_instances_results_converted_0120_00.json
done
