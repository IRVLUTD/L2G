#!/usr/bin/env bash

set -e

# Which High_Resolution split to merge: hard (scenes 1-10), easy (11-22), or all (1-22).
# Defaults to "hard" so existing behavior/output is unchanged unless you opt in.
SPLIT="all"  # hard , easy, all

case "$SPLIT" in
  hard) START_FOLDER=1;  END_FOLDER=10 ;;
  easy) START_FOLDER=11; END_FOLDER=22 ;;
  all)  START_FOLDER=1;  END_FOLDER=22 ;;
  *) echo "Unknown SPLIT='$SPLIT' (expected hard|easy|all)"; exit 1 ;;
esac

BASE_DIR="../Output/High_Resolution"
INPUT_FILE_NAME="coco_instances_results_converted.json"
PREDICTION_FILE_NAME="pred_results_USE_PE_ADAPTER=True_USE_AUGMENTED_SAM=True.json"
MERGED_OUTPUT="../eval_results/Results_COCO/High_Resolution/L2G_${SPLIT}.json"

# hard's 10 scenes x 4 query images each = 40. Each scene folder's query-image index
# restarts at 0 within its own split (hard: 0-39, easy: 0-119), so when merging "all"
# the easy folders must be shifted by +40 to match scene_gt_coco_all.json, which places
# hard at image_id 0-39 and easy at image_id 40-159.
HARD_IMAGE_COUNT=40

for i in $(seq -f "%06g" "$START_FOLDER" "$END_FOLDER"); do
    folder_num=$((10#$i))
    offset=0
    if [ "$SPLIT" = "all" ] && [ "$folder_num" -ge 11 ]; then
        offset=$HARD_IMAGE_COUNT
    fi

    echo "Converting folder: $i (image_id_offset=$offset)"

    python convert_to_coco_results.py \
      --input "${BASE_DIR}/${i}/${PREDICTION_FILE_NAME}" \
      --output "${BASE_DIR}/${i}/${INPUT_FILE_NAME}" \
      --image_id_offset "$offset"
done

echo "Conversion completed successfully. Starting merge..."

python merge_high.py \
  --root_dir "$BASE_DIR" \
  --start_folder "$START_FOLDER" \
  --end_folder "$END_FOLDER" \
  --input_file_name "$INPUT_FILE_NAME" \
  --out_file_name "$MERGED_OUTPUT"

echo "Merge completed successfully. Starting evaluation..."

python eval_results.py --dataset High_Res --split "$SPLIT"
