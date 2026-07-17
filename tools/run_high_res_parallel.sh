#!/usr/bin/env bash
# Run High_Res scenes in parallel.
#
# run.py already accepts --scene-start/--scene-end/--device and each scene writes to its
# own Output/{DATASET}/{scene_id:06d}/... files, so this script does not change run.py at
# all -- it just launches several `run.py` processes at once, each handling a slice of the
# scene range, and lets tools/mergh_high.sh merge the results afterwards exactly as before.
#
# The core use case is parallelizing *within one GPU*: give one device a scene range and
# it will be split further across WORKERS_PER_DEVICE processes that all share that GPU.
# You can optionally also list more than one device to spread scenes across several GPUs.
# Note: at the default local_view_batch_size=8 (High_Res.yaml), one worker already uses
# ~28GB, so a single 48GB GPU only fits 1 worker via "auto" -- to actually pack multiple
# workers on one GPU, either use a smaller/24GB-class comparison (auto adapts either way)
# or lower local_view_batch_size to shrink each worker's footprint. Listing multiple
# devices in DEVICES is the simplest way to get more parallelism on hardware like this one.

set -u

CONFIG="High_Res.yaml"
START_SCENE=1
END_SCENE=10

# Devices to use. Default is a single GPU -- add more entries to also spread scenes
# across several GPUs, e.g. DEVICES=("cuda:0" "cuda:1" "cuda:2" "cuda:3").
DEVICES=("cuda:0")

# Worker processes launched *per device* (how many run.py's share one GPU at once).
# Set a fixed integer, or "auto" to size it from currently-free GPU memory (works for
# GPUs of any size, e.g. 24GB vs 48GB -- it does not assume a fixed GPU size).
WORKERS_PER_DEVICE="auto"

# Only used when WORKERS_PER_DEVICE="auto": estimated peak GPU memory (GB) needed by one
# run.py worker (model weights + activations for one local_view_batch_size batch of
# crops, see post_processing.local_view_batch_size in the config). Measured ~27.7GB via
# nvidia-smi with the default local_view_batch_size=8 on High_Res -- 30 leaves a safety
# margin. On a 48GB GPU this means "auto" currently picks 1 worker per GPU at batch_size=8
# (2 would need ~56GB and OOM); to fit more workers per GPU, lower
# post_processing.local_view_batch_size in the config (less memory per worker, but also
# less of Tier2's speedup per worker) and lower PER_WORKER_GB to match, or just add more
# devices to DEVICES instead of stacking workers on one GPU.
PER_WORKER_GB=30

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_ROOT/Output/parallel_logs"

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"

workers_for_device() {
    local dev="$1"
    if [ "$WORKERS_PER_DEVICE" = "auto" ]; then
        local free_gb
        free_gb=$(python3 -c "
import torch
free, _ = torch.cuda.mem_get_info('$dev')
print(free / (1024**3))
" 2>/dev/null)
        if [ -z "$free_gb" ]; then
            echo 1
        else
            python3 -c "print(max(1, int($free_gb // $PER_WORKER_GB)))"
        fi
    else
        echo "$WORKERS_PER_DEVICE"
    fi
}

# ---- resolve how many workers each device gets, and build a flat worker->device list ----
WORKER_DEVICES=()
for dev in "${DEVICES[@]}"; do
    n=$(workers_for_device "$dev")
    echo "Device $dev: $n worker(s) (PER_WORKER_GB=$PER_WORKER_GB, WORKERS_PER_DEVICE=$WORKERS_PER_DEVICE)"
    for ((i = 0; i < n; i++)); do
        WORKER_DEVICES+=("$dev")
    done
done

NUM_SCENES=$((END_SCENE - START_SCENE + 1))
NUM_WORKERS=${#WORKER_DEVICES[@]}
if [ "$NUM_WORKERS" -gt "$NUM_SCENES" ]; then
    NUM_WORKERS=$NUM_SCENES
fi
if [ "$NUM_WORKERS" -lt 1 ]; then
    echo "Nothing to run: START_SCENE=$START_SCENE END_SCENE=$END_SCENE"
    exit 1
fi

echo "Running scenes $START_SCENE-$END_SCENE ($CONFIG) across $NUM_WORKERS worker(s)"

# ---- split [START_SCENE, END_SCENE] into NUM_WORKERS contiguous chunks, launch each ----
PIDS=()
DESCS=()
base=$((NUM_SCENES / NUM_WORKERS))
extra=$((NUM_SCENES % NUM_WORKERS))
cur=$START_SCENE
for ((w = 0; w < NUM_WORKERS; w++)); do
    size=$base
    if [ "$w" -lt "$extra" ]; then
        size=$((size + 1))
    fi
    chunk_start=$cur
    chunk_end=$((cur + size - 1))
    cur=$((chunk_end + 1))

    dev="${WORKER_DEVICES[$w]}"
    dev_tag="$(echo "$dev" | tr ':' '_')"
    log_file="$LOG_DIR/scenes_${chunk_start}-${chunk_end}_${dev_tag}.log"
    echo "  worker $w: scenes $chunk_start-$chunk_end on $dev -> $log_file"

    python run.py --config "$CONFIG" --scene-start "$chunk_start" --scene-end "$chunk_end" --device "$dev" \
        > "$log_file" 2>&1 &
    PIDS+=($!)
    DESCS+=("scenes $chunk_start-$chunk_end on $dev")
done

# ---- wait for all workers, report failures instead of aborting on the first one ----
FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "FAILED: ${DESCS[$i]} -- see log in $LOG_DIR"
        FAILED=1
    fi
done

if [ "$FAILED" -eq 1 ]; then
    echo "One or more workers failed. Check logs above, re-run just those scene ranges before merging."
    exit 1
fi

echo "All workers finished successfully. Logs in $LOG_DIR"
echo "Next: cd tools && bash mergh_high.sh"
