#!/usr/bin/env bash
# Run RoboTools or High_Res scenes in parallel across one or more GPUs.
#
# run.py already accepts --scene-start/--scene-end/--config/--device and each scene writes
# to its own Output/{DATASET}/{scene_id:06d}/... files, so this script does not change
# run.py at all -- it just launches several `run.py` processes at once, each handling a
# slice of the scene range, and lets tools/mergh_high.sh / tools/mergh_RoboTools.sh merge
# the results afterwards exactly as before.
#
# The core use case is parallelizing *within one GPU*: give one device a scene range and
# it will be split further across WORKERS_PER_DEVICE processes that all share that GPU.
# You can also list more than one device to spread scenes across several GPUs.
#
# Usage:
#   ./run_parallel.sh --dataset RoboTools [options]
#   ./run_parallel.sh --dataset High-Res --split easy [options]
#
# Options:
#   --dataset NAME            RoboTools | High-Res (required)
#   --split hard|easy|all     High-Res only: hard=1-10, easy=11-22, all=1-22 (default: all)
#   --scene-start N           Override the default scene range start
#   --scene-end N             Override the default scene range end
#   --config FILE             Override the config file under L2G_configs/ (default derived
#                              from --dataset: RoboTools.yaml / High_Res.yaml)
#   --devices LIST            Comma-separated devices, e.g. cuda:0,cuda:1,cuda:2,cuda:3
#                              (default: cuda:0)
#   --workers-per-device N|auto
#                              Worker processes launched *per device*. "auto" sizes it from
#                              currently-free GPU memory (default: auto)
#   --per-worker-gb N         Only used when --workers-per-device=auto: estimated peak GPU
#                              memory (GB) needed by one run.py worker. Measured ~27.7GB via
#                              nvidia-smi for High_Res at local_view_batch_size=8; 30 leaves a
#                              safety margin. RoboTools' footprint hasn't been measured the
#                              same way -- if "auto" picks a bad number for it, pass this
#                              explicitly. (default: 30)
#   -h, --help                Show this help and exit
#
# Note: on a 48GB GPU, "auto" currently picks 1 worker per GPU for High_Res at
# local_view_batch_size=8 (2 would need ~56GB and OOM). To fit more workers per GPU, lower
# post_processing.local_view_batch_size in the config (less memory per worker, but also less
# of Tier2's speedup per worker) and lower --per-worker-gb to match, or just list more
# devices instead of stacking workers on one GPU.

set -u

print_usage() {
    sed -n '2,35p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

DATASET=""
SPLIT="all"
SCENE_START=""
SCENE_END=""
CONFIG=""
DEVICES_ARG="cuda:0"
WORKERS_PER_DEVICE="auto"
PER_WORKER_GB=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --scene-start) SCENE_START="$2"; shift 2 ;;
        --scene-end) SCENE_END="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --devices) DEVICES_ARG="$2"; shift 2 ;;
        --workers-per-device) WORKERS_PER_DEVICE="$2"; shift 2 ;;
        --per-worker-gb) PER_WORKER_GB="$2"; shift 2 ;;
        -h|--help) print_usage; exit 0 ;;
        *) echo "Unknown argument: $1"; echo; print_usage; exit 1 ;;
    esac
done

# ---- resolve dataset -> config file, default scene range, and merge-step hint ----
case "$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')" in
    robotools)
        CONFIG="${CONFIG:-RoboTools.yaml}"
        DEFAULT_START=1
        DEFAULT_END=24
        MERGE_HINT="cd tools && bash mergh_RoboTools.sh"
        ;;
    high-res|high_res|highres)
        CONFIG="${CONFIG:-High_Res.yaml}"
        case "$SPLIT" in
            hard) DEFAULT_START=1;  DEFAULT_END=10 ;;
            easy) DEFAULT_START=11; DEFAULT_END=22 ;;
            all)  DEFAULT_START=1;  DEFAULT_END=22 ;;
            *) echo "Unknown --split='$SPLIT' (expected hard|easy|all)"; exit 1 ;;
        esac
        MERGE_HINT="cd tools && bash mergh_high.sh   # set SPLIT=\"$SPLIT\" at the top first"
        ;;
    "")
        echo "Missing required --dataset (RoboTools | High-Res)"; echo; print_usage; exit 1 ;;
    *)
        echo "Unknown --dataset='$DATASET' (expected RoboTools | High-Res)"; exit 1 ;;
esac

START_SCENE="${SCENE_START:-$DEFAULT_START}"
END_SCENE="${SCENE_END:-$DEFAULT_END}"

IFS=',' read -r -a DEVICES <<< "$DEVICES_ARG"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
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

echo "Running $DATASET scenes $START_SCENE-$END_SCENE ($CONFIG) across $NUM_WORKERS worker(s)"

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
    log_file="$LOG_DIR/${DATASET}_scenes_${chunk_start}-${chunk_end}_${dev_tag}.log"
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
echo "Next: $MERGE_HINT"
