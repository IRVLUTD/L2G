# Running the Benchmark Pipeline

Full examples for running `run.py` over a scene range, on a single GPU or split across
several GPUs, before merging and evaluating with the `tools/mergh_*.sh` scripts (see the
[Benchmark section of the README](README.md#benchmark)).

Scene ranges:
- **RoboTools**: 24 scenes total (1-24), evaluated as a single "all" split.
- **High_Resolution**: 22 scenes total — hard = 1-10, easy = 11-22, all = 1-22.

## Single GPU

```sh
# RoboTools, all 24 scenes
python run.py --config RoboTools.yaml --scene-start 1 --scene-end 24 --device cuda:0

# High_Resolution, hard split
python run.py --config High_Res.yaml --scene-start 1 --scene-end 10 --device cuda:0

# High_Resolution, easy split
python run.py --config High_Res.yaml --scene-start 11 --scene-end 22 --device cuda:0

# High_Resolution, all 22 scenes
python run.py --config High_Res.yaml --scene-start 1 --scene-end 22 --device cuda:0
```

## Multiple GPUs in parallel

Each scene writes to its own `Output/{Dataset}/{scene_id:06d}/...` files independently, so
scenes are embarrassingly parallel — split the range across GPUs by hand, backgrounding one
`run.py` per chunk and `wait`-ing for all of them:

```sh
mkdir -p Output/parallel_logs

python run.py --config RoboTools.yaml --scene-start 1  --scene-end 12 --device cuda:0 \
  > Output/parallel_logs/RoboTools_1-12_cuda0.log  2>&1 &
python run.py --config RoboTools.yaml --scene-start 13 --scene-end 24 --device cuda:1 \
  > Output/parallel_logs/RoboTools_13-24_cuda1.log 2>&1 &

wait   # blocks until both finish; check `echo $?` per job or the logs for failures
```

The same pattern works for `High_Res.yaml` — just point `--scene-start`/`--scene-end` at
the split you want (e.g. split the easy range 11-22 into two 6-scene chunks: 11-16 on
`cuda:0`, 17-22 on `cuda:1`). It also generalizes to more GPUs — add one more backgrounded
`run.py` call per device with its own scene chunk.

Monitor while it runs:
```sh
tail -f Output/parallel_logs/*.log   # note: Python buffers stdout when it's not a TTY,
                                      # so output may lag; `python -u run.py ...` disables
                                      # buffering if you need it live
nvidia-smi                           # GPU-Util / memory per device
ps aux | grep run.py                 # confirm the processes are still alive
```

### Multiple workers per GPU

You can also run more than one worker on the *same* GPU by giving each a smaller scene
chunk and the same `--device`, as long as the GPU has enough free memory. At
`High_Res.yaml`'s default `post_processing.local_view_batch_size: 8`, one worker uses
roughly 28GB, so a 48GB GPU only fits one worker at that setting; to pack more onto one
GPU, lower `local_view_batch_size` in the config to shrink each worker's footprint (at
some cost to per-worker throughput) before stacking workers on a single device.

### Stopping a run

```sh
ps aux | grep run.py        # find the PIDs
kill -TERM <pid> [<pid> ...]  # graceful; add -KILL if a process ignores it
```
Partial per-scene output already written under `Output/` is left in place — rerunning the
same `--scene-start`/`--scene-end` overwrites it.
