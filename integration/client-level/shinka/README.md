# Client-Level Shinka Workflow

This directory adds a Shinka-based heuristic search path for the client-level Heimdall experiment.

The workflow is:

1. Replay traces with `baseline` so we have raw latency traces.
2. Build Shinka-ready labeled datasets from those baseline traces.
3. Run Shinka evolution on the per-device datasets.
4. Export the best `predict()` heuristics into C headers.
5. Replay the client-level experiment with the exported Shinka heuristic.

## Prerequisite

Install the bundled `ShinkaEvolve` package first:

```bash
cd $HEIMDALL/ShinkaEvolve
python3 -m pip install -e .
```

## End-to-end command

From [`integration/client-level/experiment`](/Users/diego/SSD-Heuristics/integration/client-level/experiment):

```bash
sudo -E python ./run_shinka.py \
  -devices /dev/nvme0n1 /dev/nvme1n1 \
  -trace_dirs $HEIMDALL/integration/client-level/data/*/*/*
```

Add `-resume` to skip traces whose evolution already completed and only run the remaining ones:

```bash
sudo -E python ./run_shinka.py \
  -devices /dev/nvme0n1 /dev/nvme1n1 \
  -resume \
  -trace_dirs $HEIMDALL/integration/client-level/data/*/*/*
```

`run_shinka.py` will:

- generate datasets under `.../<dev0>...<dev1>/shinka/training_results/`
- run Shinka once per original device; device 1 is **seeded with the best heuristic from device 0** so evolution starts from a stronger baseline rather than `initial.py`
- export `best/main.py` into replay-time headers
- compile the `experiment/shinka` replayer in a temporary workspace
- replay the traces with the exported heuristic

## Manual pieces

Prepare datasets only:

```bash
python integration/client-level/shinka/prepare_dataset.py \
  -devices /dev/nvme0n1 /dev/nvme1n1 \
  -trace_dir /path/to/trace_bundle
```

Run evolution on one dataset:

```bash
python integration/client-level/shinka/run_evo.py \
  --dataset_path /path/to/dataset_device_0.csv \
  --results_dir /path/to/evolution_device_0
```

To seed evolution from an existing heuristic (e.g. transfer device 0's best program to device 1):

```bash
python integration/client-level/shinka/run_evo.py \
  --dataset_path /path/to/dataset_device_1.csv \
  --results_dir /path/to/evolution_device_1 \
  --init_program_path /path/to/evolution_device_0/best/main.py
```

Export the best program to C:

```bash
python integration/client-level/shinka/export_heuristic.py \
  --program_path /path/to/evolution_device_0/best/main.py \
  --output_header /path/to/shinka_dev_0.h \
  --function_name shinka_predict_dev_0
```

If Shinka cannot infer an LLM configuration automatically, set `SHINKA_LLM_MODELS` to a comma-separated model list.
