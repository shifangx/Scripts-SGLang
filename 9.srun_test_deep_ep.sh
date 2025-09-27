#!/bin/bash
srun -N 2 --container-image=/lustre/fsw/coreai_devtech_all/shifangx/docker/debug-sglang:v0.5.2-cu129-gb200.sqsh --container-mounts /lustre:/lustre bash -c "export WORLD_SIZE=2; export RANK=$SLURM_NODEID; python /lustre/fsw/coreai_devtech_all/shifangx/8.SGLang_benchmark/a.source_code/DeepEP/tests/test_low_latency.py"

