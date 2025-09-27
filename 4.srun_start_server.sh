#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
srun --ntasks=2 --nodes=2 --container-image=/lustre/fsw/coreai_devtech_all/shifangx/docker/my-sglang:v0.5.2-cu129-gb200.sqsh --container-mounts /lustre:/lustre bash -c "bash /lustre/fsw/coreai_devtech_all/shifangx/8.SGLang_benchmark/launch_server.sh"

