#!/bin/bash
srun -N 1 --container-image=/lustre/fsw/coreai_devtech_all/shifangx/docker/sglang:v0.5.2-cu129-gb200.sqsh \
       	--container-save=/lustre/fsw/coreai_devtech_all/shifangx/docker/my-sglang:v0.5.2-cu129-gb200.sqsh \
	--container-mounts=/lustre:/lustre bash -c "pip install sentencepiece"

