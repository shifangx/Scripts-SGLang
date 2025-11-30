#!/bin/bash
PARTITION=${PARTITION:-"batch"}
export WORLD_SIZE=14
salloc -A ${SLURM_ACCOUNT} -J ${JOB_NAME}  --gpus-per-node=4 -p ${PARTITION} -N ${WORLD_SIZE} --segment=${WORLD_SIZE} --exclusive --mem 0 --time 4:00:00
