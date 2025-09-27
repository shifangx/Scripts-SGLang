#!/bin/bash

export HEAD_NODE=ptyche0052
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
export MC_TE_METRIC=true
export SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
export NCCL_MNNVL_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export PYTHONUNBUFFERED=1

/usr/local/bin/nsys launch --cuda-graph-trace=node --trace cuda,nvtx,mpi \
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 \
	                        --trust-remote-code \
				--dist-init-addr $HEAD_NODE:8000 \
				--nnodes 2 \
				--node-rank $SLURM_NODEID \
				--tp-size 8 \
				--dp-size 8 \
				--enable-dp-attention \
				--host 0.0.0.0 \
				--decode-log-interval 1 \
				--max-running-requests 6144 \
				--disable-radix-cache \
				--moe-dense-tp-size 1 \
				--enable-dp-lm-head \
				--disable-shared-experts-fusion \
				--eplb-algorithm deepseek \
				--attention-backend cutlass_mla \
				--watchdog-timeout 1000000 \
				--disable-cuda-graph \
				--chunked-prefill-size 8192 \
				--max-total-tokens 8192 \
				--moe-a2a-backend deepep \
				--deepep-mode low_latency \
				--ep-dispatch-algorithm dynamic \
				--load-format dummy
