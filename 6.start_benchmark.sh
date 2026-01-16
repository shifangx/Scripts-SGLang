#!/bin/bash

echo "start benchmark in docker"
export ROUTER_NODE=$PREFILL_HEAD_NODE
echo "make sure you have launched router on the ROUTER_NODE($ROUTER_NODE)"

# 定义数据集路径
DATASET_DIR="${WORKSPACE}/../datasets"
DATASET_FILE="${DATASET_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"

# 检查文件是否存在
if [[ ! -f "$DATASET_FILE" ]]; then
echo "Dataset file not found at $DATASET_FILE, downloading..."
mkdir -p "$DATASET_DIR"
wget -O "$DATASET_FILE" \
	"https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
echo "Dataset downloaded successfully to $DATASET_FILE"
else
echo "Using existing dataset file at $DATASET_FILE"
fi

python3 -m sglang.bench_one_batch_server  \
--dataset-path "$DATASET_FILE" \
--model-path nvidia/DeepSeek-R1-0528-FP4 \
--base-url http://$ROUTER_NODE:7000  \
--batch-size 135168  \
--input-len 1000 \
--output-len 1000 \
--skip-warmup

# --batch-size 73728   \ # 270336, 135168, 36864, 73728, 147456, 294912
