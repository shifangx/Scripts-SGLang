#!/bin/bash

# 自动提取节点列表
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
mkdir -p ../env

# 设置预填充节点
export PREFILL_NODE_LIST="${NODES[0]},${NODES[1]}"
export PREFILL_HEAD_NODE="${NODES[0]}"

# 设置解码节点
DECODE_NODES=("${NODES[@]:2}")
# DECODE_NODES="${NODES[2]},${NODES[3]}"
export DECODE_NODE_LIST="$(IFS=','; echo "${DECODE_NODES[*]}")"
export DECODE_HEAD_NODE="${NODES[2]}"

export NODE_LIST=$PREFILL_NODE_LIST
export HEAD_NODE=$PREFILL_HEAD_NODE

# 打印环境变量
echo "PREFILL_NODE_LIST=$PREFILL_NODE_LIST"
echo "PREFILL_HEAD_NODE=$PREFILL_HEAD_NODE"
echo "DECODE_NODE_LIST=$DECODE_NODE_LIST"
echo "DECODE_HEAD_NODE=$DECODE_HEAD_NODE"
echo "NODE_LIST=$NODE_LIST"
echo "HEAD_NODE=$HEAD_NODE"

# 检查 NODE_LIST 是否为空
if [ -z "$NODE_LIST" ]; then
    echo "错误: NODE_LIST 为空，无法继续执行"
    exit 1
fi

# 保存到文件
cat <<EOF > ../env/node_list_env.sh
export PREFILL_NODE_LIST="${PREFILL_NODE_LIST}"
export PREFILL_HEAD_NODE="${PREFILL_HEAD_NODE}"
export DECODE_NODE_LIST="${DECODE_NODE_LIST}"
export DECODE_HEAD_NODE="${DECODE_HEAD_NODE}"
export NODE_LIST="${NODE_LIST}"
export HEAD_NODE="${HEAD_NODE}"
EOF
