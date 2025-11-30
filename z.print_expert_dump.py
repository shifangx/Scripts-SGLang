#!/usr/bin/env python3
import os
import sys
import glob
from pathlib import Path

import torch


def find_latest_pt(dir_path: str) -> str:
    files = sorted(glob.glob(os.path.join(dir_path, "expert_distribution_recorder_*.pt")))
    if not files:
        raise FileNotFoundError(f"No dump files found in {dir_path}")
    return files[-1]


def summarize_stat(obj):
    rank = obj.get("rank")
    lc = obj.get("logical_count")
    avg = obj.get("average_utilization_rate_over_window")
    if not isinstance(lc, torch.Tensor):
        print("logical_count missing or not a tensor")
        return
    print(f"[stat] rank={rank}")
    print(f"logical_count shape: {tuple(lc.shape)}  (buffer_len, num_layers, num_logical_experts)")
    print(f"average_utilization_rate_over_window: {avg}")
    # Show a brief per-layer top experts summary for the first buffer element
    buf0 = lc[0].cpu()
    topk = 5
    per_layer_top = []
    for layer in range(buf0.shape[0]):
        vals, idx = torch.topk(buf0[layer], k=min(topk, buf0.shape[1]))
        per_layer_top.append((layer, [(int(i), int(v)) for v, i in zip(vals, idx)]))
    print("first-buffer per-layer top experts (expert_index, count):")
    for layer, pairs in per_layer_top:
        print(f"  layer {layer}: {pairs}")


def summarize_detail(obj):
    records = obj.get("records", [])
    p2l = obj.get("last_physical_to_logical_map")
    print(f"[detail] num_records={len(records)}")
    if isinstance(p2l, torch.Tensor):
        print(f"last_physical_to_logical_map shape: {tuple(p2l.shape)}")
    if not records:
        return
    r0 = records[0]
    keys = list(r0.keys())
    print(f"first record keys: {keys}")
    for k in ("forward_pass_id", "rank", "gatherer_key"):
        if k in r0:
            print(f"{k}: {r0[k]}")
    if isinstance(r0.get("global_physical_count"), torch.Tensor):
        g = r0["global_physical_count"]
        print(f"global_physical_count shape: {tuple(g.shape)} (num_layers, num_physical_experts)")
    if isinstance(r0.get("topk_ids_of_layer"), torch.Tensor):
        t = r0["topk_ids_of_layer"]
        print(f"topk_ids_of_layer shape: {tuple(t.shape)} (num_layers, num_tokens, topk)")


def main():
    dump_dir = os.environ.get("SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR", "../decode_sglang_expert_distribution_recorder")
    path = sys.argv[1] if len(sys.argv) > 1 else find_latest_pt(dump_dir)
    print(f"Loading: {path}")
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "logical_count" in obj:
        summarize_stat(obj)
    elif isinstance(obj, dict) and "records" in obj:
        summarize_detail(obj)
    else:
        print("Unknown dump format; keys:", obj.keys() if isinstance(obj, dict) else type(obj))


if __name__ == "__main__":
    main()


