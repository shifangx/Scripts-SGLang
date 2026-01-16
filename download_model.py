from huggingface_hub import snapshot_download
import os

# Optional: supply a HF token to avoid anonymous rate limits
hf_token = os.getenv("HF_TOKEN") or None

snapshot_download(
    repo_id="nvidia/DeepSeek-R1-0528-FP4",      # 模型仓库名
    local_dir="../models/DeepSeek-R1-0528-FP4",  # 本地保存目录
    local_dir_use_symlinks=False,               # 关闭软链接，直接真实文件
    revision="main",                            # 分支/版本号，可选
    token=hf_token,                             # 使用令牌（若提供）以降低429概率
    resume_download=True,                       # 中断后可继续
)
