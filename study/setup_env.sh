#!/usr/bin/env bash
# 准备运行判题与提交流水线所需的环境。可以放进 Claude Code 云环境的 setup script，也可以本机手动运行。
#   bash study/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/3] 安装 Python 依赖（numpy、pandas、PyTorch）"
# 注意：PyPI 上 Linux 的 torch 默认是带 CUDA 的大包（数 GB）。云环境的磁盘额度有限，
# 而且默认网络策略不放行 download.pytorch.org，所以这里直接用 PyPI。
# 下载可能超过几分钟：在会话里运行时请放到后台。
python -m pip install --quiet --no-input numpy pandas torch

echo "[2/3] 检查沙箱能力"
if unshare -rn true 2>/dev/null; then
  echo "  网络隔离可用（unshare -rn）"
else
  echo "  警告：unshare -rn 不可用，提交代码将在没有网络隔离的情况下运行"
fi

echo "[3/3] 流水线自检"
python study/selftest.py
