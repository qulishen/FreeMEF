#!/usr/bin/env bash
# FreeMEF 一键训练脚本 / One-click training launcher
set -e

# 切换到脚本所在目录（仓库根目录），并将其加入 PYTHONPATH
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# ---- 可配置参数（支持环境变量覆盖）----
# 示例： GPUS=0,1,2,3 PORT=4321 bash train.sh
GPUS=${GPUS:-"0,1"}                              # 使用的 GPU 序号
NPROC=${NPROC:-$(awk -F',' '{print NF}' <<< "$GPUS")}  # 进程数（默认 = GPU 数量）
PORT=${PORT:-4321}                              # 分布式通信端口
OPT=${OPT:-"options/FreeMEF.yml"}               # 训练配置文件

echo "==> GPUs=${GPUS} | nproc_per_node=${NPROC} | master_port=${PORT} | opt=${OPT}"

CUDA_VISIBLE_DEVICES=${GPUS} torchrun \
  --nproc_per_node=${NPROC} \
  --master_port=${PORT} \
  FreeMEF/train.py \
  -opt "${OPT}" \
  --launcher pytorch \
  --auto_resume
