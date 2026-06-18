#!/usr/bin/env bash
# FreeMEF 一键测试 + 评估脚本（SICE 数据集，含 2/3/5 帧三种输入）
# One-click inference + evaluation on SICE (2-/3-/5-frame inputs, shared GT)
set -e

# 切换到脚本所在目录（仓库根目录），并将其加入 PYTHONPATH
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# ---- 可配置参数（支持环境变量覆盖）----
WEIGHTS=${WEIGHTS:-"weights/FreeMEF.pth"}      # 模型权重
OPT=${OPT:-"options/FreeMEF.yml"}             # 配置文件
ARCH=${ARCH:-"FreeMEF"}                       # 网络名称
DATA_ROOT=${DATA_ROOT:-"datasets/SICE"}       # SICE 数据集根目录
GT_DIR=${GT_DIR:-"${DATA_ROOT}/gt"}           # 三种输入共用的 GT
RESULT_ROOT=${RESULT_ROOT:-"result"}          # 输出根目录
FRAMES=${FRAMES:-"2 3 5"}                     # 待测试的帧数（可改为 "2 3" 等）

for n in ${FRAMES}; do
  INPUT_DIR="${DATA_ROOT}/input_${n}frame"
  RESULT_DIR="${RESULT_ROOT}/SICE_${n}frame"

  if [ ! -d "${INPUT_DIR}" ]; then
    echo "==> [skip] 输入目录不存在：${INPUT_DIR}"
    continue
  fi

  echo "========================================================"
  echo "==> SICE ${n}-frame | input=${INPUT_DIR} -> result=${RESULT_DIR}"
  echo "========================================================"

  # 1) 推理 / Inference
  echo "==> [1/2] Running inference (${n}-frame)"
  python test.py \
    --weights "${WEIGHTS}" \
    --opt "${OPT}" \
    --arch "${ARCH}" \
    --input_dir "${INPUT_DIR}" \
    --result_dir "${RESULT_DIR}" \
    --tile 0 --tile_stride 0

  # 2) 评估 / Evaluation (PSNR / SSIM / LPIPS)
  echo "==> [2/2] Evaluating ${RESULT_DIR} against ${GT_DIR}"
  python eval.py \
    --pred_root "${RESULT_DIR}" \
    --gt_root "${GT_DIR}" \
    --device auto
done

echo "==> All done. 各帧数指标已分别保存到 metric_result/*.txt"
