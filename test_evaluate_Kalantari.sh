#!/usr/bin/env bash
# FreeMEF 一键测试 + 评估脚本（Kalantari 数据集）
# One-click inference + evaluation on the Kalantari dataset
set -e

# 切换到脚本所在目录（仓库根目录），并将其加入 PYTHONPATH
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# ---- 可配置参数（支持环境变量覆盖）----
WEIGHTS=${WEIGHTS:-"weights/FreeMEF.pth"}                       # 模型权重
OPT=${OPT:-"options/FreeMEF.yml"}                              # 配置文件
ARCH=${ARCH:-"FreeMEF"}                                       # 网络名称
INPUT_DIR=${INPUT_DIR:-"datasets/Kalantari_MEF/Testing_input"} # 测试输入序列
GT_DIR=${GT_DIR:-"datasets/Kalantari_MEF/Testing_gt"}          # 测试 GT
RESULT_DIR=${RESULT_DIR:-"result/Kalantari"}                  # 输出目录

# 1) 推理 / Inference
echo "==> [1/2] Running inference -> ${RESULT_DIR}"
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
