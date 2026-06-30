<p align="center">
  <img src="logo.png" alt="FreeMEF Logo" width="500"/>
</p>

<h1 align="center">🎬 There and Back Again: A Flexible-Frame Transformer for Multi-Exposure Fusion 🚀</h1>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/Language-English-blue" alt="English"></a>
  <a href="README_cn.md"><img src="https://img.shields.io/badge/语言-中文-red" alt="中文"></a>
  <a href="https://arxiv.org/abs/2606.27905"><img src="https://img.shields.io/badge/arXiv-2606.27905-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
</p>

---

## 🎯 概述

|                 |                                                                                                 |
| --------------- | ----------------------------------------------------------------------------------------------- |
| 📄 **论文**     | There and Back Again: A Flexible-Frame Transformer for Multi-Exposure Fusion 🚀                 |
| 💡 **核心思想** | 循环状态空间模块 (RSSM) 可聚合任意数量的曝光帧；全局特征引导模块 (GFGB) 稳定过曝/欠曝区域       |
| ✨ **优势**     | 可使用任意帧数/顺序进行训练和推理，无需重新训练或修改架构；在各种基准测试和设备上达到 SOTA 效果 |

## 🛠️ 环境配置

> ✅ 已在 Ubuntu + A100-80GB、Python 3.10、PyTorch 2.7.1 (CUDA 12.6) 环境实测通过。

### 1️⃣ 创建环境

```bash
# 创建并激活一个干净的 Python 3.10 环境
conda create -n freemef python=3.10 -y
conda activate freemef

# 进入仓库目录，并加入 PYTHONPATH（分布式训练必需）
cd FreeMEF
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### 2️⃣ 安装 PyTorch（按本机 CUDA 选择）

```bash
# 示例：CUDA 12.6 构建，请按本机 CUDA 修改 index-url
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126
```

### 3️⃣ 安装 `mamba_ssm`（需匹配 torch/CUDA）

```bash
# 方式 A：从 PyPI 安装（会现场编译 CUDA 算子，较慢）
pip install mamba_ssm==2.2.6.post3

# 方式 B（推荐）：安装与 torch/CUDA/python 匹配的预编译 wheel
# 官方发布页下载：https://github.com/state-spaces/mamba/releases
pip install mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

### 4️⃣ 安装其余依赖

```bash
pip install -r requirements.txt
```

> 💡 无显示的服务器请保留 `opencv-python-headless`（`requirements.txt` 已固定），不要安装 `opencv-python`。

## 📂 数据集

> 🔗 数据集下载链接：[Kaggle](https://www.kaggle.com/datasets/lishenqu/freemef/data)

将数据集放置在 `datasets/` 目录下，按序列结构组织：

```
datasets/
├── SICE/
│   ├── input_2frame/<seq_name>/*.png|jpg    # 🖼️ 2帧输入
│   ├── input_3frame/<seq_name>/*.png|jpg    # 🖼️ 3帧输入
│   ├── input_5frame/<seq_name>/*.png|jpg    # 🖼️ 5帧输入
│   └── gt_resize/<seq_name>/*.png|jpg       # 🎯 真实标签（如有）
└── Kalantari_MEF/
    ├── Testing_input/<seq_name>/*.png|jpg   # 🧪 测试输入
    ├── Testing_gt/*.png|jpg                 # 🎯 测试真实标签
    ├── Training_input/...                   # 🏋️ 训练输入
    └── Training_gt/*.png|jpg                # 🎯 训练真实标签
```

📌 **说明：**

- 每个子文件夹代表一个序列；该文件夹中的所有帧将被融合成单一结果
- 分辨率不匹配时会在推理过程中自动对齐到主帧

## 🏋️ 训练

**一键启动（推荐）：**

```bash
bash train.sh
```

通过环境变量覆盖 GPU / 端口 / 配置：

```bash
# 使用 4 张 GPU，自定义端口和配置文件
GPUS=0,1,2,3 PORT=4321 OPT=options/FreeMEF.yml bash train.sh
```

**手动启动（等价命令）：**

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=4321 \
  FreeMEF/train.py \
  -opt options/FreeMEF.yml \
  --launcher pytorch --auto_resume
```

💡 **提示：**

- `train.sh` 会自动设置 `PYTHONPATH`，并根据 `GPUS` 推断 `nproc_per_node`
- `--auto_resume` 会自动从最新检查点续训
- 数据路径 / 迭代次数 / batch size 在 `options/FreeMEF.yml` 中修改

## ⚡ 推理

> 🔗 模型权重下载链接：[Google Drive](https://drive.google.com/file/d/1Tg0xX2yUhtSUyt0rzfOHi1NekWiBP9zA/view?usp=sharing)

**一键测试 + 评估（每个脚本先 `test.py` 后 `eval.py`）：**

```bash
# Kalantari
bash test_evaluate_Kalantari.sh

# SICE —— 自动依次测试 2/3/5 帧输入，并分别与同一份 GT 做评估
bash SICE_test_evaluate.sh
```

可通过环境变量覆盖路径，例如：

```bash
WEIGHTS=weights/FreeMEF.pth RESULT_DIR=result/Kalantari bash test_evaluate_Kalantari.sh

# 只测试 SICE 的部分帧数
FRAMES="2 3" DATA_ROOT=datasets/SICE bash SICE_test_evaluate.sh
```

### 📋 关键参数

| 参数                      | 描述                                          |
| ------------------------- | --------------------------------------------- |
| `--weights`               | 📦 检查点路径                                 |
| `--opt`                   | ⚙️ 推理配置文件（默认 `options/FreeMEF.yml`） |
| `--arch`                  | 🏗️ 网络名称（默认 `FreeMEF`）                 |
| `--input_dir`             | 📁 输入序列根目录                             |
| `--result_dir`            | 📤 输出目录                                   |
| `--tile`, `--tile_stride` | 🧩 启用分块推理以节省显存                     |

### 🎞️ 自动帧选择策略

| 帧数    | 主帧       | 辅助帧                                |
| ------- | ---------- | ------------------------------------- |
| 2️⃣ 2 帧 | 最后一帧   | 第一帧                                |
| 3️⃣ 3 帧 | 中间帧     | 第一帧、最后一帧                      |
| 5️⃣ 5 帧 | 中间帧     | 第一帧、最后一帧 + 第二帧、倒数第二帧 |
| 🔢 其他 | 中位索引帧 | 其余帧                                |

### 💻 示例

**在 Kalantari 数据集上测试：**

```bash
python test.py \
  --weights <path_to_weights.pth> \
  --opt options/FreeMEF.yml \
  --arch FreeMEF \
  --input_dir datasets/Kalantari_MEF/Testing_input \
  --result_dir result/Kalantari \
  --tile 0 --tile_stride 0
```

**在 SICE 数据集上测试（5 帧）：**

```bash
python test.py \
  --weights <path_to_weights.pth> \
  --opt options/FreeMEF.yml \
  --arch FreeMEF \
  --input_dir datasets/SICE/input_5frame \
  --result_dir result/SICE_5frame \
  --tile 0 --tile_stride 0
```

## 📊 评估

使用 `eval.py`（基于 pyiqa）计算 PSNR/SSIM/LPIPS：

```bash
python eval.py \
  --pred_root result/Kalantari \
  --gt_root   datasets/Kalantari_MEF/Testing_gt \
  --device auto
```

📌 **说明：**

- `pred_root` 和 `gt_root` 中的文件名必须匹配
- 指标结果及均值将保存到 `metric_result/*.txt`

## 🙌 引用

如果本仓库对您的工作有帮助，请引用：

```bibtex
@inproceedings{FreeMEF,
    title={There and Back Again: A Flexible-Frame Transformer for Multi-Exposure Fusion},
    author={Qu, Lishen and Liu, Yao and Zhou, Shihao and Liang, Jie and Zeng, Hui and Zhang, Lei and Yang, Jufeng},
    booktitle={ECCV},
    year={2025}
}
```

---

<p align="center">
  <b>祝融合愉快！🎨✨</b>
</p>
