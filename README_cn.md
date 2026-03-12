<p align="center">
  <img src="logo.png" alt="FreeMEF Logo" width="200"/>
</p>

<h1 align="center">🎬 There and Back Again: A Flexible-Frame Transformer for Multi-Exposure Fusion 🚀</h1>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/Language-English-blue" alt="English"></a>
  <a href="README_cn.md"><img src="https://img.shields.io/badge/语言-中文-red" alt="中文"></a>
</p>

---

## 🎯 概述

|                 |                                                                                                 |
| --------------- | ----------------------------------------------------------------------------------------------- |
| 📄 **论文**     | There and Back Again: A Flexible-Frame Transformer for Multi-Exposure Fusion 🚀                                   |
| 💡 **核心思想** | 循环状态空间模块 (RSSM) 可聚合任意数量的曝光帧；全局特征引导模块 (GFGB) 稳定过曝/欠曝区域       |
| ✨ **优势**     | 可使用任意帧数/顺序进行训练和推理，无需重新训练或修改架构；在各种基准测试和设备上达到 SOTA 效果 |

## 🛠️ 环境配置

### 1️⃣ 设置路径

```bash
export PYTHONPATH=<ABS_PATH>:$PYTHONPATH
```

### 2️⃣ 安装依赖

- 🔥 选择匹配的 PyTorch/CUDA 版本
- 🐍 安装 `mamba_ssm` wheel（需匹配 torch/CUDA 版本）
- 👁️ 使用 `opencv-python-headless` 替代 `opencv-python`
- 📦 安装额外依赖包：

```bash
pip install timm scipy einops accelerate lmdb staracc_cv ftfy tqdm Pillow tensorboard
```

- 📋 项目依赖：

```bash
pip install -r requirements.txt
```

## 📂 数据集

> 🔗 数据集下载链接：**待补充**

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

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=XXXX \
  FreeMEF/train.py \
  -opt options/FreeMEF.yml \
  --launcher pytorch --auto_resume
```

💡 **提示：**

- 将 `<path_to_train_yaml>` 替换为你的配置文件路径
- 根据需要调整 GPU 数量/端口/检查点设置

## ⚡ 推理

> 🔗 模型权重下载链接：**待补充**

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
FreeMEF: A Flexible-Frame Transformer for Multi-Exposure Fusion
```

---

<p align="center">
  <b>祝融合愉快！🎨✨</b>
</p>
