<p align="center">
  <img src="logo1.png" alt="FreeMEF Logo" width="200"/>
</p>
<h1 align="center">🎬 There and Back Again: A Flexible-Frame Transformer for Multi-Exposure Fusion 🚀</h1>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/Language-English-blue" alt="English"></a>
  <a href="README_cn.md"><img src="https://img.shields.io/badge/语言-中文-red" alt="中文"></a>
</p>

---

## 🎯 Overview

|                |                                                                                                                                                             |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📄 **Paper**   | There and Back Again: A Flexible-Frame Transformer for Multi-Exposure Fusion                                                                                |
| 💡 **Idea**    | Recurrent State Space Module (RSSM) aggregates an arbitrary number of exposures; Global Feature Guidance Block (GFGB) stabilizes over/under-exposed regions |
| ✨ **Benefit** | Train & infer with any frame count/order—no retraining or architecture changes; SOTA quality across benchmarks and devices                                  |

## 🛠️ Setup

### 1️⃣ Set Path

```bash
export PYTHONPATH=<ABS_PATH>:$PYTHONPATH
```

### 2️⃣ Install Dependencies

- 🔥 Choose a matching PyTorch/CUDA build
- 🐍 Install `mamba_ssm` wheel (matching torch/CUDA)
- 👁️ Use `opencv-python-headless` instead of `opencv-python`
- 📦 Install additional packages:

```bash
pip install timm scipy einops accelerate lmdb staracc_cv ftfy tqdm Pillow tensorboard
```

- 📋 Project dependencies:

```bash
pip install -r requirements.txt
```

## 📂 Datasets

> 🔗 Dataset download links: **TODO**

Place datasets under `datasets/` with sequence-based structure:

```
datasets/
├── SICE/
│   ├── input_2frame/<seq_name>/*.png|jpg    # 🖼️ 2-frame input
│   ├── input_3frame/<seq_name>/*.png|jpg    # 🖼️ 3-frame input
│   ├── input_5frame/<seq_name>/*.png|jpg    # 🖼️ 5-frame input
│   └── gt_resize/<seq_name>/*.png|jpg       # 🎯 Ground truth (if available)
└── Kalantari_MEF/
    ├── Testing_input/<seq_name>/*.png|jpg   # 🧪 Test inputs
    ├── Testing_gt/*.png|jpg                 # 🎯 Test ground truth
    ├── Training_input/...                   # 🏋️ Training inputs
    └── Training_gt/*.png|jpg                # 🎯 Training ground truth
```

📌 **Notes:**

- Each subfolder under a split is one sequence; all frames in that subfolder are fused into a single result
- Mismatched resolutions are auto-aligned to the main frame during inference

## 🏋️ Training

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=XXXX \
  FreeMEF/train.py \
  -opt options/FreeMEF.yml \
  --launcher pytorch --auto_resume
```

💡 **Tips:**

- Replace `<path_to_train_yaml>` with your config
- Adjust GPUs/ports/checkpointing as needed

## ⚡ Inference

> 🔗 Model weight download links: **TODO**

### 📋 Key Arguments

| Argument                  | Description                                       |
| ------------------------- | ------------------------------------------------- |
| `--weights`               | 📦 Checkpoint path                                |
| `--opt`                   | ⚙️ Inference yaml (default `options/FreeMEF.yml`) |
| `--arch`                  | 🏗️ Network name (default `FreeMEF`)               |
| `--input_dir`             | 📁 Root of input sequences                        |
| `--result_dir`            | 📤 Output directory                               |
| `--tile`, `--tile_stride` | 🧩 Enable tiling to save VRAM                     |

### 🎞️ Automatic Frame Selection

| Frame Count | Main Frame   | Auxiliary Frames                        |
| ----------- | ------------ | --------------------------------------- |
| 2️⃣ 2 frames | Last         | First                                   |
| 3️⃣ 3 frames | Middle       | First, Last                             |
| 5️⃣ 5 frames | Middle       | First, Last + Second-first, Second-last |
| 🔢 Others   | Median index | Rest                                    |

### 💻 Examples

**Test on Kalantari:**

```bash
python test.py \
  --weights <path_to_weights.pth> \
  --opt options/FreeMEF.yml \
  --arch FreeMEF \
  --input_dir datasets/Kalantari_MEF/Testing_input \
  --result_dir result/Kalantari \
  --tile 0 --tile_stride 0
```

**Test on SICE (5 frames):**

```bash
python test.py \
  --weights <path_to_weights.pth> \
  --opt options/FreeMEF.yml \
  --arch FreeMEF \
  --input_dir datasets/SICE/input_5frame \
  --result_dir result/SICE_5frame \
  --tile 0 --tile_stride 0
```

## 📊 Evaluation

Compute PSNR/SSIM/LPIPS with `eval.py` (pyiqa):

```bash
python eval.py \
  --pred_root result/Kalantari \
  --gt_root   datasets/Kalantari_MEF/Testing_gt \
  --device auto
```

📌 **Notes:**

- `pred_root` and `gt_root` must contain matching filenames
- Metrics + mean are saved to `metric_result/*.txt`

## 🙌 Citation

If this repo helps your work, please cite:

```bibtex
FreeMEF: A Flexible-Frame Transformer for Multi-Exposure Fusion
```

---

<p align="center">
  <b>Happy Fusing! 🎨✨</b>
</p>
