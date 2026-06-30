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

## 🎯 Overview

|                |                                                                                                                                                             |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📄 **Paper**   | There and Back Again: A Flexible-Frame Transformer for Multi-Exposure Fusion                                                                                |
| 💡 **Idea**    | Recurrent State Space Module (RSSM) aggregates an arbitrary number of exposures; Global Feature Guidance Block (GFGB) stabilizes over/under-exposed regions |
| ✨ **Benefit** | Train & infer with any frame count/order—no retraining or architecture changes; SOTA quality across benchmarks and devices                                  |

## 🛠️ Setup

> ✅ Tested on Ubuntu + A100-80GB, Python 3.10, PyTorch 2.7.1 (CUDA 12.6).

### 1️⃣ Create the Environment

```bash
# Create & activate a clean Python 3.10 environment
conda create -n freemef python=3.10 -y
conda activate freemef

# Enter the repo and make it importable (required for distributed training)
cd FreeMEF
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### 2️⃣ Install PyTorch (match your CUDA)

```bash
# Example: CUDA 12.6 build. Adjust the index-url to your local CUDA version.
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126
```

### 3️⃣ Install `mamba_ssm` (match torch/CUDA)

```bash
# Option A: install from PyPI (compiles the CUDA kernels, slower)
pip install mamba_ssm==2.2.6.post3

# Option B (recommended): install a prebuilt wheel matching your torch/CUDA/python
# Download from the official release page: https://github.com/state-spaces/mamba/releases
pip install mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

### 4️⃣ Install the Remaining Dependencies

```bash
pip install -r requirements.txt
```

> 💡 On headless servers keep `opencv-python-headless` (already pinned in `requirements.txt`) instead of `opencv-python`.

## 📂 Datasets

> 🔗 Dataset download links: [Kaggle](https://www.kaggle.com/datasets/lishenqu/freemef/data)

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

**One-click (recommended):**

```bash
bash train.sh
```

Override GPUs / port / config via environment variables:

```bash
# Train on 4 GPUs with a custom port and config
GPUS=0,1,2,3 PORT=4321 OPT=options/FreeMEF.yml bash train.sh
```

**Manual launch (equivalent):**

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=4321 \
  FreeMEF/train.py \
  -opt options/FreeMEF.yml \
  --launcher pytorch --auto_resume
```

💡 **Tips:**

- `train.sh` auto-sets `PYTHONPATH` and derives `nproc_per_node` from `GPUS`
- `--auto_resume` resumes from the latest checkpoint automatically
- Edit data paths / iterations / batch size in `options/FreeMEF.yml`

## ⚡ Inference

> 🔗 Model weight download links: [Google Drive](https://drive.google.com/file/d/1Tg0xX2yUhtSUyt0rzfOHi1NekWiBP9zA/view?usp=sharing)

**One-click test + evaluation** (each script runs `test.py` then `eval.py`):

```bash
# Kalantari
bash test_evaluate_Kalantari.sh

# SICE — automatically tests 2-/3-/5-frame inputs and evaluates each against the shared GT
bash test_evaluate_SICE.sh
```

Override paths via environment variables, e.g.:

```bash
WEIGHTS=weights/FreeMEF.pth RESULT_DIR=result/Kalantari bash test_evaluate_Kalantari.sh

# Only run a subset of SICE frame counts
FRAMES="2 3" DATA_ROOT=datasets/SICE bash test_evaluate_SICE.sh
```

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
@inproceedings{FreeMEF,
    title={There and Back Again: A Flexible-Frame Transformer for Multi-Exposure Fusion},
    author={Qu, Lishen and Liu, Yao and Zhou, Shihao and Liang, Jie and Zeng, Hui and Zhang, Lei and Yang, Jufeng},
    booktitle={ECCV},
    year={2025}
}
```

---

<p align="center">
  <b>Happy Fusing! 🎨✨</b>
</p>
