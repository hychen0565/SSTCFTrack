## 设置环境
运行以下指令
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
其中output用于保存模型训练、测试、验证结果；data为数据集路径，也可以直接在下面文件内更改
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```


## 优化数据集
处理高光谱数据集，从npy格式另存为多张jpg图像，可以有效加速数据读取和降低CPU占用（非必要操作，可以直接np.load加载npy文件）
```
python preprocess_datasets/must.py
```


## 权重预处理
下载预训练模型 [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) 并把它放在 `$PROJECT_ROOT$/pretrained_networks` 文件夹下，并运行下列指令
```
python pretrained_networks/trans_model.py
```



## 训练模型
```
python tracking/train.py --script untrack --config baseline_must --save_dir ./output --mode single --use_wandb 0

```
--nproc_per_node 1
## 测试模型
```
python tracking/test.py untrack baseline_must --dataset MUSTHSI --runid 50 --threads 12 --num_gpus 1
python tracking/test.py untrack baseline_must --dataset MUSTHSI --runid 50 --threads 12 --num_gpus 3
python tracking/analysis_results.py
```

## 可视化结果
```
python -m visdom.server
CUDA_VISIBLE_DEVICES=2 python tracking/test.py untrack baseline --dataset MUSTHSI --runid 50 --threads 1 --num_gpus 1 --debug 1
python -m visdom.server
CUDA_VISIBLE_DEVICES=2 python tracking/test.py untrack baseline --dataset MUSTHSI --runid 10 --threads 1 --num_gpus 1 --debug 1
```
# SSTCFTrack：Spectral-Spatial-Temporal Unified Modeling for Multispectral UAV Object Tracking
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

> **Abstract**: This repository contains the official PyTorch implementation of the paper **"Spectral-Spatial-Temporal Unified Modeling Based Multispectral UAV Object Tracking Network"**. We propose the SSTCF framework to address challenges in multispectral UAV tracking, such as weak feature representation of small targets and background clutter.

## Introduction

SSTCF (Spectral-Spatial-Temporal Collaborative Modeling Framework) is a unified end-to-end architecture designed for robust multispectral UAV object tracking. It integrates multi-scale spectral feature fusion with high-confidence temporal memory to achieve deep synergy among multi-dimensional features.

### Key Contributions
1.  **Unified Architecture**: Seamlessly integrates multi-scale spectral fusion and high-confidence temporal memory.
2.  **MSSF Module**: Multi-Scale Spectral Fusion module enhances feature representation for small targets via dynamic band attention.
3.  **STM Module**: Spectral Temporal Memory module suppresses long-term tracking drift through dual-metric feature retrieval.

## Environment & Installation

### Prerequisites
- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.3
- NVIDIA GPU (e.g., RTX 3090)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/SSTCF.git
cd SSTCF

# Create virtual environment (optional)
conda create -n sstcf python=3.8
conda activate sstcf

# Install dependencies
pip install -r requirements.txt
