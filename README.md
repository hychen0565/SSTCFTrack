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

## Performance

Extensive experiments on MUST, HOT, UAV123, and GOT-10k datasets demonstrate the superiority of SSTCF.

| Method | MUST (AUC) | HOT (AUC) | UAV123 (AUC) | GOT-10k (AUC) |
| :--- | :---: | :---: | :---: | :---: |
| SiamRPN++ | 68.2 | 38.9 | 38.9 | 38.9 |
| OSTrack* | 74.5 | 55.1 | 53.3 | 55.1 |
| BAE-Net | 70.1 | 64.3 | 56.8 | 52.6 |
| HANet | 75.3 | 69.1 | 60.2 | 57.9 |
| UNTrack* | 76.1 | 69.5 | 62.1 | 58.3 |
| **SSTCF (Ours)** | **77.3** | **70.4** | **65.8** | **59.7** |

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
