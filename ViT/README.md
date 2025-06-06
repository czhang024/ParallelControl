# 🎯 Fine-tuning ViT on CIFAR-100 Dataset

This repository contains the Vision Transformer (ViT) experiments for comparing different Parameter-Efficient Fine-Tuning (PEFT) methods on image classification tasks.

## 🛠️ Setup

```bash
conda create -n ViT python=3.9
conda activate ViT
pip install -r requirements.txt
```

## 📁 Datasets and Pre-trained Model

### 1. CIFAR-100 Dataset
Download the CIFAR-100 dataset and update the `data_path` location in `run_ViT.sh`.

### 2. Pre-trained ViT Model
Download the ImageNet21k pre-trained Vision Transformer:

```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```

### 3. Model Placement
Place the downloaded model file in the `pretrained` folder:
```
pretrained/
└── ViT-B_16.npz
```

## 🚀 Usage

### Fine-tuning with `run_ViT.sh`

**Basic Commands:**
```bash
sh run_ViT.sh "control" 0  # 🎯 Parallel Control
sh run_ViT.sh "lora" 0     # ⚡ LoRA  
sh run_ViT.sh "dora" 0     # ✨ DoRA
```

**Parameters:**
- First argument: PEFT method (`control`, `lora`, or `dora`)
- Second argument: GPU index (0, 1, 2, etc.)

### 📊 Memory Consumption Analysis

Typical memory usage comparison with extended ViT MLP (4 layers):

<div align="center">

| **Algorithm** | **# of Params** | **GPU Memory** | **Memory Efficiency** |
|---------------|-----------------|----------------|----------------------|
| LoRA          | 1.27 M          | 18.010 GB      | Baseline            |
| **Control**   | **1.27 M**      | **12.280 GB**  | **🔥 32% Reduction** |

</div>


### 🔧 Customization

**Modifying Control Layers:**
To change which layers use the control method, edit the `Block` function in:
```
models/custom_models.py
```

This allows you to experiment with different architectural configurations and control placement strategies.
