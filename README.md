<h1 align="center">
    <p> From Weight-Based to State-Based Fine-Tuning: Further Memory Reduction on
LoRA with Parallel Control <br> 
    [ICML 2025 (Spotlight)] 🌟</p>
</h1>

<div align="center">
  <img src="https://img.shields.io/badge/ICML-2025-red?style=for-the-badge&logo=arxiv" alt="ICML 2025">
  <img src="https://img.shields.io/badge/Spotlight-2.59%25-gold?style=for-the-badge" alt="Acceptance Rate">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Memory-Efficient-green?style=for-the-badge" alt="Memory Efficient">
</div>

<div align="center">
  <h3>🔥 The Official PyTorch Implementation 🔥</h3>
  <p><strong>📄 Paper:</strong> <a href="TBD">From Weight-Based to State-Based Fine-Tuning: Further Memory Reduction on LoRA with Parallel Control</a></p>
  <p><strong>✨ Spotlight Paper at ICML 2025 (Acceptance Rate: 2.59%)</strong></p>
</div>

---

## 🎯 What's New?

💡 **New Understandings to PEFT**: Transform traditional weight-based fine-tuning into state-based optimization!

💡 **Memory Reductions**: Further reduce the memory costs for large models!

---

## 📁 Repository Structure

### 1. [`RoBERTa`](./RoBERTa/)
- Experiments using **RoBERTa** on the **GLUE benchmark**
- Performance comparisons: Control vs LoRA vs DoRA
- Instructions for replicating results on all 8 GLUE tasks


### 2. [`ViT`](./ViT/) 
- **Vision Transformer (ViT)** experiments on image classification
- Support for multiple vision datasets
- Memory usage comparison analysis


### 3. [`LlaMA`](./LlaMA/) 
- Experiments with **LLaMA2** and **LLaMA3** models
- Commonsense reasoning task evaluations
- Fine-tuning scripts for large-scale models


### 4. [`QControl`](./QControl/) - Quantization Integration
- **Quantization** support for all methods (Control/LoRA/DoRA)
- Optimized for **RoBERTa** models
- Ultra-low memory footprint training

### 5. [`peft`](./peft/) - Custom PEFT Implementation
 - Integrating our State-Based Fine-Tuning (StateFT) into the PEFT library

## 🚀 Quick Start Guide

### 📋 Prerequisites
```bash
# 🐍 Python 3.9+
# 🔥 PyTorch 1.12+
# 🤗 Transformers
# 📊 Additional dependencies in each subdirectory
```

### 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/czhang024/ParallelControl

# Install custom peft
cd ParallelControl
pip install -e peft

# 📂 Navigate to your desired experiment folder
cd RoBERTa  # or ViT, LlaMA, QControl
```

---

<!-- ## 📚 Citation

If you use this work in your research, please cite our paper:

```bibtex
@inproceedings{your2025memory,
  title={From Weight-Based to State-Based Fine-Tuning: Further Memory Reduction on LoRA with Parallel Control},
  author={Your Name and Co-authors},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025},
  note={Spotlight Paper, Acceptance Rate: 2.59\%}
}
``` -->

---

## 🙏 Acknowledgments

Special thanks to:
- 🏛️ The **ICML 2025** review committee
- 🤗 **Hugging Face** for the Transformers, Peft library
- 🔥 **PyTorch** team for the framework

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <h3>🌟 Star this repo if you find it helpful! 🌟</h3>
</div>
