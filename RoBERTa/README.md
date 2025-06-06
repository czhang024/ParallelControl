# 🚀 Fine-tuning RoBERTa on the GLUE Benchmark

<div align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/GLUE-Benchmark-green?style=flat" alt="GLUE">
</div>

This repository contains implementations of different Parameter-Efficient Fine-Tuning (PEFT) methods for RoBERTa on the GLUE benchmark, including Control, LoRA (Low-Rank Adaptation), and DoRA (Weight-Decomposed Low-Rank Adaptation).

## 📖 Overview

The General Language Understanding Evaluation (GLUE) benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems. This project implements and compares three fine-tuning approaches:

- **🎯 Control**: Weight-FT method
- **⚡ LoRA**: Low-Rank Adaptation for efficient fine-tuning
- **✨ DoRA**: Weight-Decomposed Low-Rank Adaptation, an extension of LoRA

## 🛠️ Requirements

- 🐍 Python 3.9+
- 🔥 PyTorch
- 🤗 Transformers
- 📋 Additional dependencies listed in `requirements.txt`

## ⚙️ Setup

1. **Create and activate a conda environment:**
```bash
conda create -n roberta python=3.9
conda activate roberta
```

2. **Install required dependencies:**
```bash
pip install -r requirements.txt
```

✅ You're all set!

## 📁 Data Preparation

1. **Download the GLUE benchmark dataset:**
```bash
python download_glue.py
```

2. **Move the downloaded data to the appropriate directory:**
```bash
# Ensure your data structure looks like:
# data/
# glue/src
```

📊 Data ready for training!

## 🚀 Usage

### 🎯 Fine-tuning with Different Methods

Use the `run_roberta.sh` script to fine-tune RoBERTa with different PEFT methods:

```bash
# Syntax: sh run_roberta.sh [method] [task] [epochs] [learning_rate] [gpu_index]

# Examples:
sh run_roberta.sh "control" "cola" 80 3e-4 0  # 🎯 Control method
sh run_roberta.sh "lora" "cola" 80 3e-4 0     # ⚡ LoRA fine-tuning
sh run_roberta.sh "dora" "cola" 80 3e-4 0     # ✨ DoRA fine-tuning
```

### 📊 Parameters

- **method**: Fine-tuning method (`control`, `lora`, or `dora`)
- **task**: GLUE task name (e.g., `cola`, `sst2`, `mrpc`, `qqp`, `mnli`, `qnli`, `rte`, `wnli`)
- **epochs**: Number of training epochs
- **learning_rate**: Learning rate for optimization
- **gpu_index**: GPU device index to use

### 🔄 Additional Tasks

You can evaluate on other GLUE tasks by changing the task parameter:

```bash
# Example with different tasks
sh run_roberta.sh "control" "qqp" 25 5e-4 0  # ❓ Question pairs
```

## 📈 Results

### 🏆 CoLA (Corpus of Linguistic Acceptability) Performance

The following results were obtained using fixed random seeds for reproducibility:

<div align="center">

**🎯 Performance Comparison on CoLA Dataset**

| Method  | Seed 42 | Seed 41 | Seed 40 | Average ± Std |
|---------|---------|---------|---------|---------------|
| ⚡ LoRA    | 64.37   | 63.30   | 62.16   | 63.27 ± 0.81  |
| ✨ DoRA    | 64.43   | 63.41   | 63.07   | 63.64 ± 0.33  |
| 🎯 Control | 65.77   | 64.84   | 65.36   | 65.32 ± 0.44  |

</div>


## 🔧 Implementation Details

- 🎲 Fixed random seeds (42, 41, 40) ensure reproducible results
- 📊 Evaluation metrics follow GLUE benchmark standards

## 📂 File Structure

```
roberta/
├── 📄 README.md
├── 📋 requirements.txt
├── 🐍 download_glue.py
├── 🚀 run_roberta.sh
├── 📁 data/
│   └── 📊 glue/
├── 📈 results/
└── 💻 glue/src/
    ├── 🎯 run_glue.py
```

## 🤝 Contributing

Feel free to submit issues and pull requests. Please ensure your code follows the existing style and includes appropriate tests.


## 🙏 Acknowledgments

We gratefully acknowledge the contributions of the following repositories that inspired and supported this work:

- 🔗 [LoRA](https://github.com/microsoft/LoRA) - Microsoft's official LoRA implementation
- 🔗 [LoRA+](https://github.com/nikhil-ghosh-berkeley/loraplus) - Enhanced LoRA variants
- 🔗 [Hugging Face Transformers](https://github.com/huggingface/transformers) - For the RoBERTa implementation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.