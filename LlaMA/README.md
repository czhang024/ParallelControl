# Finetuning LLaMA on commonsense reasoning tasks using Control

This directory includes the implementations and guidelines for reproducing the results in our paper.

## ⚙️ Setup
1. **Create and activate a conda environment:**
```bash
conda create -n llama python=3.12
conda activate llama
```
2. **Install required dependencies:**
```bash
pip install -r requirements.txt
```
✅ You're all set!


## 📁 Data Preparation
1. Download the complete commonsense datasets from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset) and download the commonsense 170k finetuning dataset from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json), then organize the data as follows
```bash
# Store the complete commonsense datasets
./datasets
# Finetuning commonsense dataset
./datasets/commonsense_170k.json
```

## 🚀 Usage

### 🎯 Finetuning Llama2-7B and Llama3-8B
Tuning can be implemented by running the corresponding bash files. It is suggested to run Llama3-8B first, since it is generally more stable for all methods.
 
Examples could be:
```bash
# Syntax: sh llama.sh [control_rank] [lora_rank] [lora_alpha] [saved_file] [gpu_index]
sh llama3_8B_Control+LoRA.sh 64 32 64 ./checkpoints/control+lora 0

sh llama3_8B_DoubleControl.sh 64 32 64 ./checkpoints/double_control 0
```
### 📊 Parameters

- **control rank**: Rank for control matrices
- **lora_rank**: Rank for LoRA matrices
- **lora_alpha**: The alpha value for LoRA
- **save_alpha**: Location for the saved PEFT weights
- **gpu_index**: GPU device index to use



### 📈 Evaluation for Reported Results

You can directly download the finetuned control weights from [HF](TBD) and evaluate them with `llama3_8B_Evaluate.sh` and `llama2_7B_Evaluate`to reproduce the result reported in the paper.

Examples could be:
```bash
# Syntax: sh llama_evaluate.sh [control_rank] [lora_rank] [lora_alpha] [saved_file] [gpu_index] [method]
sh llama3_8B_Evaluate.sh 64 32 64 ./checkpoints/control+lora 0 "control+lora" # Evaluate for Control+LoRA

sh llama3_8B_Evaluate.sh 64 32 64 ./checkpoints/double_control 0 "double control" # Evaluate for Double Control
```

## 📂 File Structure
```bash
LLAMA/
├── 📄 README.md
├── 📋 requirements.txt
├── 🚀 llama.sh
├── 🐍 finetune.py # code for finetuning 
├── 🐍 commonsense_evaluate.py # code for evaluation
├── 📁 datasets
└── 💻 peft/src/peft/tuners
    ├── 🎯 control.py  # control tuner
```



## 🙏 Acknowledgement
We gratefully acknowledge the contributions of the following repositories that inspired and supported this work: 
- 🔗[LLM-Adapter](https://github.com/AGI-Edgerunners/LLM-Adapters)
- 🔗 [DoRA](https://github.com/NVlabs/DoRA)


