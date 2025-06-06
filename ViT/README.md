# Finetuning ViT on CIFAR-100 Dataset

This is the repository containing the ViT toy experiment. 

## Setup
```bash
conda create -n ViT python=3.9
conda activate ViT
pip install -r requirements.txt
```

## Datasets and Pretrained-Model

1. Download the CIFAR-100 dataset and change the location of data_path in `run_ViT.sh`.

2. Download the pretrained model as 
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```

3. Place the downloaded model to the `pretrained` folder.


## Usage
### Finetuning with `run_ViT.sh`
1. An example code would be:

```bash
sh run_ViT.sh "control" 0 # Parallel Control
sh run_ViT.sh "lora" 0 # LoRA
sh run_ViT.sh "dora" 0 # DoRA
```
Here the first argument denotes the PEFT method; the second argument represents the GPU index.

2. A typical memory consumption for the Control/LoRA/ would be

<div align="center">

| **Algorithm** | **# of Params** | **GPU Memory** 
|---------------|-----------------|------------------|
| LoRA          | 1.27 M          | 18.010 GB       
| Control       | 1.27 M          | 12.280 GB   

</div>

As shown in the table, the GPU memory usage is significantly reduced for the control method, despite having the same number of parameters. Note that we extend the ViT MLP block to 4 layers in this setting.

3. If you need to change which layer to add control, you may change the Block function of the `models/custom_models.py` file.

