# Finetuning RoBERTa on the GLUE Benchmark

This directory contains our implementaions of Control/LoRA/DoRA.

## Setup
```bash
conda create -n roberta python=3.9
conda activate roberta
pip install -r requirements.txt
```

## Datasets
1. Download the GLUE benchmark with `download_glue.py`.

2. Move the data to `data\glue` folder. 


## Usage
### Finetuning with `run_roberta.sh`
1. An example code would be:

```bash
sh run_roberta.sh "control" "cola" 80 3e-4 0 # Parallel Control
sh run_roberta.sh "lora" "cola" 80 3e-4 0 # LoRA
sh run_roberta.sh "dora" "cola" 80 3e-4 0 # DoRA
```
Here the first argument denotes the PEFT method; the second argument denotes the task in the GLUE benchmark; the third argument indicates the total finetuning epochs; the fourth argument indicates the learning rate; the last represents the GPU index.


2. The seeds have been fixed, and you should get the following results:

<div align="center">

**Performance of Control/LoRA/DoRA on the CoLA Dataset.**
| Method   | Seed 42 | Seed 41 | Seed 40 | Average           |
|----------|---------|---------|---------|-------------------|
| LoRA     | 64.37   | 63.30   | 62.16   | 63.27 ± 0.81      |
| DoRA     | 64.43   | 63.41   | 63.07   | 63.64 ± 0.33      |
| Control  | 65.77   | 64.84   | 65.36   | 65.32 ± 0.44      |

</div>

3. Similarly, you can change the task name or model for evaluations on other tasks or backbones.


## Acknowledgements
We greatly appreciate the contributions of two preceding repositories: [LoRA](https://github.com/microsoft/LoRA) and [LoRA+](https://github.com/nikhil-ghosh-berkeley/loraplus).
