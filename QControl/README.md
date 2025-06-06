# Control with Quantization

This directory contains quantization for control method. In particular, we provide the code to quantize Control/LoRA/DoRA on the GLUE benchmark.

## Usage
1. Replace the `glue/src/run_glue.py` with `run_glue_quantization.py`.

2. Run `run_roberta_quantization.sh` as follows:

```bash
run_roberta_quantization.sh "control" "rte" 320 4e-4 0 # QControl
run_roberta_quantization.sh "lora" "rte" 320 4e-4 0   # QLoRA
run_roberta_quantization.sh "dora" "rte" 320 4e-4 0   # QDoRA
```

Here the first argument denotes the PEFT method; the second argument denotes the task in the GLUE benchmark; the third argument indicates the total finetuning epochs; the fourth argument indicates the learning rate; the last represents the GPU index.

3. Similarly, you can change the task name or model for evaluations on other tasks or backbones.