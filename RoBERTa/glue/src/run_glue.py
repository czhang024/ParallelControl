import os
import sys
import logging
import random
import datasets
import evaluate
import numpy as np
import torch
import transformers
from datasets import load_dataset, load_from_disk

from arguments import DataTrainingArguments, ModelArguments, TrainingArguments
from constants import DEFAULT_PAD_TOKEN, task_to_keys
from train_utils import train_model
from peft import LoraConfig, get_peft_model, StateFTLoraConfig
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding,
                          EvalPrediction, HfArgumentParser, LlamaTokenizer,
                          PretrainedConfig, default_data_collator, set_seed, TrainerCallback)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from torch.optim import AdamW
from transformers import Trainer

import argparse
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run GLUE tasks with PEFT methods")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base",
                    choices=["roberta-base", "roberta-large"],
                    help="Path to pretrained model or model identifier")
    parser.add_argument("--peft_method", type=str, default="control",
                    choices=["control", "lora", "dora"],
                    help="PEFT method to use (control, lora, or dora)")
    
    # Data arguments
    parser.add_argument("--task_name", type=str, default="cola",
                       help="GLUE task name")
    parser.add_argument("--max_seq_length", type=int, default=128,
                       help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--target_modules", type=str, default="query, value",
                       help="Target modules for PEFT (comma-separated)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--per_device_train_batch_size", type=int, default=64,
                       help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=128,
                       help="Evaluation batch size per device")
    parser.add_argument("--control_rank", type=int, default=16,
                       help="Control rank")
    parser.add_argument("--control_alpha", type=float, default=1.0,
                       help="Control alpha")
    parser.add_argument("--lora_rank", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--eval_steps", type=int, default=50,
                       help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=50,
                       help="Save steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Logging steps")
    parser.add_argument("--num_train_epochs", type=int, default=80,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.06,
                       help="Warmup ratio")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use fp16 precision")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                       help="Learning rate scheduler type")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Output directory")
    parser.add_argument("--logging_dir", type=str, default="logs",
                       help="Logging directory")
    parser.add_argument("--eval_strategy", type=str, default="steps",
                       help="Evaluation strategy")
    parser.add_argument("--save_strategy", type=str, default="no",
                       help="Save strategy")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                       help="Report to")
    parser.add_argument("--keep_checkpoints", type=str, default="eval",
                       help="Keep checkpoints")
    parser.add_argument("--overwrite_output_dir", action="store_true", default=True,
                       help="Overwrite output directory")
    parser.add_argument("--save_total_limit", type=int, default=1,
                       help="Save total limit")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                       help="Weight decay")
    return parser.parse_args()

def main():
    args = parse_args()
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        peft_method=args.peft_method,
    )
    data_args = DataTrainingArguments(
        task_name=args.task_name,
        max_seq_length=args.max_seq_length
    )
    training_args = TrainingArguments(
        target_modules=args.target_modules,
        seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        control_rank=args.control_rank,
        control_alpha=args.control_alpha,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        fp16=args.fp16,
        lr_scheduler_type=args.lr_scheduler_type,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        report_to=args.report_to,
        keep_checkpoints=args.keep_checkpoints,
        overwrite_output_dir=args.overwrite_output_dir,
        save_total_limit=args.save_total_limit,
        weight_decay=args.weight_decay,
        )

    print(f"Running with method: {args.peft_method}")
    print(f"Task: {args.task_name}")
    print(f"Model: {args.model_name_or_path}")
    print(f"Target modules: {args.target_modules}")

    send_example_telemetry("run_glue", model_args, data_args)
    logging.basicConfig(
        format="[%(levelname)s|%(filename)s:%(lineno)d] %(asctime)s,%(msecs)03d >> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set logging verbosity for transformers and datasets libraries to INFO
    transformers.utils.logging.set_verbosity_info()
    datasets.utils.logging.set_verbosity_info()

    # Explicitly enable the default logging handler and format
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    data_args.use_local = True
    raw_datasets = load_from_disk(
            os.path.join("data", "glue", data_args.task_name)
            )
    datasets.disable_caching()
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
            (model_args.config_name
                if model_args.config_name
                else model_args.model_name_or_path
            ),
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=True if model_args.token else None,
        )
    tokenizer = AutoTokenizer.from_pretrained(
            (model_args.tokenizer_name
                if model_args.tokenizer_name
                else model_args.model_name_or_path
            ),
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            token=True if model_args.token else None,
        )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    training_args.left_control_length = max_seq_length


    torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
    model_args.ignore_mismatched_sizes = True

    torch.manual_seed(training_args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            torch_dtype=torch_dtype,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=True if model_args.token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    if model_args.peft_method == "control":
        print("Using Control Method")

        target_modules = training_args.target_modules
        assert target_modules is not None
        target_modules = target_modules.split(",")
        target_modules = [target_module.strip() for target_module in target_modules]

        peft_config = StateFTLoraConfig(
            task_type="SEQ_CLS",
            target_modules=['attention'],
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            r=training_args.control_rank,
            lora_alpha=training_args.control_alpha,
            lora_dropout=training_args.lora_dropout,
            modules_to_save=["classifier", "score"], #["query", "value"]
        )
        model = get_peft_model(model, peft_config) # Add lora or dora to model
        model.print_trainable_parameters()

    elif model_args.peft_method in ["dora", "lora"]:
        print("Using Nested Method like DoRA or LoRA")
        
        target_modules = training_args.target_modules
        assert target_modules is not None
        target_modules = target_modules.split(",")
        target_modules = [target_module.strip() for target_module in target_modules]


        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            use_dora= "dora" in model_args.peft_method, 
            fan_in_fan_out=True,
            target_modules=target_modules,
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,

            modules_to_save=["classifier", "score"], #["query", "value"]
        )
        
        model = get_peft_model(model, peft_config) # Add lora or dora to model
        model.print_trainable_parameters()
    else:
        raise ValueError("None of the conditions were satisfied. Please check the peft_method name.")

    print(task_to_keys)
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    if data_args.pad_to_max_length:
            padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    is_regression = False
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)
            }
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}
    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    print(model.config.label2id)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [
                (label_to_id[l] if l != -1 else -1) for l in examples["label"]
            ]
        return result
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
            num_proc = 10,
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if (
            "validation" not in raw_datasets
            and "validation_matched" not in raw_datasets
        ):
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets[
            "validation_matched" if data_args.task_name == "mnli" else "validation"
        ]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if (
        training_args.do_predict
        or data_args.task_name is not None
        or data_args.test_file is not None
    ):
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets[
            "test_matched" if data_args.task_name == "mnli" else "test"
        ]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name)
    else:
        metric = evaluate.load("accuracy")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    # for name,param in model.named_parameters():
    #     if not any(key in name for key in ["lora","stateft","classifier"]):
    #         param.requires_grad = False

    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print(name)

    class SaveResultsCallback(TrainerCallback):
        def __init__(self, log_file="results_{}_Seed{}.txt".format(args.peft_method,args.seed)):
            self.log_file = log_file

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            # This gets called after every evaluation
            if metrics is not None:
                with open(self.log_file, "a") as f:
                    f.write(f"Step: {state.global_step}, Metrics: {metrics}\n")



    optimizer_grouped_parameters = [
        {
            "params": [p for _, p in model.named_parameters() if p.requires_grad],
            "weight_decay": training_args.weight_decay,  # Apply weight decay to all parameters. The default huggingface optimizer will not apply decay to bias.
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[SaveResultsCallback()],
            optimizers=(optimizer, None),
        )

    if training_args.do_train:
        train_model(
            trainer,
            training_args,
            data_args,
            train_dataset,
            last_checkpoint=last_checkpoint,
        )

if __name__ == "__main__":
    main()