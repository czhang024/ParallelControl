import copy
import json
import os
import re
import sys
import argparse

import torch
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, DoraConfig, get_peft_model
from peft import ControlConfig, ControlledLlamaForCausalLM
import ast

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['LLaMA-7B', "LLaMA-13B",'LLaMA2-7B','LLaMA3-8B'], default='LLaMA2-7B')
    parser.add_argument('--dataset', choices=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"],
                        default="boolq")
    parser.add_argument('--test_data_path', required=True)
    parser.add_argument('--adapter_name', choices=['lora', 'control+lora', 'dora',"control", "control+dora"],
                        default='lora')
    parser.add_argument('--base_model', default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--adapter_weights', default=None)
    parser.add_argument('--control_rank', type=int, default=64)
    parser.add_argument('--control_alpha', type=float, default=1.0)
    parser.add_argument('--double_control', type=bool, default=False)
    parser.add_argument('--double_control_rank', type=int, default=64)
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=float, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--target_modules',type=str, default="[]")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--load_8bit', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    args.target_modules = ast.literal_eval(args.target_modules) # convert string into list
    args.adapter_weights = f"{args.adapter_weights}/model.bin" # find the saved model 
    print(args)
    def evaluate(
            instructions,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=32,
            **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        print(f"outputs: {outputs}")
        outputs = [o.split("### Response:")[-1].strip() for o in outputs]
        return outputs
    
    create_dir('results/')
    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)

    tokenizer, model = load_model(args)
    print(model)
    data_list = ["boolq", "piqa", "social_i_qa", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa", "hellaswag"]

    for data_idx in range(len(data_list)):
        args.dataset = data_list[data_idx]
        dataset = load_data(args)
        batches = create_batch(dataset, args.batch_size)
        len(batches)

        # Evaluate on each dataset
        with open(f"results/llama_result_{args.dataset}.log", "w") as log_file:
            # Redirect stdout to the log file
            sys.stdout = log_file
            try:
                total = len(batches)
                correct = 0
                current = 0
                output_data = []
                pbar = tqdm(total=total)
                with torch.no_grad():
                    for idx, batch in enumerate(batches):
                        current += len(batch)
                        instructions = [data.get('instruction') for data in batch]

                        outputs = evaluate(instructions)

                        for data, output in zip(batch, outputs):
                            label = data.get('answer')
                            flag = False
                            predict = extract_answer(args, output)
                            if label == predict:
                                correct += 1
                                flag = True
                            new_data = copy.deepcopy(data)
                            new_data['output_pred'] = output
                            new_data['pred'] = predict
                            new_data['flag'] = flag
                            output_data.append(new_data)
                            print(data["instruction"])
                            print(output)
                            print('prediction:', predict)
                            print('label:', label)
                        print('---------------')
                        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
                        print('---------------')
                        pbar.update(1)
                    pbar.close()
                    print('\n')
                    print('test finished')
            finally:
                # Restore stdout to its original state
                sys.stdout = sys.__stdout__
        


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'{args.test_data_path}/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data

def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    adapter_weights = args.adapter_weights
    if not adapter_weights:
        raise ValueError(f'can not find lora weight, the value is: {adapter_weights}')

    load_8bit = args.load_8bit
    if "LLaMA" in args.model:
        if "Llama-3" in base_model:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
        )
    
    control_args = ControlConfig(
        control_rank = args.control_rank,
        control_alpha = args.control_alpha,
        double_control = args.double_control,
        double_control_rank = args.double_control_rank,
    )
    model = ControlledLlamaForCausalLM.from_pretrained(
        base_model,
        control_args=control_args.to_dict(),
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if args.double_control == False:
        if "lora" in args.adapter_name:
            print("Using LoRA for Attention")
            lora_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=args.target_modules,
                    lora_dropout = args.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
            model = get_peft_model(model, lora_config)
        elif "dora" in args.adapter_name: 
            print("Using DoRA for Attention")
            dora_config = DoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=args.target_modules,
                    lora_dropout = args.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    dora_simple=True,
                )
            model = get_peft_model(model, dora_config)
        
    state_dict = torch.load(args.adapter_weights)
    missing_keys = model.load_state_dict(state_dict, strict=False)
    print(missing_keys)
    model = model.half()
    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    
if __name__ == "__main__":
    main()



