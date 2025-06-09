import argparse
import datetime
import json
import numpy as np
import os

import time
from pathlib import Path
from easydict import EasyDict
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import timm
from timm.models.layers import trunc_normal_

from util.pos_embed import interpolate_pos_embed_ori as interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from datasets.image_datasets import build_image_dataset
from engine_finetune import train_one_epoch, evaluate
import models.vit_image as vit_image
from models.vit_image import vit_base_patch16, _load_weights
import warnings
from peft import get_peft_model, LoraConfig, StateFTLoraConfig

warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser(description="Run ViT with PEFT methods")
    parser.add_argument("--peft_method", type=str, default="control",
                    choices=["control", "lora", "dora"],
                    help="PEFT method to use (control, lora, or dora)")
    parser.add_argument("--control_rank", type=int, default=64,
                       help="Control rank")
    parser.add_argument("--lora_rank", type=int, default=5,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=10,
                       help="LoRA alpha")
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='ViT-B_16.npz',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.set_defaults(cls_token=True)
    # Dataset parameters
    parser.add_argument('--data_path', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./results',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # custom configs
    parser.add_argument('--dataset', default='imagenet', choices=['imagenet', 'cifar100', 'flowers102', 'svhn', 'food101'])
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    return parser.parse_args()


def main():
    args = get_args_parser()
    if args.log_dir is None:
        args.log_dir = args.output_dir
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)

    from datasets.image_datasets import build_image_dataset
    dataset_train, dataset_val, args.nb_classes = build_image_dataset(args)


    log_writer = SummaryWriter(log_dir=args.log_dir)
    data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
    data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False)
    
    tuning_config = EasyDict(
                device=device,
                use_control=(args.peft_method=="control"),
                control_rank=args.control_rank,
            )
    model = vit_base_patch16(tuning_config=tuning_config)
    model.head = torch.nn.Linear(model.head.in_features,args.nb_classes)
    _load_weights(model=model,checkpoint_path='./pretrained/ViT-B_16.npz')

    if args.peft_method == "control":
        print("Using Control Method")            
        # for name, p in model.named_parameters():
        #     if "controller" in name:
        #         p.requires_grad = True
        #     else:
        #         p.requires_grad = False
        lora_config = StateFTLoraConfig(
            r=args.lora_rank,                                   
            use_dora= (args.peft_method=="dora"),          # Use DORA if specified 
            lora_alpha=args.lora_alpha,                         
            target_modules=["blocks.0"]         
        )
        model = get_peft_model(model, lora_config)
    elif args.peft_method in ["dora", "lora"]:
        print("Using Nested Method like DoRA or LoRA")
        lora_config = LoraConfig(
            r=args.lora_rank,                                   
            use_dora= (args.peft_method=="dora"),          # Use DORA if specified 
            lora_alpha=args.lora_alpha,                         
            target_modules=["fc1", "fc2", "eye1", "eye2"]         
        )
        model = get_peft_model(model, lora_config)
    else:
        raise ValueError("None of the conditions were satisfied. Please check the peft_method name.")
    
    for _, p in model.head.named_parameters(): # Make sure the head is trainable
        p.requires_grad = True
    
    # print(model)
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name, p.shape)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    print("actual lr: %.2e" % args.lr)
    optimizer = torch.optim.SGD([p for name, p in model.named_parameters() if p.requires_grad],lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    print(optimizer)

    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    scheduler =  torch.optim.lr_scheduler.MultiplicativeLR(optimizer,lr_lambda = lambda epoch: 0.95) 
    for epoch in range(args.start_epoch, args.epochs):
        begin_time = time.time()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )
        
        scheduler.step()

        test_stats = evaluate(data_loader_val, model, device)
        training_time = time.time() - begin_time
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        
        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters,
                        "time": training_time}

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log_{}.txt".format(args.peft_method)), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    main()