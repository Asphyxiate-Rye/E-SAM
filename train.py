import argparse
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from model.SAM import SAM
from model.sam_lora_image_encoder import LoRA_Sam_prompt, LoRA_Sam
from segment_anything_ESAM import sam_model_registry
from model.UNet import U_Net
from model.SwinUNETR import SwinUNETR
from trainer import *

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, help='root dir for data')
parser.add_argument('--val_path', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--dataset', type=str, default='MMWHS', help='experiment_name')
parser.add_argument('--list_dir', type=str, help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=7, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=400, help='maximum epoch number to train')
parser.add_argument('--stop_epoch', type=int,
                    default=400, help='epoch to stop training')
parser.add_argument('--batch_size', type=int,
                    default=8, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0005,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=100, help='random seed')
parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model')
parser.add_argument('--ckpt', type=str, help='Pretrained checkpoint')
parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')
parser.add_argument('--checkpoint', type=str, default=None, help='Finetuned lora checkpoint')
parser.add_argument('--lora_rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--warmup', default=True, help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid when warmup is activated')
parser.add_argument('--AdamW', default=True, help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
parser.add_argument('--dice_param', type=float, default=0.8)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--evl_chunk', type=int, default=16)  #  = args.batchsize * args.n_gpus
parser.add_argument('--which_model', type=str, default='SAMmyConv_Adapter_add_ExpertChoiceTokenmoeMLP_Attention_todecoder_topkc=2'])
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--norm_name", default="batch", type=str, help="normalization name")
args = parser.parse_args()


def main():
    args = parser.parse_args()
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    args.is_pretrain = True
    args.exp = dataset_name + '_' + str(args.img_size)
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_' + args.which_model
    snapshot_path = snapshot_path + '_epoch' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_seed' + str(args.seed)
    args.snapshot_path = snapshot_path
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1], args=args)

    for n, value in sam.image_encoder.named_parameters():
        if "Adapter" not in n:
            value.requires_grad = False
        else:
            value.requires_grad = True
    for n, value in sam.prompt_encoder.named_parameters():
        value.requires_grad = False

    net1 = sam.cuda()

    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint, weights_only=True)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_key = k[len("module."):]  
            else:
                new_key = k
            new_state_dict[new_key] = v

        net1.load_state_dict(new_state_dict)

    if args.lora_ckpt is not None:
        net1.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    low_res = img_embedding_size * 4

    trainer = {'MMWHS': trainer_MMWHS}
    trainer[args.dataset](args, net1, args.snapshot_path, multimask_output, low_res)


if __name__ == "__main__":
    main()

