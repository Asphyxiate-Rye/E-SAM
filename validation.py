import argparse
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch.backends.cudnn as cudnn

from model.SAM import SAM
from model.sam_lora_image_encoder import LoRA_Sam
from segment_anything_ESAM import sam_model_registry
from collections import Counter
from datasets.dataset_MMWHS import *
from model.SwinUNETR import SwinUNETR
from utils import *
from trainer import *

parser = argparse.ArgumentParser()
parser.add_argument('--val_path', type=str, help='root dir for data')
parser.add_argument('--output', type=str)
parser.add_argument('--dataset', type=str, default='MMWHS', help='experiment_name')
parser.add_argument('--list_dir', type=str, help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=7, help='output channel of network')
parser.add_argument('--batch_size', type=int,
                    default=1, help='batch_size per gpu')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=2345, help='random seed')
parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model')
parser.add_argument('--ckpt', type=str,
                    help='Pretrained checkpoint')
parser.add_argument('--nii_save_path', type=str)
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--lora_rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--evl_chunk', type=int, default=8)
parser.add_argument('--num_points', type=int, default=1)
parser.add_argument('--which_model', type=str, default='MYVAL_ESAM')
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

    db_test = MMWHS_dataset(base_dir=args.val_path, list_dir=args.list_dir, split='val')
    valloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1], args=args)

    sam.eval()

    if args.ckpt is not None:
        with open(args.ckpt, "rb") as f:
            state_dict = torch.load(f, weights_only=True)

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_key = k[len("module."):] 
            else:
                new_key = k
            new_state_dict[new_key] = v

        sam.load_state_dict(new_state_dict, strict=False)

    model = sam.cuda()

    model.eval()
    metric_list = []

    for i_batch, sampled_batch in enumerate(valloader):
        image, label, case_name = sampled_batch['image'].cuda(), sampled_batch['label'], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, args, classes=args.num_classes,
                                          multimask_output=True,
                                          patch_size=[args.img_size, args.img_size],
                                          input_size=[args.img_size, args.img_size],
                                         nii_save_path=args.nii_save_path, case=case_name)

        metric_i_tensor = torch.tensor(metric_i, device=image.device)
        metric_list.append(metric_i_tensor.cpu().numpy())

    metric_list = np.array(metric_list)
    metric_avg = np.mean(metric_list, axis=0)
    performance = np.mean(metric_avg, axis=0)
    print('mean_dice', performance[0])
    print('hd', performance[1])
    logging.info(f'mean_dice {performance[0]}')
    logging.info(f'hd {performance[1]}')

if __name__ == "__main__":
    main()

