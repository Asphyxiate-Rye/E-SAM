import argparse
import logging
import os
import random
import sys
import time
import math
import SimpleITK as sitk

from collections import Counter
import numpy as np
import torch
import monai
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, CosineEmbeddingLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from torchvision import transforms
from icecream import ic
from PIL import Image
def torch2D_Hausdorff_distance(x,y): # Input be like (Batch,width,height)
    x = x.float()
    y = y.float()
    distance_matrix = torch.cdist(x,y,p=2) # p=2 means Euclidean Distance
    
    value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
    value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]
    
    value = torch.cat((value1, value2), dim=1)
    
    return value.max(1)[0]


def calc_loss_init_size(outputs, label_batch, ce_loss, dice_loss, dice_weight: float = 0.8):
    logits = outputs['masks']
    loss_ce = ce_loss(logits, label_batch[:].long())
    loss_dice = dice_loss(logits, label_batch, softmax=True)
    loss = ((1 - dice_weight) * loss_ce + dice_weight * loss_dice)
    return loss, loss_ce, loss_dice

def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = ((1 - dice_weight) * loss_ce + dice_weight * loss_dice)
    return loss, loss_ce, loss_dice

def trainer_MMWHS(args, model, snapshot_path, multimask_output, low_res):
    from datasets.dataset_MMWHS import MMWHS_dataset, RandomGenerator, Sampler
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    db_train = MMWHS_dataset(base_dir=args.root_path, list_dir=args.list_dir, split='train',
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size],
                                                    low_res=[low_res, low_res])
                                    ]))

    db_test = MMWHS_dataset(base_dir=args.val_path, list_dir=args.list_dir, split='val')

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999),
                                weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9,
                              weight_decay=0.0001)
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. Total {} iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    best_performance = 0.0
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            b = time.time()
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
            optimizer.zero_grad()
            outputs = model(image_batch, multimask_output, args.img_size, gt=low_res_label_batch)
            loss, loss_ce1, loss_dice1 = calc_loss_init_size(outputs, label_batch, ce_loss, dice_loss,
                                                             dice_weight=args.dice_param)
            loss.backward()
            optimizer.step()

            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (
                            1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            c = time.time()
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce1', loss_ce1, iter_num)
            writer.add_scalar('info/loss_dice1', loss_dice1, iter_num)
            logging.info('iteration %d : loss : %f, loss_ce1: %f, loss_dice1: %f, time: %f' % (
            iter_num, loss.item(), loss_ce1.item(), loss_dice1.item(), c - b))
        if (epoch_num + 1) % args.save_interval == 0:
            model.eval()
            metric_list = []
            for i_batch, sampled_batch in enumerate(valloader):
                image, label, case_name = sampled_batch['image'].cuda(), sampled_batch['label'], \
                sampled_batch['case_name'][0]
                metric_i = test_single_volume(image, label, model, args, classes=args.num_classes,
                                              multimask_output=multimask_output,
                                              patch_size=[args.img_size, args.img_size],
                                              input_size=[args.img_size, args.img_size],
                                              test_save_path=snapshot_path + '/val', case=case_name,
                                              z_spacing=1, mode='test')
                metric_i_tensor = torch.tensor(metric_i, device=image.device)
                metric_list.append(metric_i_tensor.cpu().numpy())

            metric_list = np.array(metric_list)
            metric_avg = np.mean(metric_list, axis=0)
            performance = np.mean(metric_avg, axis=0)
            logging.info(f'mean_dice {performance}')
            writer.add_scalar('info/performance', performance, iter_num)
            if performance > best_performance:
                best_performance = performance
                logging.info(f'Testing performance in best val model: mean_dice: {best_performance}')
                save_mode_path = os.path.join(snapshot_path, 'model_best.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info(f"save model to {save_mode_path}")

        save_now_epoch_path = os.path.join(snapshot_path, 'model_now.pth')
        torch.save(model.state_dict(), save_now_epoch_path)

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            logging.info("the best_performance is {}".format(best_performance))
            logging.info('{}'.format(args.which_model))
            iterator.close()

    return "Training Finished!"