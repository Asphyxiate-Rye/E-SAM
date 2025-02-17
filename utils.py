import os
import numpy as np
import torch
import random
import math
from scipy.ndimage.interpolation import zoom
from medpy import metric
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import imageio
from einops import repeat
from icecream import ic
import csv
import time


def get_bbox(mask, bbox_shift=3):
    y_indices, x_indices = torch.where(mask > 0)
    x_min, x_max = torch.min(x_indices), torch.max(x_indices)
    y_min, y_max = torch.min(y_indices), torch.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(H, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(W, y_max + bbox_shift)
    bboxes = np.array([x_min, y_min, x_max, y_max])
    return bboxes

def get_point(mask):
    indices = torch.where(mask > 0)
    if len(indices[0]) > 0:
        random_idx = random.randint(0, len(indices[0]) - 1)
        random_point = (indices[1][random_idx], indices[0][random_idx])
    return random_point


def get_points(mask, num_points=3):
    indices = torch.where(mask > 0)
    points = []

    if len(indices[0]) > 0:
        n = min(num_points, len(indices[0]))
        random_indices = random.sample(range(len(indices[0])), n)
        points = [(indices[1][idx].item(), indices[0][idx].item())
                  for idx in random_indices]

        while len(points) < num_points:
            points.append(points[-1])
        assert len(points) == num_points

    return points


def get_centroid(mask):
    indices = torch.where(mask > 0)
    if len(indices[0]) > 0:
        centroid_x = indices[1].float().mean().item()
        centroid_y = indices[0].float().mean().item()
        return [(int(centroid_x), int(centroid_y))]
    else:
        return None


class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels, softmax=None):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        # hd95 = metric.binary.hd95(pred, gt)
        return dice
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1
    else:
        return 0

def calculate_metric_percase_val(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, args, classes, multimask_output, patch_size=[256, 256], input_size=[224, 224],
                       test_save_path=None, case=None, z_spacing=1, mode='test'):
    lab = label.squeeze(0)
    image, label = image.squeeze(0), label.squeeze(0)
    if len(image.shape) == 3:
        buoy = 0
        if args.evl_chunk:
            evl_ch = int(args.evl_chunk)
        else:
            evl_ch = int(image.shape[0])

        result = math.ceil(image.shape[0] / evl_ch) * evl_ch
        z, x, y = image.shape[0], image.shape[1], image.shape[2]
        prediction = torch.zeros([image.shape[0], image.shape[1], image.shape[2]])

        while (buoy + evl_ch) < image.shape[0]:
            slices = image[buoy:buoy + evl_ch, :, :].unsqueeze(1)
            if x != input_size[0] or y != input_size[1]:
                slices = F.interpolate(slices, size=(input_size[0], input_size[1]), mode='bilinear')
            new_x, new_y = slices.shape[2], slices.shape[3]  # [input_size[0], input_size[1]]
            if new_x != patch_size[0] or new_y != patch_size[1]:
                slices = F.interpolate(slices, size=(input_size[0], input_size[1]), mode='bilinear')
            inputs = repeat(slices, 'b c h w -> b (repeat c) h w', repeat=3)
            net.eval()

            with torch.no_grad():
                outputs = net(inputs, multimask_output, patch_size[0], None)
                output_masks = outputs['masks']

                out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1)
                out_h, out_w = out.shape[1], out.shape[2]
                if x != out_h or y != out_w:
                    pred = F.interpolate(out.unsqueeze(1).float(), (x, y), mode='nearest').squeeze(1).long()
                else:
                    pred = out
                prediction[buoy:buoy + evl_ch, :, :] = pred
            buoy += evl_ch

        slices = image[image.shape[0]-evl_ch:image.shape[0], ...].unsqueeze(1)
        if x != input_size[0] or y != input_size[1]:
            slices = F.interpolate(slices, size=(input_size[0], input_size[1]), mode='bilinear')
        inputs = repeat(slices, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0], None)
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1)
            out_h, out_w = out.shape[1], out.shape[2]
            if x != out_h or y != out_w:
                pred = F.interpolate(out.unsqueeze(1).float(), (x, y), mode='nearest').squeeze(1).long()
            else:
                pred = out
            prediction[image.shape[0]-evl_ch:, :, :] = pred
    metric_list = []
    metric_list_dice = []
    prediction = prediction.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    for i in range(1, classes + 1):
        metric_list_dice.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list_dice


def mask_latent_code_spatial_wise(latent_code, loss, percentile=1 / 3.0, random=False, loss_type='corr', if_detach=True, if_soft=False):
    '''
    given a latent code return a perturbed code where top % areas are masked 
    '''
    use_gpu = True if latent_code.device != torch.device('cpu') else False
    code = latent_code
    num_images = code.size(0)
    spatial_size = code.size(2) * code.size(3)
    H, W = code.size(2), code.size(3)

    gradient = torch.autograd.grad(loss, [code])[0]
    # mask gradient with largest response:
    spatial_mean = torch.mean(gradient, dim=1, keepdim=True)
    spatial_mean = spatial_mean.squeeze().view(num_images, spatial_size)

    # select the threshold at top XX percentile
    if random:
        percentile = np.random.rand() * percentile

    vector_thresh_percent = int(spatial_size * percentile)
    vector_thresh_value = torch.sort(spatial_mean, dim=1, descending=True)[
        0][:, vector_thresh_percent]

    vector_thresh_value = vector_thresh_value.view(
        num_images, 1).expand(num_images, spatial_size)

    if if_soft:
        vector = torch.where(spatial_mean > vector_thresh_value,
                             0.5 * torch.rand_like(spatial_mean),
                             torch.ones_like(spatial_mean))
    else:
        vector = torch.where(spatial_mean > vector_thresh_value,
                             torch.zeros_like(spatial_mean),
                             torch.ones_like(spatial_mean))

    mask_all = vector.view(num_images, 1, H, W)
    if not if_detach:
        masked_latent_code = latent_code * mask_all
    else:
        masked_latent_code = code * mask_all

    try:
        decoder_function.zero_grad()
    except:
        pass
    return masked_latent_code, mask_all

def set_grad(module, requires_grad=False):
    for p in module.parameters():  # reset requires_grad
        p.requires_grad = requires_grad
