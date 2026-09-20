"""Full-metric (Dice, HD, HD95) evaluation of a trained MMWHS checkpoint on
the 4 held-out test volumes, mirroring utils.py's test_single_volume chunked
inference but also computing HD/HD95 (the original test_single_volume only
returns Dice; calculate_metric_percase there has HD95 commented out)."""
import argparse
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from medpy import metric

sys.path.insert(0, "/home/teama/projects/project_01/long/E-SAM")
from datasets.dataset_MMWHS import MMWHS_dataset
from segment_anything_ESAM import sam_model_registry

CLASS_NAMES = {
    1: "LV myocardium",
    2: "LA blood cavity",
    3: "LV blood cavity",
    4: "RA blood cavity",
    5: "RV blood cavity",
    6: "Ascending aorta",
    7: "Pulmonary artery",
}


def per_class_metrics(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd = metric.binary.hd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd, hd95
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0, 0.0
    # exactly one empty: undefined distance: penalize with the volume's
    # spatial diagonal, matching this project's convention elsewhere for
    # this edge case (finite penalty rather than inf/NaN).
    diag = math.sqrt(sum(s ** 2 for s in gt.shape))
    return 0.0, diag, diag


@torch.no_grad()
def eval_volume(net, image, label, classes, img_size, evl_chunk=16):
    image, label = image.squeeze(0), label.squeeze(0)  # [Z,H,W]
    z, x, y = image.shape
    prediction = torch.zeros_like(image, dtype=torch.long)
    buoy = 0
    net.eval()
    while buoy < z:
        end = min(buoy + evl_chunk, z)
        slices = image[buoy:end, :, :].unsqueeze(1)
        if slices.shape[-2:] != (img_size, img_size):
            slices = F.interpolate(slices, size=(img_size, img_size), mode="bilinear")
        inputs = repeat(slices, "b c h w -> b (repeat c) h w", repeat=3).cuda()
        outputs = net(inputs, True, img_size, None)
        out = torch.argmax(torch.softmax(outputs["masks"], dim=1), dim=1)
        if (x, y) != (out.shape[1], out.shape[2]):
            out = F.interpolate(out.unsqueeze(1).float(), (x, y), mode="nearest").squeeze(1).long()
        prediction[buoy:end] = out.cpu()
        buoy = end

    pred_np = prediction.numpy()
    label_np = label.numpy()
    return [per_class_metrics(pred_np == c, label_np == c) for c in range(1, classes + 1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="model_best.pth (finetuned weights)")
    p.add_argument("--sam-ckpt", required=True, help="original sam_vit_b checkpoint (build_sam_vit_b returns None without one; overwritten right after by --ckpt)")
    p.add_argument("--val-path", required=True)
    p.add_argument("--list-dir", required=True)
    p.add_argument("--num-classes", type=int, default=7)
    p.add_argument("--img-size", type=int, default=256)
    args = p.parse_args()

    class Args:
        batch_size = 1

    net, _ = sam_model_registry["vit_b"](
        image_size=args.img_size, num_classes=args.num_classes,
        checkpoint=args.sam_ckpt, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1], args=Args(),
    )
    state_dict = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)
    net = net.cuda()

    dataset = MMWHS_dataset(base_dir=args.val_path, list_dir=args.list_dir, split="val")
    print(f"{len(dataset)} test volumes")

    all_metrics = []  # [num_volumes][num_classes] of (dice, hd, hd95)
    for i in range(len(dataset)):
        sample = dataset[i]
        image = torch.from_numpy(sample["image"]).float().unsqueeze(0)
        label = torch.from_numpy(sample["label"]).float().unsqueeze(0)
        case_metrics = eval_volume(net, image, label, args.num_classes, args.img_size)
        all_metrics.append(case_metrics)
        case_name = sample.get("case_name", f"case_{i}")
        dice_str = ", ".join(f"{CLASS_NAMES[c]}={case_metrics[c-1][0]:.3f}" for c in range(1, args.num_classes + 1))
        print(f"[{case_name}] {dice_str}")

    arr = np.array(all_metrics)  # [volumes, classes, 3]
    per_class_mean = arr.mean(axis=0)  # [classes, 3]

    print("\n=== Per-class mean over 4 test volumes ===")
    for c in range(1, args.num_classes + 1):
        d, hd, hd95 = per_class_mean[c - 1]
        print(f"  {CLASS_NAMES[c]:20s} DSC={d:.4f}  HD={hd:.3f}  HD95={hd95:.3f}")

    overall = per_class_mean.mean(axis=0)
    print("\n=== Overall (mean over classes and volumes) ===")
    print(f"  DSC  = {overall[0]:.4f}")
    print(f"  HD   = {overall[1]:.3f}")
    print(f"  HD95 = {overall[2]:.3f}")


if __name__ == "__main__":
    main()
