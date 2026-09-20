"""Full-metric (Dice, HD, HD95) evaluation of a trained checkpoint on its
held-out test volumes. Generalizes eval_mmwhs_metrics.py to also cover the
BTCV_dataset-based runs (13-organ BTCV, 8-organ Synapse CT)."""
import argparse
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from medpy import metric

sys.path.insert(0, "/home/teama/projects/project_01/long/E-SAM")
from segment_anything_ESAM import sam_model_registry

CLASS_NAMES = {
    "mmwhs": {
        1: "LV myocardium", 2: "LA blood cavity", 3: "LV blood cavity",
        4: "RA blood cavity", 5: "RV blood cavity", 6: "Ascending aorta",
        7: "Pulmonary artery",
    },
    "btcv": {
        1: "spleen", 2: "right kidney", 3: "left kidney", 4: "gallbladder",
        5: "esophagus", 6: "liver", 7: "stomach", 8: "aorta",
        9: "IVC", 10: "portal/splenic vein", 11: "pancreas",
        12: "right adrenal", 13: "left adrenal",
    },
    "synapse_ct": {
        1: "aorta", 2: "gallbladder", 3: "kidney (L)", 4: "kidney (R)",
        5: "liver", 6: "pancreas", 7: "spleen", 8: "stomach",
    },
    "acdc": {
        1: "right ventricle", 2: "myocardium", 3: "left ventricle",
    },
}


def per_class_metrics(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if pred.sum() > 0 and gt.sum() > 0:
        return metric.binary.dc(pred, gt), metric.binary.hd(pred, gt), metric.binary.hd95(pred, gt)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0, 0.0
    diag = math.sqrt(sum(s ** 2 for s in gt.shape))
    return 0.0, diag, diag


@torch.no_grad()
def eval_volume(net, image, label, classes, img_size, evl_chunk=16):
    image, label = image.squeeze(0), label.squeeze(0)
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
    pred_np, label_np = prediction.numpy(), label.numpy()
    return [per_class_metrics(pred_np == c, label_np == c) for c in range(1, classes + 1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sam-ckpt", required=True)
    p.add_argument("--val-path", required=True)
    p.add_argument("--list-dir", required=True)
    p.add_argument("--dataset", required=True, choices=CLASS_NAMES.keys())
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--img-size", type=int, required=True)
    p.add_argument("--evl-chunk", type=int, default=16, help="lower this if GPU is shared with a concurrent training run")
    args = p.parse_args()

    names = CLASS_NAMES[args.dataset]

    if args.dataset == "mmwhs":
        from datasets.dataset_MMWHS import MMWHS_dataset as DatasetClass
    else:
        from datasets.dataset_BTCV import BTCV_dataset as DatasetClass

    class Args:
        batch_size = 1

    net, _ = sam_model_registry["vit_b"](
        image_size=args.img_size, num_classes=args.num_classes,
        checkpoint=args.sam_ckpt, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1], args=Args(),
    )
    net.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    net = net.cuda()

    dataset = DatasetClass(base_dir=args.val_path, list_dir=args.list_dir, split="val")
    print(f"{len(dataset)} test volumes")

    all_metrics = []
    for i in range(len(dataset)):
        sample = dataset[i]
        image = torch.from_numpy(sample["image"]).float().unsqueeze(0)
        label = torch.from_numpy(sample["label"]).float().unsqueeze(0)
        case_metrics = eval_volume(net, image, label, args.num_classes, args.img_size, evl_chunk=args.evl_chunk)
        all_metrics.append(case_metrics)
        case_name = sample.get("case_name", f"case_{i}")
        dice_str = ", ".join(f"{names[c]}={case_metrics[c-1][0]:.3f}" for c in range(1, args.num_classes + 1))
        print(f"[{case_name}] {dice_str}")

    arr = np.array(all_metrics)
    per_class_mean = arr.mean(axis=0)

    print(f"\n=== Per-class mean over {len(dataset)} test volumes ===")
    for c in range(1, args.num_classes + 1):
        d, hd, hd95 = per_class_mean[c - 1]
        print(f"  {names[c]:20s} DSC={d:.4f}  HD={hd:.3f}  HD95={hd95:.3f}")

    overall = per_class_mean.mean(axis=0)
    print("\n=== Overall (mean over classes and volumes) ===")
    print(f"  DSC  = {overall[0]:.4f}")
    print(f"  HD   = {overall[1]:.3f}")
    print(f"  HD95 = {overall[2]:.3f}")


if __name__ == "__main__":
    main()
