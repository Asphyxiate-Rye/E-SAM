"""Zero-shot "SAM (1 pt)" baseline from Table 1: the original, unmodified
Meta SAM (not the Sam_my/Adapter/MoE fork), prompted with a single point
sampled from the ground-truth mask, no fine-tuning at all. Evaluated on the
same test_vol_h5 volumes already built for the MoE-SAM runs.

Not specified by the paper (own choices):
  - point sampling: one point drawn uniformly at random from each class's
    foreground pixels *per slice* (paper says "a unified point sampled from
    the ground-truth segmentation mask" but SAM only takes 2D images, and
    every other model in this project already treats these volumes as
    independent 2D slices, so a per-slice point is the natural reading).
  - with only one point, mask ambiguity is high, so multimask_output=True
    (SAM's default recommendation for sparse prompts). Empirically checked
    both selection rules on a handful of BTCV organs before committing:
    picking SAM's own highest-predicted-IoU candidate gave mean Dice 0.44,
    picking the *smallest*-area candidate gave 0.72 -- SAM's IoU score is
    calibrated on natural images and, prompted near the interior of a
    faint-contrast CT organ, consistently over-scores a big candidate that
    bleeds into surrounding tissue. Smallest-area is used here instead.
  - grayscale CT/MRI slices are repeated to 3 channels and scaled to uint8
    (SAM's image encoder expects an RGB-like image).
  - --img-size: matches the MoE-SAM runs' own preprocessing -- the native
    slice (512x512 for CT, native per-patient for ACDC) is downsized to the
    same img_size MoE-SAM trained/evaluated at (224 for Synapse CT, 256 for
    the rest) *before* being handed to SAM, so both baselines see the same
    effective resolution; SAM's own ResizeLongestSide then upsamples that
    to its fixed 1024 working resolution as usual. The predicted mask is
    resized back up (nearest, matching test_single_volume's own convention)
    to native resolution before scoring against the untouched native GT.
"""
import argparse
import math
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
from segment_anything import SamPredictor, sam_model_registry

sys.path.insert(0, "/home/teama/projects/project_01/long/E-SAM")

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


def slice_to_rgb(slice_2d: np.ndarray) -> np.ndarray:
    """[0,1] float grayscale -> uint8 HxWx3, as SAM's image encoder expects."""
    img = np.clip(slice_2d, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    return np.repeat(img[:, :, None], 3, axis=2)


def per_class_metrics(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    if pred.sum() > 0 and gt.sum() > 0:
        return metric.binary.dc(pred, gt), metric.binary.hd(pred, gt), metric.binary.hd95(pred, gt)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0, 0.0
    diag = math.sqrt(sum(s ** 2 for s in gt.shape))
    return 0.0, diag, diag


def sample_point(mask: np.ndarray, rng: random.Random):
    ys, xs = np.nonzero(mask)
    idx = rng.randrange(len(xs))
    return int(xs[idx]), int(ys[idx])  # (x, y) as SamPredictor expects


def resize_image_to(image_rgb: np.ndarray, img_size: int) -> np.ndarray:
    t = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float()
    t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).byte().numpy()


def resize_mask_to(mask: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    t = F.interpolate(t, size=out_hw, mode="nearest")
    return t.squeeze(0).squeeze(0).bool().numpy()


@torch.no_grad()
def predict_slice(predictor: SamPredictor, image_rgb: np.ndarray, point_xy, img_size: int | None):
    native_hw = image_rgb.shape[:2]  # (H, W)
    if img_size is not None and img_size != native_hw[0]:
        # Non-square natives (common in ACDC, e.g. 216x256) need separate
        # x/y scale factors -- a single scale derived from H alone was
        # silently wrong on the W axis for every non-square slice, exactly
        # mirroring RandomGenerator's own zoom(image, (out[0]/x, out[1]/y))
        # in dataset_MMWHS.py, which already scales each axis independently.
        scale_y = img_size / native_hw[0]
        scale_x = img_size / native_hw[1]
        image_for_sam = resize_image_to(image_rgb, img_size)
        point_for_sam = (point_xy[0] * scale_x, point_xy[1] * scale_y)
    else:
        image_for_sam = image_rgb
        point_for_sam = point_xy

    predictor.set_image(image_for_sam)
    point_coords = np.array([[point_for_sam[0], point_for_sam[1]]])
    point_labels = np.array([1])
    masks, scores, _ = predictor.predict(
        point_coords=point_coords, point_labels=point_labels, multimask_output=True,
    )
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    mask = masks[np.argmin(areas)]
    if img_size is not None and img_size != native_hw[0]:
        mask = resize_mask_to(mask, native_hw)
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sam-ckpt", required=True, help="original Meta sam_vit_b_01ec64.pth")
    p.add_argument("--val-path", required=True)
    p.add_argument("--list-dir", required=True)
    p.add_argument("--dataset", required=True, choices=CLASS_NAMES.keys())
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--img-size", type=int, default=None,
                    help="resize slices to this before feeding SAM, matching the MoE-SAM run's own resolution (224 Synapse CT, 256 others); omit to feed native resolution")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    names = CLASS_NAMES[args.dataset]
    if args.dataset == "mmwhs":
        from datasets.dataset_MMWHS import MMWHS_dataset as DatasetClass
    else:
        from datasets.dataset_BTCV import BTCV_dataset as DatasetClass

    sam = sam_model_registry["vit_b"](checkpoint=args.sam_ckpt).cuda().eval()
    predictor = SamPredictor(sam)
    rng = random.Random(args.seed)

    dataset = DatasetClass(base_dir=args.val_path, list_dir=args.list_dir, split="val")
    print(f"{len(dataset)} test volumes")

    all_metrics = []
    for i in range(len(dataset)):
        sample = dataset[i]
        image_vol = sample["image"]  # [Z, H, W], float [0,1]
        label_vol = sample["label"]  # [Z, H, W]
        z = image_vol.shape[0]

        pred_vol = np.zeros_like(label_vol, dtype=np.uint8)
        for zi in range(z):
            image_rgb = slice_to_rgb(image_vol[zi])
            slice_label = label_vol[zi]
            for c in range(1, args.num_classes + 1):
                class_mask = slice_label == c
                if not class_mask.any():
                    continue
                point_xy = sample_point(class_mask, rng)
                mask = predict_slice(predictor, image_rgb, point_xy, args.img_size)
                # multiple classes can claim the same pixel; last write wins,
                # matching how these single-point-per-class baselines are
                # scored (each class against its own binary mask below, not
                # against a combined argmax map -- see per_class_metrics).
                pred_vol[zi][mask] = c

        case_metrics = []
        for c in range(1, args.num_classes + 1):
            d, hd, hd95 = per_class_metrics(pred_vol == c, label_vol == c)
            case_metrics.append((d, hd, hd95))
        all_metrics.append(case_metrics)

        case_name = sample.get("case_name", f"case_{i}")
        dice_str = ", ".join(f"{names[c]}={case_metrics[c-1][0]:.3f}" for c in range(1, args.num_classes + 1))
        print(f"[{case_name}] {dice_str}", flush=True)

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
