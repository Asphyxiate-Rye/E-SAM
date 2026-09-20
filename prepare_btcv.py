"""Convert Synapse/BTCV NIfTI volumes into the npz/h5 layout E-SAM's
dataset classes expect, using EXACTLY the case split that
nhan/moe_research/manifests/synapse_btcv_v2.json already froze.

Point of this script: run the (patched) upstream E-SAM code on the same
data, same split, same 13 organs that nhan's framework produced
E0=64.50 / E3=58.46 volumetric Dice on, so that any difference in result
is attributable to the model/training code alone.

Preprocessing deliberately mirrors nhan's scripts/data/ct_conversion.py
rather than E-SAM's own dataset.py:
  - HU window [-125, 275] -> [0,1] (TransUNet's abdominal-CT convention,
    which nhan uses). E-SAM's dataset.py hardcodes (x+750)/1500, a cardiac
    window suited to MMWHS, and upstream ships no BTCV code at all.
  - Native 512x512 kept on disk; the resize to img_size happens later in
    RandomGenerator/test_single_volume, exactly like nhan resizes in
    src/data/transforms.py.
  - Labels are already integers 1..13 in the raw data, so unlike MMWHS
    (205/420/...) there is nothing to remap.

Test volumes span [min, max] labeled slice index of each case, which is the
same span nhan's volumetric evaluation reconstructs (it drops all-background
slices at conversion time, then stacks what remains at its true index).
"""
import argparse
import json
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

HU_MIN, HU_MAX = -125.0, 275.0


def window(volume):
    return (np.clip(volume, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)


def case_number(case_id):
    """'synapse_0001' -> '0001'."""
    return case_id.split("_")[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path,
                   default=Path("/home/teama/projects/project_01/dataset/raw/synapse/RawData/Training"))
    p.add_argument("--manifest", type=Path,
                   default=Path("/home/teama/projects/project_01/nhan/moe_research/manifests/synapse_btcv_v2.json"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/home/teama/projects/project_01/dataset/btcv_esam"))
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    train_cases = sorted({r["case_id"] for r in manifest["training"]})
    test_cases = sorted({r["case_id"] for r in manifest["test"]})
    print(f"{len(train_cases)} train / {len(test_cases)} test cases (from {args.manifest.name})")

    npz_dir = args.out_dir / "train_npz"
    h5_dir = args.out_dir / "test_vol_h5"
    lists_dir = args.out_dir / "lists"
    for d in (npz_dir, h5_dir, lists_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_slices = []
    for case_id in train_cases:
        n = case_number(case_id)
        image = nib.load(args.raw_root / "img" / f"img{n}.nii.gz").get_fdata().astype(np.float32)
        label = nib.load(args.raw_root / "label" / f"label{n}.nii.gz").get_fdata().astype(np.float32)
        assert image.shape == label.shape, f"{case_id}: {image.shape} vs {label.shape}"
        image = window(image)

        kept = 0
        for z in range(image.shape[2]):
            if not label[:, :, z].any():
                continue
            name = f"{case_id}_slice{z:04d}"
            np.savez(npz_dir / f"{name}.npz",
                     image=image[:, :, z].astype(np.float32),
                     label=label[:, :, z].astype(np.float32))
            train_slices.append(name)
            kept += 1
        print(f"[train] {case_id}: {kept}/{image.shape[2]} labeled slices")

    test_names = []
    for case_id in test_cases:
        n = case_number(case_id)
        image = nib.load(args.raw_root / "img" / f"img{n}.nii.gz").get_fdata().astype(np.float32)
        label = nib.load(args.raw_root / "label" / f"label{n}.nii.gz").get_fdata().astype(np.float32)
        image = window(image)

        labeled = [z for z in range(label.shape[2]) if label[:, :, z].any()]
        lo, hi = labeled[0], labeled[-1]
        image_hwz = image[:, :, lo:hi + 1]
        label_hwz = label[:, :, lo:hi + 1]
        with h5py.File(h5_dir / f"{case_id}.h5", "w") as f:
            f.create_dataset("image", data=image_hwz)
            f.create_dataset("label", data=label_hwz)
        test_names.append(case_id)
        print(f"[test]  {case_id}: volume {image_hwz.shape} (slices {lo}..{hi})")

    (lists_dir / "train.txt").write_text("\n".join(train_slices) + "\n")
    (lists_dir / "val.txt").write_text("\n".join(test_names) + "\n")
    print(f"\ntrain.txt: {len(train_slices)} slices\nval.txt  : {len(test_names)} volumes")


if __name__ == "__main__":
    main()
