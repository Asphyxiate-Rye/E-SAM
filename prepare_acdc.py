"""Build ACDC (cardiac MRI, RV/myocardium/LV) into the npz/h5 layout
E-SAM's Dataset classes expect. No ACDC code exists anywhere in the E-SAM
repo's history (same situation as BTCV), so this is written from scratch.

Split: the official ACDC challenge packaging already partitions all 150
labeled patients into training/ (100) and testing/ (50) folders -- used
verbatim here rather than inventing a split, since it is the one boundary
in this dataset that is not arbitrary.

Each patient contributes 2 labeled 3D frames (ED and ES, named in Info.cfg),
each a short-axis stack of only ~5-15 slices -- unlike the ~100-300 slice CT
volumes, so treating every frame as one "case" for train/test is standard
practice in this literature, not a shortcut.

Normalization: confirmed by the MoE-SAM authors by email on 2026-09-07:
"ACDC: min-max normalization per loaded sample -- per slice during training
and per volume during evaluation." What is loaded per training example is
one 2D slice (see the npz branch of convert_split below), so training
normalization here is per-slice min-max, computed independently for each
slice rather than from the whole 3D volume's min/max; what is loaded per
evaluation example is the full 3D frame (the h5 branch), so evaluation
normalization is per-volume min-max over that frame. An earlier pass instead
normalized once over the whole 3D volume before slicing (so every training
slice from one volume shared that volume's min/max, not its own) and added
a [0.5, 99.5] percentile clip before the min-max, neither of which the
authors' reply describes; both are removed here to match their recipe.
"""
import argparse
import re
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np


def normalize(array):
    """Plain min-max to [0,1], over whatever array is passed in.

    The caller decides the scope: convert_split calls this per 2D slice for
    training data and per 3D volume for evaluation data, matching the
    authors' "per loaded sample" description. A degenerate all-constant
    array (max == min) would otherwise divide by zero; it normalizes to all
    zeros instead, which only affects synthetic edge cases (a slice/volume
    with a single intensity value throughout).
    """
    lo, hi = float(array.min()), float(array.max())
    if hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    # float64 min/max on a float32 array upcasts the result silently, which
    # then crashes Conv2d at eval time ("Input type (double) and bias type
    # (float) should be the same") for the h5 volumes this writes;
    # RandomGenerator's own explicit cast hides this for the training npz
    # path, but nothing casts back for the val h5 path since it is read
    # with no transform.
    return ((array - lo) / (hi - lo)).astype(np.float32)


def find_frames(patient_dir: Path):
    """Return [(frame_tag, image_path, label_path), ...] for ED and ES."""
    cfg = (patient_dir / "Info.cfg").read_text()
    ed = int(re.search(r"ED:\s*(\d+)", cfg).group(1))
    es = int(re.search(r"ES:\s*(\d+)", cfg).group(1))
    frames = []
    for tag, num in (("ED", ed), ("ES", es)):
        image_path = patient_dir / f"{patient_dir.name}_frame{num:02d}.nii.gz"
        label_path = patient_dir / f"{patient_dir.name}_frame{num:02d}_gt.nii.gz"
        if image_path.is_file() and label_path.is_file():
            frames.append((tag, image_path, label_path))
    return frames


def convert_split(patients_root: Path, patient_dirs, npz_dir=None, h5_dir=None):
    slice_names, volume_names = [], []
    for patient_dir in sorted(patient_dirs):
        for tag, image_path, label_path in find_frames(patient_dir):
            raw_image = nib.load(image_path).get_fdata().astype(np.float32)
            label = nib.load(label_path).get_fdata().astype(np.float32)
            case_id = f"{patient_dir.name}_{tag}"

            if npz_dir is not None:
                kept = 0
                for z in range(raw_image.shape[2]):
                    if not label[:, :, z].any():
                        continue
                    # Normalized per slice, independently of the rest of the
                    # volume, matching what a training example actually is.
                    image_slice = normalize(raw_image[:, :, z])
                    name = f"{case_id}_slice{z:02d}"
                    np.savez(npz_dir / f"{name}.npz",
                             image=image_slice, label=label[:, :, z])
                    slice_names.append(name)
                    kept += 1
                print(f"[train] {case_id}: {kept}/{raw_image.shape[2]} labeled slices")
            else:
                # Normalized once over the whole 3D frame, matching what an
                # evaluation example actually is.
                image = normalize(raw_image)
                with h5py.File(h5_dir / f"{case_id}.h5", "w") as f:
                    f.create_dataset("image", data=image)
                    f.create_dataset("label", data=label)
                volume_names.append(case_id)
                print(f"[test]  {case_id}: volume {image.shape}")
    return slice_names, volume_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path,
                   default=Path("/home/teama/projects/project_01/dataset/raw/database_acdc"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/home/teama/projects/project_01/dataset/acdc_esam"))
    args = p.parse_args()

    npz_dir = args.out_dir / "train_npz"
    h5_dir = args.out_dir / "test_vol_h5"
    lists_dir = args.out_dir / "lists"
    for d in (npz_dir, h5_dir, lists_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_patients = sorted((args.raw_root / "training").glob("patient*"))
    test_patients = sorted((args.raw_root / "testing").glob("patient*"))
    print(f"{len(train_patients)} train patients / {len(test_patients)} test patients (official ACDC split)")

    train_slices, _ = convert_split(args.raw_root, train_patients, npz_dir=npz_dir)
    _, test_volumes = convert_split(args.raw_root, test_patients, h5_dir=h5_dir)

    (lists_dir / "train.txt").write_text("\n".join(train_slices) + "\n")
    (lists_dir / "val.txt").write_text("\n".join(test_volumes) + "\n")
    print(f"\ntrain.txt: {len(train_slices)} slices\nval.txt  : {len(test_volumes)} volumes (frames)")


if __name__ == "__main__":
    main()
