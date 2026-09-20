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

Normalization: ACDC is MRI, not CT, so there is no fixed HU scale to window
against. Per-volume min-max to [0,1] (own choice; the paper doesn't specify)
is the simplest standard choice.
"""
import argparse
import re
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np


def normalize(volume):
    """Percentile-clip then min-max to [0,1].

    Raw ACDC intensities are inconsistent across patients/sites: some
    volumes are already rescaled to [0,255], others keep native scanner
    units up to ~2300 with a handful of bright outlier voxels (partial
    volume / vessel). A first pass used plain per-volume min-max, which is
    not robust to those outliers -- for patients where the true max is
    2-3x the 99th percentile, min-max crushes all real anatomy into a
    small fraction of [0,1] while wasting the rest of the range on a few
    outlier pixels, adding uncontrolled per-patient contrast noise. This
    is the standard nnU-Net/MRI-pipeline fix: clip to [0.5, 99.5]
    percentile first so a handful of extreme voxels can't dominate the
    scale, then min-max the clipped range.
    """
    lo, hi = np.percentile(volume, [0.5, 99.5])
    if hi <= lo:
        return np.zeros_like(volume)
    # np.percentile returns float64 scalars, which silently upcasts the
    # float32 `volume` array below to float64 -- the h5 test volumes this
    # writes then crash Conv2d at eval time ("Input type (double) and bias
    # type (float) should be the same"); RandomGenerator's own explicit
    # cast hid this for the training npz path, but nothing casts back for
    # the val h5 path since it's read with no transform.
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


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
            image = normalize(nib.load(image_path).get_fdata().astype(np.float32))
            label = nib.load(label_path).get_fdata().astype(np.float32)
            case_id = f"{patient_dir.name}_{tag}"

            if npz_dir is not None:
                kept = 0
                for z in range(image.shape[2]):
                    if not label[:, :, z].any():
                        continue
                    name = f"{case_id}_slice{z:02d}"
                    np.savez(npz_dir / f"{name}.npz",
                             image=image[:, :, z], label=label[:, :, z])
                    slice_names.append(name)
                    kept += 1
                print(f"[train] {case_id}: {kept}/{image.shape[2]} labeled slices")
            else:
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
