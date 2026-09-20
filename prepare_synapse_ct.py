"""Build the 8-organ "Synapse CT" benchmark (the TransUNet protocol the paper's
Synapse column reports against) from the same raw Synapse/BTCV NIfTI files.

Why this benchmark: it is the one dataset in the paper whose split is public.
The paper says only "18 training cases and 12 testing cases, covering eight
abdominal organs" and "Synapse CT images are resized to 224x224"; the actual
case lists come from TransUNet (github.com/Beckschen/TransUNet,
lists/lists_Synapse), which every paper in this lineage reuses verbatim so
numbers stay comparable. So unlike BTCV and MMWHS -- where the split had to
be guessed -- here every variable is pinned to the published protocol.

Label remap: raw BTCV labels are 1..13; this benchmark scores 8 of them and
treats the other five (esophagus, IVC, portal/splenic vein, both adrenal
glands) as background. Target indices follow the alphabetical organ order
TransUNet-lineage papers report per-class tables in (aorta, gallbladder,
kidney L, kidney R, liver, pancreas, spleen, stomach). Mean DSC is invariant
to how the indices are permuted; the order only matters for reading per-class
rows against published tables.

Windowing is HU [-125, 275] -> [0,1] (TransUNet's abdominal convention, also
what nhan's ct_conversion.py uses). Slices are written at native 512x512 and
resized to img_size later by RandomGenerator/test_single_volume.

Own choice, not from the protocol: training slices whose label is entirely
background (after the 8-organ remap) are dropped, matching what
prepare_btcv.py and nhan's pipeline do. Test volumes keep every slice in the
labeled span so volumetric metrics see a contiguous volume.
"""
import argparse
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

HU_MIN, HU_MAX = -125.0, 275.0

# TransUNet's published case split (lists/lists_Synapse/{train,test_vol}.txt).
TRAIN_CASES = [5, 6, 7, 9, 10, 21, 23, 24, 26, 27, 28, 30, 31, 33, 34, 37, 39, 40]
TEST_CASES = [1, 2, 3, 4, 8, 22, 25, 29, 32, 35, 36, 38]

# raw BTCV label id -> Synapse-CT 8-organ id (absent ids collapse to background)
LABEL_MAP = {
    8: 1,   # aorta
    4: 2,   # gallbladder
    3: 3,   # kidney (left)
    2: 4,   # kidney (right)
    6: 5,   # liver
    11: 6,  # pancreas
    1: 7,   # spleen
    7: 8,   # stomach
}
ORGAN_NAMES = {
    1: "aorta", 2: "gallbladder", 3: "kidney (L)", 4: "kidney (R)",
    5: "liver", 6: "pancreas", 7: "spleen", 8: "stomach",
}


def window(volume):
    return (np.clip(volume, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)


def remap(label):
    out = np.zeros_like(label, dtype=np.float32)
    for src, dst in LABEL_MAP.items():
        out[label == src] = dst
    return out


def load_case(raw_root, n):
    image = nib.load(raw_root / "img" / f"img{n:04d}.nii.gz").get_fdata().astype(np.float32)
    label = nib.load(raw_root / "label" / f"label{n:04d}.nii.gz").get_fdata().astype(np.float32)
    assert image.shape == label.shape, f"case {n}: {image.shape} vs {label.shape}"
    return window(image), remap(label)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path,
                   default=Path("/home/teama/projects/project_01/dataset/raw/synapse/RawData/Training"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/home/teama/projects/project_01/dataset/synapse_ct"))
    args = p.parse_args()

    npz_dir = args.out_dir / "train_npz"
    h5_dir = args.out_dir / "test_vol_h5"
    lists_dir = args.out_dir / "lists"
    for d in (npz_dir, h5_dir, lists_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"{len(TRAIN_CASES)} train / {len(TEST_CASES)} test cases (TransUNet split)")

    train_slices = []
    for n in TRAIN_CASES:
        image, label = load_case(args.raw_root, n)
        kept = 0
        for z in range(image.shape[2]):
            if not label[:, :, z].any():
                continue
            name = f"case{n:04d}_slice{z:04d}"
            np.savez(npz_dir / f"{name}.npz",
                     image=image[:, :, z].astype(np.float32),
                     label=label[:, :, z].astype(np.float32))
            train_slices.append(name)
            kept += 1
        print(f"[train] case{n:04d}: {kept}/{image.shape[2]} slices with 8-organ labels")

    test_names = []
    for n in TEST_CASES:
        image, label = load_case(args.raw_root, n)
        labeled = [z for z in range(label.shape[2]) if label[:, :, z].any()]
        lo, hi = labeled[0], labeled[-1]
        name = f"case{n:04d}"
        with h5py.File(h5_dir / f"{name}.h5", "w") as f:
            f.create_dataset("image", data=image[:, :, lo:hi + 1])
            f.create_dataset("label", data=label[:, :, lo:hi + 1])
        test_names.append(name)
        print(f"[test]  {name}: volume {(512, 512, hi - lo + 1)} (slices {lo}..{hi})")

    (lists_dir / "train.txt").write_text("\n".join(train_slices) + "\n")
    (lists_dir / "val.txt").write_text("\n".join(test_names) + "\n")
    print(f"\ntrain.txt: {len(train_slices)} slices\nval.txt  : {len(test_names)} volumes")


if __name__ == "__main__":
    main()
