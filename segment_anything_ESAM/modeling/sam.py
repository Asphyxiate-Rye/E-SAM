# Stub: modeling/__init__.py re-exports `Sam` because build_sam.py imports
# it, but the vit_b/MMWHS training path (build_sam_vit_b -> _build_sam_my)
# only ever constructs `Sam_my`, never `Sam`. This file/class was never
# present in this repo's git history (checked). Stub only so the package
# import doesn't crash on the dead import.
import torch.nn as nn


class Sam(nn.Module):
    pass
