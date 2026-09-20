# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# predictor.py/automatic_mask_generator.py (interactive inference, not used
# by training) are dropped: predictor.py imports a `segment_anything`
# top-level package that isn't part of this repo and isn't installed here.
# train.py only needs sam_model_registry.
from .build_sam import (
    build_sam,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    sam_model_registry,
)
