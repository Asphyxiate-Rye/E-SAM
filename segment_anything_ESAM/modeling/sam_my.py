# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import time

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple
from .image_encoder import ImageEncoderViT, ImageEncoderViT_features
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .multiscale_CNN import CNN
from model.SwinUNETR import SwinUNETR
from model.MoE import *
from model.ExpertChoiceTokenConvLoRAMoE import *


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout=0.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _separate_heads(self, x, num_heads):
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x):
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q, k, v):
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Linear layers for query, key, value (separately for X_l and X_d)
        self.wq_l = nn.Linear(dim, dim, bias=qkv_bias)  # Query for shallow (X_l)
        self.wk_d = nn.Linear(dim, dim, bias=qkv_bias)  # Key for deep (X_d)
        self.wv_d = nn.Linear(dim, dim, bias=qkv_bias)  # Value for deep (X_d)

        self.wq_d = nn.Linear(dim, dim, bias=qkv_bias)  # Query for deep (X_d)
        self.wk_l = nn.Linear(dim, dim, bias=qkv_bias)  # Key for shallow (X_l)
        self.wv_l = nn.Linear(dim, dim, bias=qkv_bias)  # Value for shallow (X_l)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_l = nn.Linear(dim, dim)
        self.proj_d = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_l, x_d):
        """
        x_l: shallow feature (batch, seq_len_l, dim)
        x_d: deep feature (batch, seq_len_d, dim)
        """
        B, N_l, C = x_l.shape
        _, N_d, _ = x_d.shape

        # 1. Shallow feature (X_l) as Query, Deep feature (X_d) as Key and Value
        q_l = self.wq_l(x_l).reshape(B, N_l, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k_d = self.wk_d(x_d).reshape(B, N_d, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v_d = self.wv_d(x_d).reshape(B, N_d, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Compute attention for shallow -> deep
        attn_l_to_d = (q_l @ k_d.transpose(-2, -1)) * self.scale
        attn_l_to_d = attn_l_to_d.softmax(dim=-1)
        attn_l_to_d = self.attn_drop(attn_l_to_d)
        out_l = (attn_l_to_d @ v_d).transpose(1, 2).reshape(B, N_l, C)
        out_l = self.proj_l(out_l)
        out_l = self.proj_drop(out_l)

        # 2. Deep feature (X_d) as Query, Shallow feature (X_l) as Key and Value
        q_d = self.wq_d(x_d).reshape(B, N_d, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k_l = self.wk_l(x_l).reshape(B, N_l, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v_l = self.wv_l(x_l).reshape(B, N_l, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Compute attention for deep -> shallow
        attn_d_to_l = (q_d @ k_l.transpose(-2, -1)) * self.scale
        attn_d_to_l = attn_d_to_l.softmax(dim=-1)
        attn_d_to_l = self.attn_drop(attn_d_to_l)
        out_d = (attn_d_to_l @ v_l).transpose(1, 2).reshape(B, N_d, C)
        out_d = self.proj_d(out_d)
        out_d = self.proj_drop(out_d)

        return out_l + out_d


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class Sam_my(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        args,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        # cnn: CNN,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.args = args
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.neck5 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1, bias=False, ),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False, ),
            LayerNorm2d(256),
        )

        self.ExpertChoiceTokenMoE = ExpertChoiceTokenSparseMoE(n_embed=self.image_encoder.embed_dim, num_experts=4, top_k=int(args.batch_size*12*196*2/4))

        self.moe_encoder = nn.Sequential(
                nn.ConvTranspose2d(self.image_encoder.out_chans, self.image_encoder.out_chans // 4, kernel_size=2,
                                   stride=2),
                LayerNorm2d(self.image_encoder.out_chans // 4),
                nn.GELU(),
                nn.ConvTranspose2d(self.image_encoder.out_chans // 4, self.image_encoder.out_chans // 8, kernel_size=2,
                                   stride=2),  
                LayerNorm2d(self.image_encoder.out_chans // 8),
                nn.GELU(),)
        self.neck5 = nn.Sequential(
                nn.Conv2d(768, 256, kernel_size=1, bias=False, ),
                LayerNorm2d(256),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False, ),
                LayerNorm2d(256),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.attn = Attention(768, 8)
        self.norm = LayerNorm2d(768)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        batched_input, multimask_output, image_size, gt=None, mode='train'):

        input_images = self.preprocess(batched_input)
        image_embeddings, low_image_embeddings = self.image_encoder(input_images)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=None, masks=None, image_embedding=image_embeddings
        )

        embedding_moe = torch.cat([embed.unsqueeze(1) for embed in low_image_embeddings], dim=1).permute(0, 1, 4, 2, 3).contiguous()

        embedding_moe = embedding_moe.permute(0, 1, 3, 4, 2).contiguous()
        bs, num_features, h, w, dim = embedding_moe.shape
        embedding_moe, indices = self.ExpertChoiceTokenMoE(embedding_moe.reshape(bs*num_features, h*w, dim))
        embedding_moe = embedding_moe.reshape(bs, -1, dim)
        embedding_moe = self.attn(embedding_moe, embedding_moe, embedding_moe).reshape(bs, num_features, h, w, dim).permute(0, 1, 4, 2, 3).contiguous()
        embedding_moe = embedding_moe.mean(1)
        image_embeddings = image_embeddings + self.neck5(embedding_moe)

        low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
        )


        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )

        outputs = {
                "masks": masks,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_masks,
                "indices": indices}

        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


if __name__ == '__main__':
    model = ImageEncoderViT()
    input_tensor = torch.randn(1, 3, 256, 256)

    P = model(input_tensor)
    print(P[0].size(), P[1].size(), P[2].size(), P[3].size())