# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.
#
# This file was added in the fastforks/segment-anything fork (not part of the original release).
"""Fast inference for SAM: any image size, nothing compiled or tuned per size.

    from segment_anything import sam_model_registry
    from segment_anything.fast import FastSamPredictor
    pred = FastSamPredictor(sam_model_registry["vit_h"](checkpoint=...).cuda())   # same API as SamPredictor

Changes vs upstream (the network math is unchanged; only precision / kernels / data movement):
- image encoder in fp16 (weights converted once); LayerNorms run in fp16 I/O with fp32 statistics (torch kernels);
  the neck's LayerNorm2d runs channels-last through F.layer_norm (fp32 statistics); embedding handed to the decoder in fp32
- attention with the decomposed relative-position bias through SDPA: the bias rel_h[q, kh] + rel_w[q, kw] is folded into
  an augmented q/k dot product (q' = [q * scale, rel_h, rel_w], k' = [k, onehot(kh), onehot(kw)]), so the fused
  attention kernels run without materializing the N x N attention matrix; v is zero-padded to the same width so the flash
  kernel applies
- optional fp16-accumulation GEMMs (`gemm_acc16`, off here; 2x tensor-core rate on GeForce) -- validated by mask IoU
- preprocessing on the GPU (uint8 upload, antialiased bilinear resize like PIL, normalize + pad fused)
- prompt encoder / mask decoder / mask upsampling unchanged (fp32), masks thresholded on the GPU before the host copy
"""
import contextlib

import numpy as np
import torch
import torch.nn.functional as F

from .modeling.image_encoder import get_rel_pos
from .predictor import SamPredictor


@contextlib.contextmanager
def fp16_accumulate(enabled=True):
    old = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = old


def _onehot_k(Hk, Wk, device, dtype, cache={}):
    key = (Hk, Wk, device, dtype)
    if key not in cache:
        oh = torch.zeros(Hk, Wk, Hk + Wk, device=device, dtype=dtype)
        a = torch.arange(Hk, device=device)
        oh[a, :, a] = 1
        b = torch.arange(Wk, device=device)
        oh[:, b, Hk + b] = 1
        cache[key] = oh.view(Hk * Wk, Hk + Wk)
    return cache[key]


def relpos_attention(qkv, nh, rel_pos_h, rel_pos_w, hw, rel_cache=None):
    """qkv [B, N, 3*C] (packed linear output, fp16) -> [B, N, C]. Same math as upstream Attention.forward (use_rel_pos)."""
    B, N, C3 = qkv.shape
    C = C3 // 3
    d = C // nh
    H, W = hw
    qkv = qkv.view(B, N, 3, nh, d)
    q, k, v = qkv.unbind(2)  # [B, N, nh, d] strided views
    if rel_pos_h is None:
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return o.transpose(1, 2).reshape(B, N, C)
    key = (H, W)
    if rel_cache is not None and key in rel_cache:
        Rh, Rw = rel_cache[key]
    else:
        Rh = get_rel_pos(H, H, rel_pos_h).to(qkv.dtype)
        Rw = get_rel_pos(W, W, rel_pos_w).to(qkv.dtype)
        if rel_cache is not None:
            rel_cache[key] = (Rh, Rw)
    r_q = q.reshape(B, H, W, nh, d)
    D = d + H + W
    Dp = (D + 7) // 8 * 8
    qa = torch.empty(B, nh, N, Dp, device=qkv.device, dtype=qkv.dtype)
    ka = torch.empty(B, nh, N, Dp, device=qkv.device, dtype=qkv.dtype)
    qa[..., :d] = q.transpose(1, 2) * d ** -0.5
    qa5 = qa.view(B, nh, H, W, Dp)
    qa5[..., d:d + H] = torch.einsum("bhwnc,hkc->bnhwk", r_q, Rh)
    qa5[..., d + H:D] = torch.einsum("bhwnc,wkc->bnhwk", r_q, Rw)
    ka[..., :d] = k.transpose(1, 2)
    ka[..., d:D] = _onehot_k(H, W, qkv.device, qkv.dtype)
    if Dp > D:
        qa[..., D:] = 0
        ka[..., D:] = 0
    # v is zero-padded to the same width so that the flash kernel runs; the padded columns are dropped
    va = torch.zeros(B, nh, N, Dp, device=qkv.device, dtype=qkv.dtype)
    va[..., :d] = v.transpose(1, 2)
    o = F.scaled_dot_product_attention(qa, ka, va, scale=1.0)[..., :d]
    return o.transpose(1, 2).reshape(B, N, C)


def pil_bilinear(x, nh, nw):
    """PIL BILINEAR resize (antialiased, as torchvision.resize on a PIL image) of a float [B,C,H,W] 0..255 tensor on the GPU:
    separable horizontal then vertical pass with uint8 rounding in between, like PIL (<= 0.1 % of pixels differ by 1)."""
    h, w = x.shape[-2:]
    if nw != w:
        x = F.interpolate(x, (h, nw), mode="bilinear", align_corners=False, antialias=True).round_().clamp_(0, 255)
    if nh != h:
        x = F.interpolate(x, (nh, nw), mode="bilinear", align_corners=False, antialias=True).round_().clamp_(0, 255)
    return x


def _window_partition(x, ws):
    B, H, W, C = x.shape
    ph, pw = (ws - H % ws) % ws, (ws - W % ws) % ws
    if ph or pw:
        x = F.pad(x, (0, 0, 0, pw, 0, ph))
    Hp, Wp = H + ph, W + pw
    x = x.view(B, Hp // ws, ws, Wp // ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, C)
    return x, (Hp, Wp)


def _window_unpartition(x, ws, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = x.shape[0] // (Hp * Wp // ws // ws)
    x = x.view(B, Hp // ws, Wp // ws, ws, ws, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    return x[:, :H, :W] if (Hp > H or Wp > W) else x


class FastImageEncoder:
    def __init__(self, enc, gemm_acc16=False, dtype=torch.float16):
        self.enc = enc.to(dtype).eval()
        self.dtype = dtype
        self.gacc = gemm_acc16
        self.rel_cache = [dict() for _ in enc.blocks]

    def block(self, i, blk, x):
        shortcut = x
        x = blk.norm1(x)
        if blk.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = _window_partition(x, blk.window_size)
        B, h, w, C = x.shape
        a = blk.attn
        qkv = a.qkv(x.reshape(B, h * w, C))
        o = relpos_attention(qkv, a.num_heads, a.rel_pos_h if a.use_rel_pos else None,
                             a.rel_pos_w if a.use_rel_pos else None, (h, w), self.rel_cache[i])
        x = a.proj(o).view(B, h, w, C)
        if blk.window_size > 0:
            x = _window_unpartition(x, blk.window_size, pad_hw, (H, W))
        x = shortcut + x
        m = blk.mlp
        return x + m.lin2(F.gelu(m.lin1(blk.norm2(x))))

    def __call__(self, x):
        e = self.enc
        with fp16_accumulate(self.gacc):
            x = e.patch_embed(x.to(self.dtype))
            if e.pos_embed is not None:
                x = x + e.pos_embed
            for i, blk in enumerate(e.blocks):
                x = self.block(i, blk, x)
            n = e.neck  # conv1x1, LN2d, conv3x3, LN2d  (convs fp16 channels-last; LN fp32 statistics)
            y = F.linear(x, n[0].weight.view(n[0].weight.shape[0], -1))
            y = F.layer_norm(y, (y.shape[-1],), n[1].weight, n[1].bias, n[1].eps)
            y = n[2](y.permute(0, 3, 1, 2))
            y = F.layer_norm(y.permute(0, 2, 3, 1), (y.shape[1],), n[3].weight, n[3].bias, n[3].eps)
        return y.float().permute(0, 3, 1, 2).contiguous()


class FastSamPredictor(SamPredictor):
    def __init__(self, sam_model, gemm_acc16=False, dtype=torch.float16):
        super().__init__(sam_model)
        self.fenc = FastImageEncoder(sam_model.image_encoder, gemm_acc16, dtype)

    @torch.no_grad()
    def set_image(self, image, image_format="RGB"):
        assert image_format in ["RGB", "BGR"]
        if image_format != self.model.image_format:
            image = image[..., ::-1]
        h, w = image.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(image)).to(self.device, non_blocking=False)
        x = x.permute(2, 0, 1)[None].float()
        nh, nw = self.transform.get_preprocess_shape(h, w, self.transform.target_length)
        if (nh, nw) != (h, w):
            x = pil_bilinear(x, nh, nw)
        self.reset_image()
        self.original_size = (h, w)
        self.input_size = (nh, nw)
        x = self.model.preprocess(x)
        self.features = self.fenc(x)
        self.is_image_set = True
