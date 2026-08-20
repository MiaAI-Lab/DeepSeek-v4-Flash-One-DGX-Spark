#!/usr/bin/env python3
"""DSV4 432-byte NVFP4 cache round-trip with PADDED physical page stride.

The 432-mode engine pads compressed/SWA physical pages to 32 KiB (the
compressor-state anchor) while the logical payload stays 64*432 = 27648 B.
The stock selftest only exercises a contiguous cache (stride == width), so a
kernel that ignores swa_k_cache.stride(0) passes it and still breaks the real
engine. This variant reproduces the engine geometry:

  physical page stride : 32768 B (anchor padding)
  logical page width   : 27648 B (64 slots x 432 B)
  layer storage offset : 512 B (packed per-layer view)

Pass criteria: cosine > 0.99 for both rows=1 and rows=16, same as selftest.py.
"""

from __future__ import annotations

import math

import torch

import vllm  # noqa: F401  # Loads the stable libtorch custom-op library.
from sparkinfer.attention.compressed_mla import Caps, plan, run

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
RECORD_BYTES = 432
PAGE_SIZE = 64
TOKENS = 128
HEADS = 32

PHYSICAL_STRIDE = 32768  # 32 KiB anchor-padded page (432-mode engine layout)
LAYER_OFFSET = 512  # simulate a packed per-layer storage offset


def make_cos_sin_cache(max_pos: int) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, ROPE_DIM, 2, device="cuda", dtype=torch.float32)
            / ROPE_DIM
        )
    )
    positions = torch.arange(max_pos, device="cuda", dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def make_strided_cache(kv: torch.Tensor) -> torch.Tensor:
    """Write kv through the fused insert op into a padded-stride page view."""
    tokens = kv.shape[0]
    pages = math.ceil(tokens / PAGE_SIZE)
    backing = torch.zeros(
        LAYER_OFFSET + pages * PHYSICAL_STRIDE + 64,
        device="cuda",
        dtype=torch.uint8,
    )
    cache = torch.as_strided(
        backing,
        size=(pages, PAGE_SIZE * RECORD_BYTES),
        stride=(PHYSICAL_STRIDE, 1),
        storage_offset=LAYER_OFFSET,
    )
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
    q_dummy = torch.randn(
        tokens, 8, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q_dummy,
        kv,
        cache,  # [pages, 27648] uint8, stride(0)=32768, storage_offset=512
        slots,
        positions,
        make_cos_sin_cache(tokens + 1),
        8,
        1e-6,
        PAGE_SIZE,
        True,  # use_nvfp4
    )
    return cache


def dequantize_cache(cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    records = cache.reshape(-1, RECORD_BYTES)[:TOKENS]
    packed = records[:, :256]
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(
        TOKENS, HEAD_DIM
    )
    magnitude = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device="cuda",
    )
    latent = magnitude[(nibbles & 0x7).long()]
    latent = torch.where((nibbles & 0x8).bool(), -latent, latent)
    scales = records[:, 256:288].contiguous().view(torch.float8_e4m3fn).float()
    latent = latent * scales.repeat_interleave(16, dim=-1)
    rope = records[:, 304:432].contiguous().view(torch.bfloat16).float()
    key = torch.cat((latent[:, :NOPE_DIM], rope), dim=-1)
    return key, latent


def make_binding(q: torch.Tensor, *, use_cuda_graph: bool):
    rows = q.shape[0]
    indices = torch.arange(TOKENS, device="cuda", dtype=torch.int32).repeat(
        rows, 1
    )
    lengths = torch.full((rows,), TOKENS, device="cuda", dtype=torch.int32)
    scratch_plan = plan(
        Caps(
            device="cuda",
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=HEADS,
            head_dim=HEAD_DIM,
            v_head_dim=HEAD_DIM,
            max_width=TOKENS,
            max_q_rows=rows,
            max_batch=rows,
            max_kv_rows=rows * TOKENS,
            max_chunks_per_row=1,
        )
    )
    (spec,) = scratch_plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        swa_indices=indices,
        swa_lengths=lengths,
        indexed_indices=None,
        indexed_lengths=None,
        indexed_page_table=None,
    )
    binding.scratch.mode = "decode"
    binding.scratch.use_cuda_graph = use_cuda_graph
    return binding


def oracle(q: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(HEAD_DIM)
    scores = torch.einsum("rhd,td->rht", q.float(), key) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("rht,td->rhd", probs, value)


def check(rows: int, cache: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    q = (
        torch.randn(rows, HEADS, HEAD_DIM, device="cuda", dtype=torch.float32)
        * 0.04
    ).to(torch.bfloat16)
    binding = make_binding(q, use_cuda_graph=True)

    def invoke() -> torch.Tensor:
        return run(
            binding=binding,
            swa_k_cache=cache,  # strided view, stride(0)=32768 != width 27648
            swa_page_size=PAGE_SIZE,
            sm_scale=1.0 / math.sqrt(HEAD_DIM),
            expected_num_q_heads=HEADS,
            scale_format=2,
            fp8_rope=False,
        )

    actual = invoke()
    torch.cuda.synchronize()
    expected = oracle(q, key, value)
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().reshape(-1), expected.reshape(-1), dim=0
    ).item()
    max_abs = (actual.float() - expected).abs().max().item()
    mean_abs = (actual.float() - expected).abs().mean().item()
    return {
        "rows": rows,
        "cosine": cosine,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
    }


def main() -> None:
    torch.manual_seed(0)
    kv = torch.randn(TOKENS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    cache = make_strided_cache(kv)
    key, latent = dequantize_cache(cache)
    value = latent  # MQA: value == latent (matches selftest.py)
    results = [check(rows, cache, key, value) for rows in (1, 16)]
    print(
        {
            "device": torch.cuda.get_device_name(0),
            "record_bytes": RECORD_BYTES,
            "physical_stride": PHYSICAL_STRIDE,
            "logical_width": PAGE_SIZE * RECORD_BYTES,
            "results": results,
        }
    )
    ok = all(r["cosine"] > 0.99 for r in results)
    print("PADDED-STRIDE SELFTEST:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
