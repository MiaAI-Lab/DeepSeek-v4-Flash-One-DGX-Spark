#!/usr/bin/env python3
"""Validate the DSV4 432-byte NVFP4 cache against SparkInfer on SM121."""

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


def write_cache(kv: torch.Tensor) -> torch.Tensor:
    tokens = kv.shape[0]
    q_dummy = torch.randn(
        tokens, 8, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    pages = math.ceil(tokens / PAGE_SIZE)
    cache = torch.zeros(
        pages, PAGE_SIZE, RECORD_BYTES, device="cuda", dtype=torch.uint8
    )
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q_dummy,
        kv,
        cache.view(pages, -1),
        slots,
        positions,
        make_cos_sin_cache(tokens + 1),
        8,
        1e-6,
        PAGE_SIZE,
        True,
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
            swa_k_cache=cache.view(-1, PAGE_SIZE * RECORD_BYTES),
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
    if cosine < 0.99:
        raise AssertionError(
            f"rows={rows}: cosine={cosine:.8f}, max_abs={max_abs:.6f}, "
            f"mean_abs={mean_abs:.6f}"
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = invoke()
    graph.replay()
    torch.cuda.synchronize()
    replay_delta = (graph_output.float() - actual.float()).abs().max().item()
    if replay_delta != 0.0:
        raise AssertionError(f"rows={rows}: CUDA graph replay delta={replay_delta}")
    return {
        "rows": rows,
        "cosine": cosine,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "graph_replay_max_abs": replay_delta,
    }


def main() -> None:
    torch.manual_seed(20260805)
    kv = torch.randn(TOKENS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    cache = write_cache(kv)
    key, value = dequantize_cache(cache)
    if torch.count_nonzero(cache[:, :, 288:304]).item() != 0:
        raise AssertionError("NVFP4 cache padding is not zero")
    results = [check(1, cache, key, value), check(16, cache, key, value)]
    print(
        {
            "device": torch.cuda.get_device_name(),
            "compute_capability": torch.cuda.get_device_capability(),
            "record_bytes": RECORD_BYTES,
            "results": results,
        }
    )


if __name__ == "__main__":
    main()
