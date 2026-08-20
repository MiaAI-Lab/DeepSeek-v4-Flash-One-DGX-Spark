#!/usr/bin/env python3
"""DSV4 432-byte NVFP4 EXTEND (prefill) round-trip incl. the indexed path.

The engine's C4A/C128A layers prefill through _run_compressed_mla(
mode="extend") with an SWA cache (64-slot pages) plus an indexed compressed
cache (C4A: 64-slot pages; C128A: 2-slot pages). The stock selftest only
covers mode="decode". This reproduces the prefill contract with padded
physical page strides and checks against a float32 oracle.

Cases:
  A. extend, swa-only, rows=8            (draft-layer prefill shape)
  B. extend, swa(128) + indexed(16), indexed_page_size=64   (C4A shape)
  C. extend, swa(128) + indexed(16), indexed_page_size=2    (C128A shape)
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
HEADS = 32
ROWS = 8

SWA_TOKENS = 128
PHYSICAL_STRIDE = 32768
IDX_SLOTS = 128  # selected compressed entries per row (must be >= 64: kernel reads 64-entry tiles)


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


def make_strided_cache(
    kv: torch.Tensor, slots: torch.Tensor, page_slots: int
) -> torch.Tensor:
    """Write kv into a padded-stride page view via the fused insert op."""
    max_slot = int(slots.max().item())
    pages = math.ceil((max_slot + 1) / page_slots)
    backing = torch.zeros(
        pages * PHYSICAL_STRIDE + 64, device="cuda", dtype=torch.uint8
    )
    cache = torch.as_strided(
        backing,
        size=(pages, page_slots * RECORD_BYTES),
        stride=(PHYSICAL_STRIDE, 1),
    )
    tokens = kv.shape[0]
    q_dummy = torch.randn(
        tokens, 8, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q_dummy,
        kv,
        cache,
        slots,
        torch.arange(tokens, device="cuda", dtype=torch.int64),
        make_cos_sin_cache(tokens + 1),
        8,
        1e-6,
        page_slots,
        True,  # use_nvfp4
    )
    return cache


def dequantize(
    cache: torch.Tensor, slot: int, page_slots: int = PAGE_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    page, pos = divmod(slot, page_slots)
    record = cache[page, pos * RECORD_BYTES : (pos + 1) * RECORD_BYTES]
    packed = record[:256]
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(HEAD_DIM)
    magnitude = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device="cuda",
    )
    latent = magnitude[(nibbles & 0x7).long()]
    latent = torch.where((nibbles & 0x8).bool(), -latent, latent)
    scales = record[256:288].contiguous().view(torch.float8_e4m3fn).float()
    latent = latent * scales.repeat_interleave(16, dim=-1)
    rope = record[304:432].contiguous().view(torch.bfloat16).float()
    key = torch.cat((latent[:NOPE_DIM], rope), dim=-1)
    return key, latent


def run_case(
    name: str,
    *,
    indexed_page_size: int | None,
    rows: int = ROWS,
) -> dict:
    torch.manual_seed(0)
    # SWA cache: slots 0..63
    kv_swa = torch.randn(SWA_TOKENS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    swa_cache = make_strided_cache(
        kv_swa, torch.arange(SWA_TOKENS, device="cuda"), PAGE_SIZE
    )
    swa_indices = torch.arange(SWA_TOKENS, device="cuda", dtype=torch.int32)
    swa_indices = swa_indices.unsqueeze(0).repeat(rows, 1)
    swa_lens = torch.full((rows,), SWA_TOKENS, device="cuda", dtype=torch.int32)

    keys = []
    values = []
    for s in range(SWA_TOKENS):
        k, v = dequantize(swa_cache, s)
        keys.append(k)
        values.append(v)

    indexed_indices = None
    indexed_lens = None
    if indexed_page_size is not None:
        # Selected compressed slots spread across pages.
        sel = torch.arange(0, IDX_SLOTS, device="cuda", dtype=torch.int32)
        sel = (sel * indexed_page_size + 3)  # spread across pages
        max_slot = int(sel.max().item())
        tokens_idx = max_slot + 1
        kv_idx = torch.randn(
            tokens_idx, HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )
        idx_cache = make_strided_cache(
            kv_idx,
            torch.arange(tokens_idx, device="cuda"),
            indexed_page_size,
        )
        indexed_indices = sel.unsqueeze(0).repeat(rows, 1).contiguous()
        indexed_lens = torch.full((rows,), IDX_SLOTS, device="cuda", dtype=torch.int32)
        for s in sel.tolist():
            k, v = dequantize(idx_cache, s, indexed_page_size)
            keys.append(k)
            values.append(v)
    else:
        idx_cache = None

    key = torch.stack(keys)
    value = torch.stack(values)
    total = key.shape[0]

    q = (
        torch.randn(rows, HEADS, HEAD_DIM, device="cuda", dtype=torch.float32) * 0.04
    ).to(torch.bfloat16)

    width = SWA_TOKENS + (IDX_SLOTS if indexed_page_size is not None else 0)
    scratch_plan = plan(
        Caps(
            device="cuda",
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=HEADS,
            head_dim=HEAD_DIM,
            v_head_dim=HEAD_DIM,
            page_size=PAGE_SIZE,
            max_width=width,
            max_q_rows=rows,
            max_batch=rows,
            max_kv_rows=rows * width,
            max_chunks_per_row=64,
        )
    )
    (spec,) = scratch_plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lens,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lens,
        indexed_page_table=None,
    )
    binding.scratch.mode = "extend"

    actual = run(
        binding=binding,
        swa_k_cache=swa_cache,
        swa_page_size=PAGE_SIZE,
        indexed_k_cache=idx_cache,
        indexed_page_size=indexed_page_size,
        sm_scale=1.0 / math.sqrt(HEAD_DIM),
        expected_num_q_heads=HEADS,
        scale_format=2,
        fp8_rope=False,
    )
    torch.cuda.synchronize()

    # Oracle: attention over the union (swa entries then indexed entries).
    scale = 1.0 / math.sqrt(HEAD_DIM)
    scores = torch.einsum("rhd,td->rht", q.float(), key) * scale
    probs = torch.softmax(scores, dim=-1)
    expected = torch.einsum("rht,td->rhd", probs, value)

    has_nan = bool(torch.isnan(actual.float()).any().item())
    cosine = (
        torch.nn.functional.cosine_similarity(
            actual.float().reshape(-1), expected.reshape(-1), dim=0
        ).item()
    )
    max_abs = (actual.float() - expected).abs().max().item()
    return {
        "case": name,
        "rows": rows,
        "total_kv": total,
        "nan": has_nan,
        "cosine": round(cosine, 6),
        "max_abs": round(max_abs, 6),
    }


def main() -> None:
    results = [
        run_case("A: extend swa-only", indexed_page_size=None),
        run_case("B: extend + indexed p64 (C4A)", indexed_page_size=64),
        run_case("C: extend + indexed p2 (C128A)", indexed_page_size=2),
    ]
    for r in results:
        print(r)
    ok = all(not r["nan"] and r["cosine"] > 0.99 for r in results)
    print("EXTEND SELFTEST:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
