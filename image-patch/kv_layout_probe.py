#!/usr/bin/env python3
"""Offline replication of the DSV4 432-mode KV cache layout.

Rebuilds the exact kv_cache_spec dict from model configs, runs the real
grouping/allocation functions from vllm, and prints per-layer tensor
geometry (page size, padded page, block stride). Validates against the
live engine's observed indexer stride (953728 B per page dim).
"""

import json

import torch

from vllm.v1.core.kv_cache_utils import (
    _bucket_layers_by_page_size,
    _get_kv_cache_groups_uniform_groups,
    _get_lockstep_mla_tensor_slots,
    _use_lockstep_mla_allocation,
    group_and_unify_kv_cache_specs,
)
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

TARGET_CFG = "/models/tp1/config.json"
BLOCK_SIZE = 256
HEAD_DIM = 512
SWA_BLOCK = 64
INDEX_HEAD = 128 + 4  # 128 fp8 + 1 fp32 scale
CACHE_DTYPE = "nvfp4_ds_mla"
ALIGN = 432  # VLLM_DSV4_PADDED_NVFP4=0


def build_specs():
    cfg = json.load(open(TARGET_CFG))
    ratios = cfg["compress_ratios"]
    specs = {}
    for i, cr in enumerate(ratios):
        p = f"model.layers.{i}.self_attn"
        specs[f"{p}.swa_cache"] = SlidingWindowMLASpec(
            block_size=SWA_BLOCK,
            num_kv_heads=1,
            head_size=HEAD_DIM,
            dtype=torch.uint8,
            sliding_window=cfg["sliding_window"],
            cache_dtype_str=CACHE_DTYPE,
            alignment=ALIGN,
            model_version="deepseek_v4",
        )
        if cr > 1:
            specs[p] = MLAAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=HEAD_DIM,
                dtype=torch.uint8,
                compress_ratio=cr,
                cache_dtype_str=CACHE_DTYPE,
                alignment=ALIGN,
                model_version="deepseek_v4",
            )
            specs[f"{p}.indexer.k_cache"] = MLAAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=INDEX_HEAD,
                dtype=torch.uint8,
                compress_ratio=cr,
                alignment=512,
            )
            coff = 1 + (cr == 4)
            specs[f"{p}.compressor.state_cache"] = SlidingWindowMLASpec(
                block_size=4 if cr == 4 else 8,
                num_kv_heads=1,
                head_size=2 * coff * HEAD_DIM,
                dtype=torch.float32,
                sliding_window=coff * cr,
                alignment=512,
                dcp_replicated=True,
            )
    # draft layers: 3 SWA-only (dspark), inserted last (eagle annotation).
    for j in range(3):
        specs[f"draft.layers.{43 + j}.self_attn.swa_cache"] = SlidingWindowMLASpec(
            block_size=SWA_BLOCK,
            num_kv_heads=1,
            head_size=HEAD_DIM,
            dtype=torch.uint8,
            sliding_window=cfg["sliding_window"],
            cache_dtype_str=CACHE_DTYPE,
            alignment=ALIGN,
            model_version="deepseek_v4",
        )
    return specs


def main():
    specs = build_specs()
    grouped = group_and_unify_kv_cache_specs(specs, 1, 1)
    assert grouped is not None, "DSV4 grouping path not taken"
    groups = _get_kv_cache_groups_uniform_groups(grouped)

    print(f"{'#groups':>8}: {len(groups)}")
    for gi, g in enumerate(groups):
        gs = g.kv_cache_spec
        layer_specs = (
            gs.kv_cache_specs if isinstance(gs, UniformTypeKVCacheSpecs) else None
        )
        kinds = {}
        for n in g.layer_names:
            s = layer_specs[n] if layer_specs else gs
            kinds.setdefault(
                (type(s).__name__, s.block_size, s.page_size_bytes), []
            ).append(n)
        print(f"\ngroup[{gi}] layers={len(g.layer_names)}")
        for (kind, bs, ps), names in sorted(kinds.items()):
            print(
                f"  {kind:<22} block={bs:<5} page={ps:<7} "
                f"n={len(names):<3} e.g. {names[0]}"
            )

    if _use_lockstep_mla_allocation(groups, 1, 1):
        slots = _get_lockstep_mla_tensor_slots(groups)
        mode = "lockstep"
    else:
        slots = [
            (ps, slot)
            for ps, slots_ in _bucket_layers_by_page_size(groups).items()
            for slot in slots_
        ]
        mode = "bucketed"
    total = sum(ps for ps, _ in slots)
    print(f"\nallocation mode: {mode}")
    print(f"layers-in-tensor: {len(slots)}  total bytes/block: {total}")

    off = 0
    print(f"{'offset':>8} {'page':>7} {'stride':>8}  layer")
    for ps, names in slots:
        for n in names:
            print(f"{off:>8} {ps:>7} {total:>8}  {n}")
        off += ps

    print("\nVALIDATION vs live engine (indexer kq stride=953728):")
    idx = [
        (o, ps, n) for o, ps, n in _iter(slots)
        if ".indexer.k_cache" in n and "layers.2." in n
    ]
    for o, ps, n in idx:
        print(f"  layer-2 indexer: offset={o} page={ps} stride={total}"
              f"  match={'YES' if total == 953728 else 'NO'}")


def _iter(slots):
    off = 0
    for ps, names in slots:
        for n in names:
            yield off, ps, n
        off += ps


if __name__ == "__main__":
    main()
