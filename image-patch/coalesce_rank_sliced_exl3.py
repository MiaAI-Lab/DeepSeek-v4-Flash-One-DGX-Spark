#!/usr/bin/env python3
"""Losslessly coalesce a TP rank-sliced EXL3 archive into TP1.

The operation never decodes or requantizes Trellis payloads.  It concatenates
partitioned Trellis/rotation tensors along their original tensor-parallel axes
and requires replicated rotations and codebook markers to be byte-identical.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


RANKED_EXL3_KEY = re.compile(
    r"^(?P<prefix>.+\.(?P<projection>gate_proj|up_proj|down_proj|w1|w2|w3))"
    r"\.rank(?P<rank>\d+)\.(?P<field>trellis|suh|svh|mcg|mul1)$"
)
RANKED_FILENAME = re.compile(
    r"^(?P<prefix>.+)-tp(?P<tp>\d+)-rank(?P<rank>\d+)\.safetensors$"
)
DTYPE_BYTES = {
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I8": 1,
    "U8": 1,
    "BF16": 2,
    "F16": 2,
    "I16": 2,
    "U16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--link-carried",
        action="store_true",
        help="Hard-link unchanged files when input and output share a filesystem.",
    )
    parser.add_argument(
        "--no-checksums",
        action="store_true",
        help="Skip the final SHA-256 file inventory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of rank families to coalesce concurrently (default: 1).",
    )
    parser.add_argument(
        "--reuse-complete",
        action="store_true",
        help="Reuse atomic family outputs whose safetensors keys match the source.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_file_bytes(path: Path) -> int:
    total = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():  # noqa: SIM118
            tensor_slice = handle.get_slice(key)
            dtype = str(tensor_slice.get_dtype())
            try:
                itemsize = DTYPE_BYTES[dtype]
            except KeyError as exc:
                raise ValueError(
                    f"unsupported safetensors dtype {dtype} in {path}"
                ) from exc
            elements = 1
            for size in tensor_slice.get_shape():
                elements *= int(size)
            total += elements * itemsize
    return total


def copy_or_link(source: Path, target: Path, link: bool) -> None:
    if target.exists() and target.stat().st_size == source.stat().st_size:
        return
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    if link:
        try:
            os.link(source, temporary)
        except OSError:
            # source is on a different filesystem (e.g. shared network folder):
            # no hardlink possible, fall back to copying.
            shutil.copy2(source, temporary)
    else:
        shutil.copy2(source, temporary)
    os.replace(temporary, target)


def _partition_axis(projection: str, field: str) -> int | None:
    if projection in {"gate_proj", "up_proj", "w1", "w3"}:
        if field == "trellis":
            return 1
        if field == "svh":
            return 0
        return None
    if projection in {"down_proj", "w2"}:
        if field in {"trellis", "suh"}:
            return 0
        return None
    raise ValueError(f"unsupported EXL3 projection {projection!r}")


def combine_rank_tensors(
    tensors: list[torch.Tensor],
    *,
    projection: str,
    field: str,
    logical_name: str,
) -> torch.Tensor:
    if not tensors:
        raise ValueError(f"no tensors supplied for {logical_name}")
    dtype = tensors[0].dtype
    if any(tensor.dtype != dtype for tensor in tensors):
        raise ValueError(f"rank dtype mismatch for {logical_name}")
    axis = _partition_axis(projection, field)
    if axis is not None:
        reference = list(tensors[0].shape)
        for tensor in tensors[1:]:
            shape = list(tensor.shape)
            if len(shape) != len(reference) or any(
                shape[index] != reference[index]
                for index in range(len(shape))
                if index != axis
            ):
                raise ValueError(f"rank shape mismatch for {logical_name}")
        return torch.cat(tensors, dim=axis).contiguous()

    reference = tensors[0]
    for rank, tensor in enumerate(tensors[1:], start=1):
        if not torch.equal(reference, tensor):
            raise ValueError(
                f"replicated EXL3 tensor differs at rank {rank}: {logical_name}"
            )
    return reference.contiguous()


def collect_rank_families(
    weight_map: dict[str, str],
) -> tuple[int, dict[str, dict[int, str]], set[str]]:
    checkpoint_tp: int | None = None
    families: dict[str, dict[int, str]] = {}
    carried_files: set[str] = set()
    for key, filename in weight_map.items():
        key_match = RANKED_EXL3_KEY.match(key)
        if key_match is None:
            carried_files.add(filename)
            continue
        file_match = RANKED_FILENAME.match(filename)
        if file_match is None:
            raise ValueError(
                f"rank-sliced tensor maps to an unrecognized filename: {key} -> {filename}"
            )
        key_rank = int(key_match.group("rank"))
        file_rank = int(file_match.group("rank"))
        file_tp = int(file_match.group("tp"))
        if key_rank != file_rank:
            raise ValueError(f"tensor/file rank mismatch: {key} -> {filename}")
        if checkpoint_tp is None:
            checkpoint_tp = file_tp
        elif checkpoint_tp != file_tp:
            raise ValueError("rank-sliced filenames contain multiple TP degrees")
        family = file_match.group("prefix")
        prior = families.setdefault(family, {}).get(file_rank)
        if prior is not None and prior != filename:
            raise ValueError(f"rank family {family!r} has duplicate rank {file_rank}")
        families[family][file_rank] = filename

    if checkpoint_tp is None or not families:
        raise ValueError("input index contains no rank-sliced EXL3 tensors")
    expected_ranks = set(range(checkpoint_tp))
    for family, rank_files in families.items():
        if set(rank_files) != expected_ranks:
            raise ValueError(
                f"rank family {family!r} is incomplete: "
                f"found={sorted(rank_files)}, expected={sorted(expected_ranks)}"
            )
    return checkpoint_tp, families, carried_files


def coalesce_family(
    *,
    input_dir: Path,
    output_dir: Path,
    family: str,
    rank_files: dict[int, str],
) -> tuple[dict[str, str], Path]:
    rank_payloads = [
        load_file(input_dir / rank_files[rank], device="cpu")
        for rank in sorted(rank_files)
    ]
    grouped: dict[str, dict[int, tuple[re.Match[str], torch.Tensor]]] = {}
    for rank, payload in enumerate(rank_payloads):
        for key, tensor in payload.items():
            match = RANKED_EXL3_KEY.match(key)
            if match is None:
                raise ValueError(f"unexpected non-EXL3 tensor in {rank_files[rank]}: {key}")
            tensor_rank = int(match.group("rank"))
            if tensor_rank != rank:
                raise ValueError(
                    f"tensor rank {tensor_rank} found in rank-{rank} file: {key}"
                )
            logical_name = f"{match.group('prefix')}.{match.group('field')}"
            grouped.setdefault(logical_name, {})[rank] = (match, tensor)

    output_tensors: dict[str, torch.Tensor] = {}
    output_filename = f"{family}-tp1-rank0.safetensors"
    output_map: dict[str, str] = {}
    expected_ranks = set(range(len(rank_payloads)))
    for logical_name, values in sorted(grouped.items()):
        if set(values) != expected_ranks:
            raise ValueError(
                f"logical tensor {logical_name} is missing ranks: "
                f"found={sorted(values)}, expected={sorted(expected_ranks)}"
            )
        match = values[0][0]
        output_key = f"{match.group('prefix')}.rank0.{match.group('field')}"
        output_tensors[output_key] = combine_rank_tensors(
            [values[rank][1] for rank in sorted(values)],
            projection=match.group("projection"),
            field=match.group("field"),
            logical_name=logical_name,
        )
        output_map[output_key] = output_filename

    target = output_dir / output_filename
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    save_file(
        output_tensors,
        temporary,
        metadata={
            "operation": "lossless_tp_rank_coalesce",
            "source_tp": str(len(rank_payloads)),
            "target_tp": "1",
        },
    )
    os.replace(temporary, target)
    del output_tensors, rank_payloads, grouped
    gc.collect()
    return output_map, target


def coalesce_family_item(
    item: tuple[Path, Path, str, dict[int, str], bool],
) -> tuple[str, dict[str, str], Path]:
    input_dir, output_dir, family, rank_files, reuse_complete = item
    output_filename = f"{family}-tp1-rank0.safetensors"
    output = output_dir / output_filename
    if reuse_complete and output.is_file():
        expected_keys: set[str] = set()
        for rank, filename in sorted(rank_files.items()):
            with safe_open(input_dir / filename, framework="pt", device="cpu") as handle:
                for key in handle.keys():  # noqa: SIM118
                    match = RANKED_EXL3_KEY.match(key)
                    if match is None or int(match.group("rank")) != rank:
                        raise ValueError(f"invalid ranked EXL3 source key: {key}")
                    expected_keys.add(
                        f"{match.group('prefix')}.rank0.{match.group('field')}"
                    )
        with safe_open(output, framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
        if actual_keys == expected_keys:
            mapped = {key: output_filename for key in expected_keys}
            print(f"reused {family} -> {output.name}", flush=True)
            return family, mapped, output

    mapped, output = coalesce_family(
        input_dir=input_dir,
        output_dir=output_dir,
        family=family,
        rank_files=rank_files,
    )
    print(f"coalesced {family} -> {output.name}", flush=True)
    return family, mapped, output


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("input and output directories must differ")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_json(input_dir / "config.json")
    quant_config = read_json(input_dir / "quantization_config.json")
    index = read_json(input_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json has no weight_map")

    metadata = config.get("hybrid_tr3_tail")
    if not isinstance(metadata, dict) or metadata.get("format") != "exl3-trellis":
        raise ValueError("config.json has no rank-sliced EXL3 metadata")
    checkpoint_tp, families, carried_files = collect_rank_families(weight_map)
    if int(metadata.get("tp", -1)) != checkpoint_tp:
        raise ValueError(
            "config/index TP mismatch: "
            f"config={metadata.get('tp')}, files={checkpoint_tp}"
        )

    for filename in sorted(carried_files):
        copy_or_link(
            input_dir / filename,
            output_dir / filename,
            args.link_carried,
        )

    output_weight_map = {
        key: filename
        for key, filename in weight_map.items()
        if RANKED_EXL3_KEY.match(key) is None
    }
    output_tensor_files = [output_dir / name for name in sorted(carried_files)]
    family_items = [
        (input_dir, output_dir, family, rank_files, args.reuse_complete)
        for family, rank_files in sorted(families.items())
    ]

    if args.workers == 1:
        results = map(coalesce_family_item, family_items)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(coalesce_family_item, family_items)

    try:
        for _, mapped, output in results:
            overlap = set(output_weight_map).intersection(mapped)
            if overlap:
                raise ValueError(f"duplicate output tensor keys: {sorted(overlap)[:4]}")
            output_weight_map.update(mapped)
            output_tensor_files.append(output)
    finally:
        if args.workers != 1:
            executor.shutdown(wait=True, cancel_futures=True)

    output_metadata = dict(metadata)
    output_metadata["tp"] = 1
    output_metadata["source_tp"] = checkpoint_tp
    output_metadata["source_operation"] = "lossless_tp_rank_coalesce"
    config["hybrid_tr3_tail"] = output_metadata
    quant_config["rank_sliced"] = output_metadata
    quant_config["source_operation"] = "lossless_tp_rank_coalesce"

    tensor_bytes = sum(tensor_file_bytes(path) for path in output_tensor_files)
    output_index = {
        "metadata": {**index.get("metadata", {}), "total_size": tensor_bytes},
        "weight_map": dict(sorted(output_weight_map.items())),
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "quantization_config.json", quant_config)
    write_json(output_dir / "model.safetensors.index.json", output_index)

    reserved = {
        "config.json",
        "quantization_config.json",
        "model.safetensors.index.json",
    }
    for source in input_dir.iterdir():
        if not source.is_file() or source.name in reserved:
            continue
        if source.suffix == ".safetensors":
            continue
        copy_or_link(source, output_dir / source.name, args.link_carried)

    manifest: dict[str, Any] = {
        "format": "rank-sliced-exl3-tp1-v1",
        "source": str(input_dir),
        "source_tp": checkpoint_tp,
        "target_tp": 1,
        "tensor_count": len(output_weight_map),
        "tensor_bytes": tensor_bytes,
        "files": [],
    }
    if not args.no_checksums:
        manifest["files"] = [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(output_tensor_files)
        ]
    write_json(output_dir / "rank-sliced-tp1-manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
