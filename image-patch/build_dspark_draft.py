#!/usr/bin/env python3
"""Build a compact, target-preserving DSpark draft checkpoint.

The target checkpoint is never modified.  Only ``mtp.*`` tensors are copied
into a separate draft directory.  Routed draft experts are selected from the
REAP provenance map, with the two structured-output calibration categories
represented before the global REAP ranking fills the remaining slots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


EXPERT_RE = re.compile(r"^(mtp\.(\d+)\.ffn\.experts\.)(\d+)(\..+)$")
GATE_NAMES = {"ffn.gate.weight", "ffn.gate.bias"}
STRUCTURED_CATEGORIES = ("agentic_tool_trajectory", "tool_calling")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--structured-per-category", type=int, default=32)
    parser.add_argument("--plan", type=Path)
    return parser.parse_args()


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def add_ranked(
    destination: list[int], ranking: list[int], available: set[int], limit: int
) -> None:
    for expert in ranking:
        if expert in available and expert not in destination:
            destination.append(expert)
            if len(destination) == limit:
                return


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    plan_path = (args.plan or source / "REAP_K216_PLAN.json").resolve()

    if source == output:
        raise SystemExit("source and output must be different directories")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    if args.experts < 8:
        raise SystemExit("--experts must be at least 8")

    index_path = source / "model.safetensors.index.json"
    config_path = source / "config.json"
    index = json.loads(index_path.read_text())
    plan = json.loads(plan_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    mtp_keys = sorted(key for key in weight_map if key.startswith("mtp."))
    if not mtp_keys:
        raise SystemExit("source index contains no mtp.* tensors")

    keep_maps = plan["keep_maps"]
    current_to_original: list[int] = keep_maps["mtp_keep"]
    if current_to_original != sorted(current_to_original):
        raise SystemExit("mtp_keep must be sorted to reconstruct current expert IDs")
    original_to_current = {
        original: current for current, original in enumerate(current_to_original)
    }
    if args.experts >= len(current_to_original):
        raise SystemExit(
            f"--experts must be smaller than the current {len(current_to_original)}"
        )

    structured = keep_maps["structured_ranked_by_category"]
    layer = str(keep_maps["mtp_keep_from_layer"])
    selected_original: list[int] = []
    available = set(current_to_original)
    for category in STRUCTURED_CATEGORIES:
        ranking = structured[category][layer][: args.structured_per_category]
        add_ranked(selected_original, ranking, available, args.experts)
    add_ranked(selected_original, keep_maps["mtp_ranked"], available, args.experts)
    if len(selected_original) != args.experts:
        raise SystemExit(
            f"selection produced {len(selected_original)} experts, expected {args.experts}"
        )

    selected_current = sorted(original_to_current[x] for x in selected_original)
    current_to_new = {current: new for new, current in enumerate(selected_current)}

    # Validate all three DSpark stages before allocating output tensors.
    expert_ids_by_stage: dict[int, set[int]] = {}
    for key in mtp_keys:
        match = EXPERT_RE.match(key)
        if match:
            expert_ids_by_stage.setdefault(int(match.group(2)), set()).add(
                int(match.group(3))
            )
    expected_current = set(range(len(current_to_original)))
    if sorted(expert_ids_by_stage) != [0, 1, 2]:
        raise SystemExit(f"expected DSpark stages 0,1,2; got {expert_ids_by_stage.keys()}")
    for stage, expert_ids in expert_ids_by_stage.items():
        if expert_ids != expected_current:
            raise SystemExit(
                f"stage {stage} expert IDs are incomplete: "
                f"{len(expert_ids)} vs {len(expected_current)}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        tensors: dict[str, torch.Tensor] = {}
        output_weight_map: dict[str, str] = {}
        shard_name = "dspark-draft.safetensors"
        shard_paths = sorted({weight_map[key] for key in mtp_keys})
        with ExitStack() as stack:
            shards = {
                name: stack.enter_context(
                    safe_open(source / name, framework="pt", device="cpu")
                )
                for name in shard_paths
            }
            for key in mtp_keys:
                tensor = shards[weight_map[key]].get_tensor(key)
                match = EXPERT_RE.match(key)
                output_key = key
                if match:
                    current = int(match.group(3))
                    if current not in current_to_new:
                        continue
                    output_key = (
                        f"{match.group(1)}{current_to_new[current]}{match.group(4)}"
                    )
                elif any(key.endswith(name) for name in GATE_NAMES):
                    if tensor.shape[0] != len(current_to_original):
                        raise SystemExit(
                            f"unexpected gate shape for {key}: {tuple(tensor.shape)}"
                        )
                    tensor = tensor[selected_current].contiguous()
                tensors[output_key] = tensor
                output_weight_map[output_key] = shard_name

            save_file(
                tensors,
                temp / shard_name,
                metadata={
                    "format": "pt",
                    "purpose": "target-preserving compact DSpark draft",
                },
            )

        config = json.loads(config_path.read_text())
        config["n_routed_experts"] = args.experts
        if args.experts % int(config.get("n_group", 1)) != 0:
            raise SystemExit(
                f"{args.experts} experts must be divisible by n_group={config['n_group']}"
            )
        (temp / "config.json").write_text(json.dumps(config, indent=2) + "\n")

        total_size = sum(tensor_nbytes(tensor) for tensor in tensors.values())
        output_index = {
            "metadata": {"total_size": total_size},
            "weight_map": output_weight_map,
        }
        (temp / "model.safetensors.index.json").write_text(
            json.dumps(output_index, indent=2, sort_keys=True) + "\n"
        )

        if args.structured_per_category:
            selection_policy = (
                "top structured specialists per category, then global REAP fill"
            )
        else:
            selection_policy = "global REAP ranking"

        provenance = {
            "source": str(source),
            "source_plan": str(plan_path),
            "source_experts": len(current_to_original),
            "draft_experts": args.experts,
            "selection_policy": selection_policy,
            "structured_categories": list(STRUCTURED_CATEGORIES),
            "structured_per_category": args.structured_per_category,
            "selected_original_expert_ids": selected_original,
            "selected_current_expert_ids": selected_current,
            "current_to_new_expert_id": {
                str(current): new for current, new in current_to_new.items()
            },
            "tensor_count": len(tensors),
            "total_size": total_size,
        }
        (temp / "DSPARK_DRAFT_PLAN.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )

        # Reload metadata and assert the saved expert/gate shapes before publish.
        with safe_open(temp / shard_name, framework="pt", device="cpu") as check:
            saved_keys = set(check.keys())
            for stage in range(3):
                expert_ids = {
                    int(match.group(3))
                    for key in saved_keys
                    if (match := EXPERT_RE.match(key)) and int(match.group(2)) == stage
                }
                if expert_ids != set(range(args.experts)):
                    raise SystemExit(
                        f"saved stage {stage} expert IDs invalid: {sorted(expert_ids)}"
                    )
                for suffix in GATE_NAMES:
                    shape = check.get_slice(f"mtp.{stage}.{suffix}").get_shape()
                    if shape[0] != args.experts:
                        raise SystemExit(
                            f"saved mtp.{stage}.{suffix} shape invalid: {shape}"
                        )

        provenance["sha256"] = {shard_name: sha256(temp / shard_name)}
        (temp / "DSPARK_DRAFT_PLAN.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temp, output)
        print(json.dumps(provenance, indent=2, sort_keys=True))
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
