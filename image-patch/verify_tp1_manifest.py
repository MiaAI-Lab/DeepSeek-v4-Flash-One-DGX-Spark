#!/usr/bin/env python3
"""Fail closed if the losslessly coalesced TP1 weight inventory is incomplete."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args()
    manifest_path = args.model_dir / "rank-sliced-tp1-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "rank-sliced-exl3-tp1-v1":
        raise ValueError(f"unexpected manifest format: {manifest.get('format')!r}")
    if manifest.get("target_tp") != 1:
        raise ValueError(f"expected target_tp=1, got {manifest.get('target_tp')!r}")
    files = manifest.get("files")
    if not files:
        raise ValueError("manifest has no checksummed tensor files")
    verified_bytes = 0
    for item in files:
        path = args.model_dir / item["name"]
        actual_bytes = path.stat().st_size
        expected_bytes = int(item["bytes"])
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"size mismatch for {path.name}: {actual_bytes} != {expected_bytes}"
            )
        if not args.skip_checksums:
            actual_sha256 = sha256_file(path)
            if actual_sha256 != item["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {path.name}")
        verified_bytes += actual_bytes
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "tensor_files": len(files),
                "tensor_bytes": verified_bytes,
                "checksums": not args.skip_checksums,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
