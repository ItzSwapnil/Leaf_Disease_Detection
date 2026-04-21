#!/usr/bin/env python3
"""Create a leakage-free, stratified dataset split from an existing train/val/test layout.

Features:
- Exact deduplication across all source splits using SHA-1.
- Optional near-duplicate filtering per class via dHash (threshold 0 or 1).
- Deterministic stratified splitting with fixed seed.
- Manifest + summary reports for auditability.
- Optional materialization of a new split tree using hardlinks/copies/symlinks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class CanonicalItem:
    sha1: str
    class_name: str
    source_split: str
    source_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Dataset root containing train/val/test directories.",
    )
    parser.add_argument(
        "--source-splits",
        default="train,val,test",
        help="Comma-separated source split folders to scan.",
    )
    parser.add_argument(
        "--ratios",
        default="0.85,0.075,0.075",
        help="Target train,val,test ratios.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic assignment.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/leakage_free_split_manifest.csv"),
        help="CSV manifest output path.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/leakage_free_split_summary.json"),
        help="JSON summary output path.",
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Materialize the new split under --output-root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset_clean"),
        help="Output root for materialized split tree.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="How to materialize files when --materialize is enabled.",
    )
    parser.add_argument(
        "--prefix-sha1",
        action="store_true",
        help="Prefix target file names with short SHA-1 for stable uniqueness.",
    )
    parser.add_argument(
        "--dhash-threshold",
        type=int,
        default=-1,
        help=(
            "Optional per-class near-duplicate filtering threshold on dHash. "
            "-1 disables, 0 removes exact dHash duplicates, 1 removes Hamming<=1."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap for quick dry runs (0 = scan all files).",
    )
    return parser.parse_args()


def parse_csv_tokens(raw: str) -> list[str]:
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def parse_ratios(raw: str) -> tuple[float, float, float]:
    tokens = parse_csv_tokens(raw)
    if len(tokens) != 3:
        raise ValueError("--ratios must contain exactly three comma-separated values")
    train, val, test = (float(tok) for tok in tokens)
    total = train + val + test
    if total <= 0.0:
        raise ValueError("--ratios must sum to a positive value")
    train /= total
    val /= total
    test /= total
    return train, val, test


def iter_source_images(dataset_root: Path, source_splits: list[str]) -> Iterable[tuple[str, str, Path]]:
    for split in source_splits:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            class_name = class_dir.name
            for path in class_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                    yield split, class_name, path


def sha1_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def dhash64(path: Path) -> int | None:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "Pillow is required for --dhash-threshold. Install it and rerun."
        ) from exc

    try:
        with Image.open(path) as image:
            gray = image.convert("L").resize((9, 8))
            pixels = list(gray.getdata())
    except Exception:
        return None

    bits = 0
    for y in range(8):
        row = pixels[y * 9 : (y + 1) * 9]
        for x in range(8):
            bits = (bits << 1) | (1 if row[x] > row[x + 1] else 0)
    return bits


def is_within_hamming_threshold(value: int, seen: set[int], threshold: int) -> bool:
    if value in seen:
        return True
    if threshold <= 0:
        return False
    if threshold == 1:
        for bit in range(64):
            if (value ^ (1 << bit)) in seen:
                return True
        return False
    raise ValueError("Only dHash thresholds -1, 0, and 1 are supported")


def split_counts_for_class(n_items: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if n_items <= 0:
        return 0, 0, 0
    if n_items == 1:
        return 1, 0, 0
    if n_items == 2:
        return 1, 0, 1

    train_ratio, val_ratio, test_ratio = ratios
    train_n = int(n_items * train_ratio)
    val_n = int(n_items * val_ratio)
    test_n = n_items - train_n - val_n

    train_n = max(train_n, 1)
    val_n = max(val_n, 1)
    test_n = max(test_n, 1)

    while train_n + val_n + test_n > n_items:
        if train_n >= val_n and train_n >= test_n and train_n > 1:
            train_n -= 1
        elif val_n >= test_n and val_n > 1:
            val_n -= 1
        elif test_n > 1:
            test_n -= 1
        else:
            break

    while train_n + val_n + test_n < n_items:
        train_n += 1

    return train_n, val_n, test_n


def safe_materialize(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
        return

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def unique_target_path(base_dir: Path, file_name: str) -> Path:
    candidate = base_dir / file_name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    index = 1
    while True:
        alt = base_dir / f"{stem}_{index}{suffix}"
        if not alt.exists():
            return alt
        index += 1


def main() -> None:
    args = parse_args()

    source_splits = parse_csv_tokens(args.source_splits)
    if not source_splits:
        raise ValueError("At least one source split must be provided")

    if args.dhash_threshold not in {-1, 0, 1}:
        raise ValueError("--dhash-threshold supports only -1, 0, or 1")

    ratios = parse_ratios(args.ratios)
    rng = random.Random(args.seed)

    by_sha1: dict[str, CanonicalItem] = {}
    duplicate_sources: dict[str, int] = defaultdict(int)
    class_conflicts: list[dict[str, str]] = []

    scanned = 0
    for source_split, class_name, path in iter_source_images(args.dataset_root, source_splits):
        scanned += 1
        if args.max_files > 0 and scanned > args.max_files:
            break

        digest = sha1_file(path)
        existing = by_sha1.get(digest)
        if existing is None:
            by_sha1[digest] = CanonicalItem(
                sha1=digest,
                class_name=class_name,
                source_split=source_split,
                source_path=path,
            )
            continue

        duplicate_sources[digest] += 1
        if existing.class_name != class_name:
            class_conflicts.append(
                {
                    "sha1": digest,
                    "kept_class": existing.class_name,
                    "conflict_class": class_name,
                    "kept_path": str(existing.source_path),
                    "conflict_path": str(path),
                }
            )

    canonical_items = list(by_sha1.values())

    # Optional near-duplicate filtering via dHash.
    dhash_removed = 0
    dhash_skipped = 0
    if args.dhash_threshold >= 0:
        grouped: dict[str, list[CanonicalItem]] = defaultdict(list)
        for item in canonical_items:
            grouped[item.class_name].append(item)

        filtered: list[CanonicalItem] = []
        for class_name, class_items in grouped.items():
            seen_hashes: set[int] = set()
            for item in sorted(class_items, key=lambda it: it.sha1):
                value = dhash64(item.source_path)
                if value is None:
                    filtered.append(item)
                    dhash_skipped += 1
                    continue
                if is_within_hamming_threshold(value, seen_hashes, args.dhash_threshold):
                    dhash_removed += 1
                    continue
                seen_hashes.add(value)
                filtered.append(item)
        canonical_items = filtered

    by_class: dict[str, list[CanonicalItem]] = defaultdict(list)
    for item in canonical_items:
        by_class[item.class_name].append(item)

    split_sha1: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    split_class_counts: dict[str, dict[str, int]] = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int),
    }

    assignments: list[dict[str, str]] = []

    for class_name in sorted(by_class.keys()):
        items = sorted(by_class[class_name], key=lambda it: it.sha1)
        rng.shuffle(items)

        train_n, val_n, test_n = split_counts_for_class(len(items), ratios)
        train_items = items[:train_n]
        val_items = items[train_n : train_n + val_n]
        test_items = items[train_n + val_n : train_n + val_n + test_n]

        for split_name, bucket in (("train", train_items), ("val", val_items), ("test", test_items)):
            for item in bucket:
                split_sha1[split_name].add(item.sha1)
                split_class_counts[split_name][class_name] += 1

                base_name = item.source_path.name
                if args.prefix_sha1:
                    base_name = f"{item.sha1[:12]}_{base_name}"

                target_rel = Path(split_name) / class_name / base_name
                assignments.append(
                    {
                        "sha1": item.sha1,
                        "class_name": class_name,
                        "source_split": item.source_split,
                        "source_path": str(item.source_path),
                        "assigned_split": split_name,
                        "target_relpath": str(target_rel),
                    }
                )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sha1",
                "class_name",
                "source_split",
                "source_path",
                "assigned_split",
                "target_relpath",
            ],
        )
        writer.writeheader()
        writer.writerows(assignments)

    if args.materialize:
        for record in assignments:
            target_rel = Path(record["target_relpath"])
            target_dir = args.output_root / target_rel.parent
            target_path = unique_target_path(target_dir, target_rel.name)
            safe_materialize(Path(record["source_path"]), target_path, args.link_mode)

    overlap_tv = len(split_sha1["train"] & split_sha1["val"])
    overlap_tt = len(split_sha1["train"] & split_sha1["test"])
    overlap_vt = len(split_sha1["val"] & split_sha1["test"])

    summary = {
        "config": {
            "dataset_root": str(args.dataset_root),
            "source_splits": source_splits,
            "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
            "seed": args.seed,
            "dhash_threshold": args.dhash_threshold,
            "materialize": bool(args.materialize),
            "output_root": str(args.output_root),
            "link_mode": args.link_mode,
        },
        "scan": {
            "files_scanned": scanned if args.max_files == 0 else min(scanned, args.max_files),
            "unique_sha1_items": len(by_sha1),
            "exact_duplicates_removed": int(sum(duplicate_sources.values())),
            "class_conflicts": len(class_conflicts),
            "class_conflict_examples": class_conflicts[:20],
            "dhash_removed": dhash_removed,
            "dhash_unreadable_skipped": dhash_skipped,
            "final_unique_items": len(assignments),
        },
        "split_counts": {
            split: {
                "total": int(sum(per_class.values())),
                "per_class": {k: int(v) for k, v in sorted(per_class.items())},
            }
            for split, per_class in split_class_counts.items()
        },
        "sha1_overlap_check": {
            "train_val": overlap_tv,
            "train_test": overlap_tt,
            "val_test": overlap_vt,
        },
        "outputs": {
            "manifest": str(args.manifest),
            "summary": str(args.summary),
        },
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Leakage-free split generation complete.")
    print(f"Scanned files: {summary['scan']['files_scanned']}")
    print(f"Unique SHA-1 items: {summary['scan']['unique_sha1_items']}")
    print(f"Exact duplicates removed: {summary['scan']['exact_duplicates_removed']}")
    if args.dhash_threshold >= 0:
        print(f"dHash-filtered removals: {summary['scan']['dhash_removed']}")
    print(
        "SHA-1 overlaps after assignment: "
        f"train-val={overlap_tv}, train-test={overlap_tt}, val-test={overlap_vt}"
    )
    print(f"Manifest: {args.manifest}")
    print(f"Summary : {args.summary}")
    if args.materialize:
        print(f"Materialized split tree: {args.output_root}")


if __name__ == "__main__":
    main()
