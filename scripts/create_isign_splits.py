"""Create and validate the reproducible iSign v1.1 group-aware split manifests.

This script reads the source CSV and MP4 directory only.  It never alters either.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DATASET_NAME = "iSign v1.1"
PROTOCOL = "isign-v1.1-group-split-v1"
SEED = 42
TARGET_ROWS = {"train": 101_790, "dev": 12_724, "test": 12_723}
SPLITS = tuple(TARGET_ROWS)
PARSER_RULES = [
    r"^(?P<source_video_id>.+)--(?P<segment>\d+)$",
    r"^(?P<source_video_id>.+)-(?P<segment>\d+)$",
    r"^(?P<source_video_id>.+)_(?P<segment>[A-Za-z]\d*)$",
]
UID_PATTERNS = tuple(re.compile(rule) for rule in PARSER_RULES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_video_id(uid: str) -> str:
    for pattern in UID_PATTERNS:
        match = pattern.fullmatch(uid)
        if match:
            return match.group("source_video_id")
    raise ValueError(f"UID does not match a supported iSign v1.1 pattern: {uid!r}")


def group_rank(group_id: str) -> bytes:
    value = f"{PROTOCOL}:{SEED}:{group_id}".encode("utf-8")
    return hashlib.sha256(value).digest()


def tie_rank(group_id: str, split: str) -> bytes:
    value = f"{PROTOCOL}:{SEED}:{group_id}:{split}".encode("utf-8")
    return hashlib.sha256(value).digest()


def choose_split(group_id: str, assigned_rows: Counter) -> str:
    """Choose the split with the largest remaining fraction of its target."""
    return min(
        SPLITS,
        key=lambda split: (
            -((TARGET_ROWS[split] - assigned_rows[split]) / TARGET_ROWS[split]),
            tie_rank(group_id, split),
            split,
        ),
    )


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["uid", "text"]:
            raise ValueError(f"Expected CSV columns ['uid', 'text'], got {reader.fieldnames!r}")
        return list(reader)


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    workspace_root = Path(__file__).resolve().parents[2]
    dataset_root = workspace_root / "dataset" / "iSign"
    csv_path = dataset_root / "iSign_v1.1.csv"
    video_dir = dataset_root / "videos" / "iSign-videos_v1.1"
    output_dir = dataset_root / "splits" / "v1"

    rows = read_rows(csv_path)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[source_video_id(row["uid"])].append(row)

    assigned_rows: Counter = Counter()
    assigned_groups: Counter = Counter()
    group_split: dict[str, str] = {}
    for group_id, group_rows in sorted(groups.items(), key=lambda item: (-len(item[1]), group_rank(item[0]))):
        split = choose_split(group_id, assigned_rows)
        group_split[group_id] = split
        assigned_rows[split] += len(group_rows)
        assigned_groups[split] += 1

    split_rows = {split: [] for split in SPLITS}
    manifest_rows = []
    for row in rows:
        group_id = source_video_id(row["uid"])
        split = group_split[group_id]
        split_rows[split].append(
            {"uid": row["uid"], "text": row["text"], "source_video_id": group_id}
        )
        manifest_rows.append({"uid": row["uid"], "source_video_id": group_id, "split": split})

    # Validate the source videos before committing output artifacts.
    uid_set = {row["uid"] for row in rows}
    video_stems = {path.stem for path in video_dir.glob("*.mp4")}
    missing_videos = uid_set - video_stems
    unreferenced_videos = video_stems - uid_set
    if missing_videos:
        raise RuntimeError(f"Found {len(missing_videos)} UIDs without a corresponding MP4 file")

    split_uid_sets = {split: {row["uid"] for row in split_rows[split]} for split in SPLITS}
    if len(uid_set) != len(rows):
        raise RuntimeError("Duplicate UIDs found in the source CSV")
    if sum(len(split_uid_sets[split]) for split in SPLITS) != len(rows):
        raise RuntimeError("A UID is duplicated or missing across split assignments")
    if set().union(*split_uid_sets.values()) != uid_set:
        raise RuntimeError("Split assignments do not cover exactly the source CSV UIDs")
    source_splits: dict[str, set[str]] = defaultdict(set)
    for manifest_row in manifest_rows:
        source_splits[manifest_row["source_video_id"]].add(manifest_row["split"])
    leaked_groups = [group_id for group_id, split_set in source_splits.items() if len(split_set) != 1]
    if leaked_groups:
        raise RuntimeError(f"Found {len(leaked_groups)} source-video groups in multiple splits")
    if dict(assigned_rows) != TARGET_ROWS:
        raise RuntimeError(f"Could not meet exact row targets: {dict(assigned_rows)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "split_manifest.csv"
    write_csv(manifest_path, ("uid", "source_video_id", "split"), manifest_rows)
    split_paths = {}
    for split in SPLITS:
        path = output_dir / f"{split}.csv"
        write_csv(path, ("uid", "text", "source_video_id"), split_rows[split])
        split_paths[split] = path

    metadata = {
        "dataset_name": DATASET_NAME,
        "split_protocol": PROTOCOL,
        "random_seed": SEED,
        "source_csv": str(csv_path),
        "source_csv_sha256": sha256_file(csv_path),
        "uid_parser_rules_in_precedence_order": PARSER_RULES,
        "total_rows": len(rows),
        "total_source_video_groups": len(groups),
        "target_row_counts": TARGET_ROWS,
        "actual_row_counts": {split: len(split_rows[split]) for split in SPLITS},
        "source_video_group_counts": dict(assigned_groups),
        "missing_video_count": len(missing_videos),
        "unreferenced_video_count": len(unreferenced_videos),
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    generated_paths = [manifest_path, *(split_paths[split] for split in SPLITS), metadata_path]
    print("Validation passed")
    print("row_counts=", metadata["actual_row_counts"])
    print("group_counts=", metadata["source_video_group_counts"])
    print("missing_videos=", len(missing_videos))
    print("unreferenced_videos=", len(unreferenced_videos))
    for path in generated_paths:
        print(f"sha256 {path.name} {sha256_file(path)}")


if __name__ == "__main__":
    main()
