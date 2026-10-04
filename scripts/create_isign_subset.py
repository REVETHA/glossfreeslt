"""Create a deterministic, copy-only iSign subset in repository LMDB format."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import pickle
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lmdb
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils
from dataloader.datasets import S2T_Dataset


DEFAULT_SOURCE = Path(r"D:\iSign_LMDB")
DEFAULT_OUTPUT = Path(r"D:\iSign_LMDB_subset_16k")
DEFAULT_TARGETS = {"train": 16_000, "dev": 2_000, "test": 2_000}
PROGRESS_FILE = ".create_isign_subset.json"
STAGING_DIR = ".staging"


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rb") as stream:
        records = pickle.load(stream)
    if not isinstance(records, dict):
        raise ValueError(f"Expected a pickled label dictionary in {path}")
    return records


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_digest(selections: dict[str, list[str]]) -> str:
    serialized = json.dumps(
        {split: sorted(uids) for split, uids in selections.items()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid subset manifest: {path}")
    return value


def create_manifest(path: Path, manifest: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


def inspect_sample(sample_dir: Path) -> int:
    data_file = sample_dir / "data.mdb"
    if not data_file.is_file() or data_file.stat().st_size == 0:
        raise ValueError(f"Missing or empty LMDB data file: {data_file}")

    environment = lmdb.open(
        str(sample_dir), readonly=True, lock=False, readahead=False, subdir=True
    )
    try:
        with environment.begin(write=False) as transaction:
            protocol_raw = transaction.get(b"protocol")
            if protocol_raw is None:
                raise ValueError(f"Missing b'protocol' in {sample_dir}")
            protocol = pickle.loads(protocol_raw)
            if not isinstance(protocol, int):
                raise ValueError(f"Invalid pickle protocol in {sample_dir}: {protocol!r}")

            keys_raw = transaction.get(pickle.dumps("keys", protocol=protocol))
            if keys_raw is None:
                raise ValueError(f"Missing pickled 'keys' catalog in {sample_dir}")
            keys = pickle.loads(keys_raw)
            if not isinstance(keys, list) or not keys:
                raise ValueError(f"Invalid or empty frame-key catalog in {sample_dir}")

            frame_ids: list[int] = []
            for encoded_key in keys:
                if not isinstance(encoded_key, bytes):
                    raise ValueError(f"Invalid frame key in {sample_dir}: {encoded_key!r}")
                frame_id = pickle.loads(encoded_key)
                if not isinstance(frame_id, int):
                    raise ValueError(f"Invalid frame index in {sample_dir}: {frame_id!r}")
                frame_ids.append(frame_id)
            if frame_ids != list(range(len(frame_ids))):
                raise ValueError(f"Non-contiguous frame keys in {sample_dir}")

            for frame_id in (frame_ids[0], frame_ids[-1]):
                payload = transaction.get(pickle.dumps(frame_id, protocol=protocol))
                if payload is None:
                    raise ValueError(f"Missing frame {frame_id} in {sample_dir}")
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    if image.size != (256, 256):
                        raise ValueError(
                            f"Unexpected frame size in {sample_dir}: {image.size}"
                        )
                    if image.mode != "RGB":
                        raise ValueError(f"Expected RGB frame in {sample_dir}, got {image.mode}")

            expected_payload_keys = {
                "protocol",
                pickle.dumps("keys", protocol=protocol).hex(),
                *(key.hex() for key in keys),
            }
            if transaction.get(b"metadata") is not None:
                expected_payload_keys.add(b"metadata".hex())
            observed_keys = {
                key.decode("latin-1") for key, _ in transaction.cursor()
            }
            expected_as_text = {
                key if key == "protocol" else bytes.fromhex(key).decode("latin-1")
                for key in expected_payload_keys
            }
            if observed_keys != expected_as_text:
                raise ValueError(f"LMDB has missing or unexpected records: {sample_dir}")
            return len(frame_ids)
    finally:
        environment.close()


def copy_sample(source: Path, destination: Path, staging: Path) -> bool:
    if destination.exists():
        inspect_sample(destination)
        if source_file_inventory(source) != source_file_inventory(destination):
            raise ValueError(f"Existing destination sample differs from source: {destination}")
        return False

    staging.mkdir(parents=True, exist_ok=True)
    source_inventory = source_file_inventory(source)
    staged_inventory = source_file_inventory(staging)
    extra_files = set(staged_inventory) - set(source_inventory)
    if extra_files:
        raise ValueError(f"Unexpected staged files in {staging}: {sorted(extra_files)[:5]}")
    for relative_path, expected_size in source_inventory.items():
        source_file = source / relative_path
        staged_file = staging / relative_path
        if staged_file.exists():
            if staged_file.stat().st_size != expected_size:
                raise ValueError(f"Partial staged file differs; refusing to overwrite: {staged_file}")
            continue
        staged_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, staged_file)
        if staged_file.stat().st_size != expected_size:
            raise ValueError(f"Copied file size mismatch: {staged_file}")
    inspect_sample(staging)
    if source_inventory != source_file_inventory(staging):
        raise ValueError(f"Copied sample file inventory differs from source: {staging}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)
    return True


def source_file_inventory(directory: Path) -> dict[str, int]:
    return {
        path.relative_to(directory).as_posix(): path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file()
    }


def write_labels(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as raw_stream:
        with gzip.GzipFile(fileobj=raw_stream, mode="wb", mtime=0) as stream:
            pickle.dump(records, stream, protocol=pickle.DEFAULT_PROTOCOL)


def tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def validate_output(
    output_root: Path,
    selected_records: dict[str, dict[str, dict[str, Any]]],
    sample_counts: dict[str, int],
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for split, selected in selected_records.items():
        labels_path = output_root / "labels" / f"labels.{split}"
        loaded = utils.load_dataset_file(str(labels_path))
        if loaded != selected:
            raise ValueError(f"Reduced labels do not exactly match selection: {labels_path}")
        sample_root = output_root / split
        actual_uids = {path.name for path in sample_root.iterdir() if path.is_dir()}
        if actual_uids != set(selected):
            missing = sorted(set(selected) - actual_uids)
            extra = sorted(actual_uids - set(selected))
            raise ValueError(
                f"{split} sample mismatch: missing={len(missing)}, extra={len(extra)}"
            )
        if len(loaded) != sample_counts[split] or len(actual_uids) != sample_counts[split]:
            raise ValueError(f"{split} has an unexpected label or sample count")
        sizes[split] = tree_size(sample_root)
    sizes["labels"] = tree_size(output_root / "labels")
    return sizes


def validate_active_loader(
    output_root: Path, selected_records: dict[str, dict[str, dict[str, Any]]]
) -> None:
    args = type("LoaderArgs", (), {"input_size": 224, "resize": 256})()
    config = {
        "data": {
            "dataset_name": "isign",
            "img_lmdb_path": str(output_root),
            "max_length": 128,
        }
    }
    for split, records in selected_records.items():
        dataset = object.__new__(S2T_Dataset)
        dataset.config = config
        dataset.args = args
        dataset.training_refurbish = False
        dataset.raw_data = records
        dataset.tokenizer = None
        dataset.img_lmdb_path = str(output_root)
        dataset.phase = split
        dataset.max_length = 128
        dataset.list = list(records)
        dataset.dataset_name = "isign"
        dataset.seq = lambda images: images
        dataset.seq_color = None
        dataset.img_len = 0

        for index, uid in enumerate(dataset.list[:3]):
            name, frames, text, frame_count = dataset[index]
            if name != f"{split}/{uid}" or text != records[uid]["text"]:
                raise ValueError(f"Active loader returned wrong name/text for {split}/{uid}")
            if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
                raise ValueError(f"Active loader did not return frame tensors for {split}/{uid}")
            if frames.shape[1:] != (3, 224, 224) or frame_count != frames.shape[0]:
                raise ValueError(f"Unexpected active-loader frame shape for {split}/{uid}")
            if not 1 <= frame_count <= 128:
                raise ValueError(f"Invalid frame count for {split}/{uid}: {frame_count}")
            if not torch.isfinite(frames).all().item():
                raise ValueError(f"Non-finite frame values for {split}/{uid}")
            print(
                f"[LOADER] {split}/{uid}: text={text!r}, "
                f"frames={frame_count}, tensor={tuple(frames.shape)}"
            )
            del frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=DEFAULT_TARGETS["train"])
    parser.add_argument("--dev-count", type=int, default=DEFAULT_TARGETS["dev"])
    parser.add_argument("--test-count", type=int, default=DEFAULT_TARGETS["test"])
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = {
        "train": args.train_count,
        "dev": args.dev_count,
        "test": args.test_count,
    }
    if any(count <= 0 for count in targets.values()) or args.progress_every <= 0:
        parser.error("sample counts and --progress-every must be positive")

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if (
        source_root == output_root
        or source_root in output_root.parents
        or output_root in source_root.parents
    ):
        parser.error("source and output must be separate, non-nested directories")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset directory does not exist: {source_root}")

    if output_root.exists():
        if not args.resume:
            raise FileExistsError(
                f"Output already exists; refusing to modify it: {output_root}. "
                "Use --resume only for a dataset previously created by this script."
            )
        manifest_path = output_root / PROGRESS_FILE
        if not manifest_path.is_file():
            raise FileExistsError(
                f"Output exists without this script's manifest; refusing to modify it: "
                f"{output_root}"
            )
    elif args.resume:
        raise FileNotFoundError(f"Cannot resume; output directory does not exist: {output_root}")

    label_paths = {
        split: source_root / "labels" / f"labels.{split}" for split in targets
    }
    source_records = {split: load_labels(path) for split, path in label_paths.items()}

    selections: dict[str, list[str]] = {}
    selected_records: dict[str, dict[str, dict[str, Any]]] = {}
    for split, count in targets.items():
        records = source_records[split]
        if len(records) != len(set(records)):
            raise ValueError(f"Duplicate label keys in {label_paths[split]}")
        if len(records) < count:
            raise ValueError(
                f"{split} has only {len(records)} labels; {count} were requested"
            )
        available = sorted(records)
        selected = random.Random(args.seed).sample(available, count)
        selected_set = set(selected)
        selections[split] = selected
        selected_records[split] = {
            uid: record for uid, record in records.items() if uid in selected_set
        }

    label_hashes = {split: file_sha256(path) for split, path in label_paths.items()}
    manifest = {
        "format": "isign-subset-v1",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "targets": targets,
        "source_label_sha256": label_hashes,
        "selection_sha256": selection_digest(selections),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = output_root / PROGRESS_FILE
    if output_root.exists():
        existing_manifest = read_manifest(manifest_path)
        for key, value in manifest.items():
            if key != "created_at_utc" and existing_manifest.get(key) != value:
                raise ValueError(f"Cannot resume: manifest differs for {key}")

    missing_samples: list[str] = []
    for split, selected in selected_records.items():
        sample_root = source_root / split
        if not sample_root.is_dir():
            raise FileNotFoundError(f"Missing source split directory: {sample_root}")
        for uid, record in selected.items():
            if Path(uid).name != uid or uid in {".", ".."}:
                raise ValueError(f"Invalid sample ID in {split} labels: {uid!r}")
            if record.get("name") != f"{split}/{uid}":
                raise ValueError(f"Invalid label name for {split}/{uid}: {record.get('name')!r}")
            sample_dir = sample_root / uid
            if not sample_dir.is_dir() or not (sample_dir / "data.mdb").is_file():
                missing_samples.append(f"{split}/{uid}")
    if missing_samples:
        raise FileNotFoundError(
            f"Missing {len(missing_samples)} selected source samples; first: "
            f"{', '.join(missing_samples[:10])}"
        )

    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print(f"Seed: {args.seed}")
    print(f"Selected samples: {targets}")
    if args.dry_run:
        print(f"[DRY RUN] Source labels and all {sum(targets.values())} selected sample folders exist.")
        return 0

    if not output_root.exists():
        output_root.mkdir(parents=True)
        create_manifest(manifest_path, manifest)
    for split in targets:
        split_dir = output_root / split
        if split_dir.exists() and not split_dir.is_dir():
            raise NotADirectoryError(f"Output split path is not a directory: {split_dir}")
        split_dir.mkdir(exist_ok=True)

    for split, records in selected_records.items():
        labels_path = output_root / "labels" / f"labels.{split}"
        if labels_path.exists():
            if utils.load_dataset_file(str(labels_path)) != records:
                raise FileExistsError(
                    f"Existing reduced label file differs; refusing to overwrite: {labels_path}"
                )
        else:
            write_labels(labels_path, records)

    counts = {
        split: {"selected": len(records), "copied": 0, "missing": 0, "failed": 0, "skipped": 0}
        for split, records in selected_records.items()
    }
    for split, records in selected_records.items():
        staging_root = output_root / STAGING_DIR / split
        staging_root.mkdir(parents=True, exist_ok=True)
        for index, (uid, _) in enumerate(records.items(), start=1):
            source_sample = source_root / split / uid
            destination_sample = output_root / split / uid
            staging_sample = staging_root / uid
            try:
                copied = copy_sample(source_sample, destination_sample, staging_sample)
                counts[split]["copied" if copied else "skipped"] += 1
            except Exception as error:
                counts[split]["failed"] += 1
                print(f"[{split.upper()}] ERROR {uid}: {type(error).__name__}: {error}", flush=True)
                print(f"Progress summary before stop: {counts}", flush=True)
                raise
            if index % args.progress_every == 0 or index == len(records):
                print(f"[{split.upper()}] {index}/{len(records)}", flush=True)

    sizes = validate_output(output_root, selected_records, targets)
    validate_active_loader(output_root, selected_records)
    sizes["total"] = sum(sizes.values())
    print(f"Copy summary: {counts}")
    print("Post-creation validation: labels, sample directories, and loader checks passed.")
    for split in ("train", "dev", "test", "labels", "total"):
        print(f"{split.title()} size: {sizes[split]:,} bytes ({sizes[split] / (1024 ** 3):.3f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
