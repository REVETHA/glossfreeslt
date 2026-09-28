"""Scalable, resumable iSign v1.1 MP4 to LMDB preparation pipeline.

Converts iSign v1.1 MP4 videos to the ImageDatabase-compatible LMDB format
consumed by S2T_Dataset.load_imgs_lmdb() and generates dataset label files.

Directory structure:
  <output_root>/<split>/<uid>/data.mdb
                             /lock.mdb

LMDB schema per sample:
  - key b"protocol"                               -> pickle.dumps(4)
  - key pickle.dumps(i, protocol=4)               -> JPEG bytes (256x256, RGB)
  - key pickle.dumps("keys", protocol=4)          -> pickle.dumps([pickle.dumps(i)...])

Labels generated:
  <labels_dir>/labels.train
  <labels_dir>/labels.dev
  <labels_dir>/labels.test
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import lmdb
import numpy as np
import torch
from PIL import Image

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils
from dataloader.database import ImageDatabase
from dataloader.datasets import S2T_Dataset


FRAME_SIZE = (256, 256)
TOTAL_ISIGN_SAMPLES = 127_237
SPLIT_TOTALS = {
    "train": 101_790,
    "dev": 12_724,
    "test": 12_723,
}


def is_sample_valid(
    sample_dir: Path,
    expected_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, int, int]:
    """Check if an existing sample LMDB is fully valid.

    Returns:
        (is_valid, num_frames, total_disk_bytes)
    """
    data_file = sample_dir / "data.mdb"
    if not sample_dir.is_dir() or not data_file.is_file() or data_file.stat().st_size == 0:
        return False, 0, 0

    env = None
    try:
        env = lmdb.open(str(sample_dir), readonly=True, lock=False, max_spare_txns=16)
        with env.begin() as txn:
            proto_raw = txn.get(b"protocol")
            if proto_raw is None:
                return False, 0, 0
            proto = pickle.loads(proto_raw)

            keys_raw = txn.get(pickle.dumps("keys", protocol=proto))
            if keys_raw is None:
                return False, 0, 0
            keys = pickle.loads(keys_raw)
            num_frames = len(keys)
            if num_frames == 0:
                return False, 0, 0

            if expected_metadata is not None:
                metadata_raw = txn.get(b"metadata")
                if metadata_raw is None:
                    return False, 0, 0
                metadata = pickle.loads(metadata_raw)
                for key, expected_value in expected_metadata.items():
                    if metadata.get(key) != expected_value:
                        return False, 0, 0

            # Verify first and last frame decodability
            first_raw = txn.get(pickle.dumps(0, protocol=proto))
            last_raw = txn.get(pickle.dumps(num_frames - 1, protocol=proto))
            if first_raw is None or last_raw is None:
                return False, 0, 0

            img0 = Image.open(io.BytesIO(first_raw))
            img_last = Image.open(io.BytesIO(last_raw))
            if img0.size != FRAME_SIZE or img_last.size != FRAME_SIZE:
                return False, 0, 0

        total_bytes = sum(f.stat().st_size for f in sample_dir.iterdir() if f.is_file())
        return True, num_frames, total_bytes
    except Exception:
        return False, 0, 0
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def read_sample_metadata(sample_dir: Path) -> Dict[str, Any]:
    """Read optional preparation metadata from a valid sample LMDB."""
    env = None
    try:
        env = lmdb.open(str(sample_dir), readonly=True, lock=False)
        with env.begin() as txn:
            raw = txn.get(b"metadata")
            return pickle.loads(raw) if raw is not None else {}
    except Exception:
        return {}
    finally:
        if env is not None:
            env.close()


def uniform_sample_indices(total_frames: int, max_frames: int) -> List[int]:
    """Return unique, sorted endpoints-inclusive temporal sample indices."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")

    sample_count = min(total_frames, max_frames)
    indices = np.linspace(0, total_frames - 1, sample_count, dtype=np.int64).tolist()
    if len(indices) != len(set(indices)):
        raise AssertionError("Uniform sampling produced duplicate frame indices")
    if indices != sorted(indices) or indices[0] != 0 or indices[-1] != total_frames - 1:
        raise AssertionError("Uniform sampling did not preserve sorted endpoints")
    return indices


def _encode_frame(frame_bgr: np.ndarray, jpeg_quality: int) -> bytes:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb).resize(FRAME_SIZE, Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=jpeg_quality)
    return buf.getvalue()


def extract_and_encode_frames_with_metadata(
    video_path: Path,
    jpeg_quality: int = 95,
    max_frames: Optional[int] = None,
    sampling: str = "all",
) -> Tuple[List[bytes], Dict[str, Any]]:
    """Decode, sample, resize, and JPEG encode frames with sampling metadata."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if sampling not in ("all", "uniform"):
        raise ValueError(f"Unsupported sampling mode: {sampling}")
    if sampling == "uniform" and max_frames is None:
        raise ValueError("--max-frames is required when --sampling uniform is used")

    if sampling == "all" or max_frames is None:
        selected_indices: List[int] = []
        frames_jpeg: List[bytes] = []
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")
        try:
            frame_index = 0
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frames_jpeg.append(_encode_frame(frame_bgr, jpeg_quality))
                selected_indices.append(frame_index)
                frame_index += 1
        finally:
            cap.release()

        if not frames_jpeg:
            raise ValueError(f"Decoded zero frames from: {video_path}")
        return frames_jpeg, {
            "original_frame_count": len(frames_jpeg),
            "stored_frame_count": len(frames_jpeg),
            "sampling": "all",
            "max_frames": None,
            "jpeg_quality": jpeg_quality,
            "selected_indices": selected_indices,
        }

    # Count first, then decode a second time while retaining only selected frames.
    # This avoids keeping an entire long video in memory and uses the actual decoded count.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    original_frame_count = 0
    try:
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            original_frame_count += 1
    finally:
        cap.release()

    selected_indices = uniform_sample_indices(original_frame_count, max_frames)
    selected_set = set(selected_indices)
    frames_jpeg = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not reopen video: {video_path}")
    try:
        frame_index = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_index in selected_set:
                frames_jpeg.append(_encode_frame(frame_bgr, jpeg_quality))
            frame_index += 1
    finally:
        cap.release()

    if len(frames_jpeg) != len(selected_indices):
        raise RuntimeError(
            f"Selected {len(frames_jpeg)} frames but expected {len(selected_indices)} from {video_path}"
        )
    return frames_jpeg, {
        "original_frame_count": original_frame_count,
        "stored_frame_count": len(frames_jpeg),
        "sampling": "uniform",
        "max_frames": max_frames,
        "jpeg_quality": jpeg_quality,
        "selected_indices": selected_indices,
    }


def extract_and_encode_frames(video_path: Path, jpeg_quality: int = 95) -> List[bytes]:
    """Preserve the legacy all-frame extraction API."""
    frames_jpeg, _ = extract_and_encode_frames_with_metadata(video_path, jpeg_quality=jpeg_quality)
    return frames_jpeg


def write_sample_lmdb(
    sample_dir: Path,
    frames_jpeg: List[bytes],
    protocol: int = pickle.DEFAULT_PROTOCOL,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """Write an ImageDatabase-compatible LMDB for a single sample."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    num_frames = len(frames_jpeg)
    # Leave room for LMDB pages, frame keys, the key catalog, and optional metadata.
    payload_bytes = sum(len(b) for b in frames_jpeg)
    catalog_bytes = num_frames * 16 + 4096
    metadata_bytes = len(pickle.dumps(metadata, protocol=protocol)) if metadata is not None else 0
    overhead_bytes = 1024 * 1024 + catalog_bytes + metadata_bytes
    map_size = max(2 * 1024 * 1024, int((payload_bytes + overhead_bytes) * 1.5))
    map_size = ((map_size + 4095) // 4096) * 4096

    env = lmdb.open(str(sample_dir), map_size=map_size, subdir=True, lock=True)
    try:
        with env.begin(write=True) as txn:
            # 1. Protocol
            txn.put(b"protocol", pickle.dumps(protocol, protocol=protocol), dupdata=False)

            # 2. Frames
            list_of_keys = []
            for i, f_bytes in enumerate(frames_jpeg):
                key = pickle.dumps(i, protocol=protocol)
                txn.put(key, f_bytes, dupdata=False)
                list_of_keys.append(key)

            # 3. Keys catalog
            txn.put(
                pickle.dumps("keys", protocol=protocol),
                pickle.dumps(list_of_keys, protocol=protocol),
                dupdata=False,
            )
            if metadata is not None:
                txn.put(b"metadata", pickle.dumps(metadata, protocol=protocol), dupdata=False)
    finally:
        env.close()

    return sum(f.stat().st_size for f in sample_dir.iterdir() if f.is_file())


def process_single_sample(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function to process one video sample."""
    split: str = task["split"]
    uid: str = task["uid"]
    text: str = task["text"]
    video_path = Path(task["video_path"])
    sample_dir = Path(task["sample_dir"])
    jpeg_quality: int = task["jpeg_quality"]
    overwrite: bool = task["overwrite"]
    max_frames: Optional[int] = task["max_frames"]
    sampling: str = task["sampling"]
    expected_metadata = None
    if max_frames is not None or sampling != "all":
        expected_metadata = {
            "sampling": sampling,
            "max_frames": max_frames,
            "jpeg_quality": jpeg_quality,
        }

    start_time = time.time()

    # Resumability check
    if not overwrite:
        valid, existing_frames, existing_bytes = is_sample_valid(sample_dir, expected_metadata)
        if valid:
            existing_metadata = read_sample_metadata(sample_dir)
            return {
                "uid": uid,
                "split": split,
                "text": text,
                "status": "SKIPPED",
                "num_frames": existing_frames,
                "original_frame_count": existing_metadata.get("original_frame_count", existing_frames),
                "selected_indices": existing_metadata.get("selected_indices", []),
                "disk_bytes": existing_bytes,
                "duration_s": time.time() - start_time,
                "error": None,
            }

    # Clean up any partial or corrupt folder
    if sample_dir.exists():
        shutil.rmtree(sample_dir, ignore_errors=True)

    try:
        frames_jpeg, metadata = extract_and_encode_frames_with_metadata(
            video_path,
            jpeg_quality=jpeg_quality,
            max_frames=max_frames,
            sampling=sampling,
        )
        disk_bytes = write_sample_lmdb(sample_dir, frames_jpeg, metadata=metadata)
        return {
            "uid": uid,
            "split": split,
            "text": text,
            "status": "PROCESSED",
            "num_frames": len(frames_jpeg),
            "original_frame_count": metadata["original_frame_count"],
            "selected_indices": metadata["selected_indices"],
            "disk_bytes": disk_bytes,
            "duration_s": time.time() - start_time,
            "error": None,
        }
    except Exception as exc:
        if sample_dir.exists():
            shutil.rmtree(sample_dir, ignore_errors=True)
        return {
            "uid": uid,
            "split": split,
            "text": text,
            "status": "FAILED",
            "num_frames": 0,
            "original_frame_count": 0,
            "selected_indices": [],
            "disk_bytes": 0,
            "duration_s": time.time() - start_time,
            "error": str(exc),
        }


def read_split_csv(csv_path: Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Read a split CSV file and return rows."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def update_label_file(
    labels_file: Path,
    split: str,
    records: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Resumably update a gzipped pickle label file."""
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Dict[str, Any]] = {}
    if labels_file.is_file():
        try:
            with gzip.open(labels_file, "rb") as f:
                existing = pickle.load(f)
        except Exception:
            existing = {}

    for uid, info in records.items():
        existing[uid] = {
            "name": f"{split}/{uid}",
            "text": info["text"],
            "length": info["num_frames"],
            "gloss": "",
        }
        if "original_frame_count" in info:
            existing[uid]["original_length"] = info["original_frame_count"]
        if "sampling" in info:
            existing[uid]["sampling"] = info["sampling"]
        if "max_frames" in info:
            existing[uid]["max_frames"] = info["max_frames"]

    with gzip.open(labels_file, "wb") as f:
        pickle.dump(existing, f, protocol=pickle.DEFAULT_PROTOCOL)

    return existing


def validate_prepared_dataset(
    output_root: Path,
    labels_dir: Path,
    splits: List[str],
    processed_records: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Comprehensive validation suite for generated LMDBs and label files."""
    val_report: Dict[str, Any] = {
        "all_passed": True,
        "split_results": {},
        "errors": [],
    }

    print("\n" + "=" * 80)
    print("VALIDATION SUITE: ImageDatabase, Labels, and S2T_Dataset Verification")
    print("=" * 80)

    for split in splits:
        split_records = processed_records.get(split, {})
        labels_file = labels_dir / f"labels.{split}"
        print(f"\nValidating split [{split}] ({len(split_records)} samples):")

        # 1. Check label file
        if not labels_file.is_file():
            msg = f"Label file missing: {labels_file}"
            print(f"  [FAIL] {msg}")
            val_report["errors"].append(msg)
            val_report["all_passed"] = False
            continue

        try:
            loaded_labels = utils.load_dataset_file(str(labels_file))
            print(f"  [PASS] Successfully loaded {labels_file.name} with utils.load_dataset_file() ({len(loaded_labels)} total records)")
        except Exception as e:
            msg = f"Failed to load {labels_file}: {e}"
            print(f"  [FAIL] {msg}")
            val_report["errors"].append(msg)
            val_report["all_passed"] = False
            continue

        # 2. Validate every sample LMDB via ImageDatabase
        split_samples_ok = True
        for uid, sample_info in split_records.items():
            sample_lmdb_path = output_root / split / uid
            expected_frames = sample_info["num_frames"]

            try:
                db = ImageDatabase(path=str(sample_lmdb_path))
                if db.protocol != 4:
                    raise AssertionError(f"Protocol mismatch: {db.protocol} != 4")
                if len(db.keys) != expected_frames or len(db) != expected_frames:
                    raise AssertionError(f"Frame count mismatch: {len(db.keys)} != {expected_frames}")

                # Decode frame 0
                img0 = db[0]
                if not isinstance(img0, Image.Image) or img0.size != FRAME_SIZE or img0.mode != "RGB":
                    raise AssertionError(f"Invalid frame 0: size={img0.size}, mode={img0.mode}")

                # Decode last frame
                img_last = db[expected_frames - 1]
                if not isinstance(img_last, Image.Image) or img_last.size != FRAME_SIZE or img_last.mode != "RGB":
                    raise AssertionError(f"Invalid last frame: size={img_last.size}, mode={img_last.mode}")

            except Exception as e:
                msg = f"ImageDatabase validation failed for {split}/{uid}: {e}"
                print(f"    [FAIL] {msg}")
                val_report["errors"].append(msg)
                split_samples_ok = False

        if split_samples_ok:
            print(f"  [PASS] All {len(split_records)} samples open and decode cleanly via ImageDatabase")
        else:
            val_report["all_passed"] = False

        # 3. Validate S2T_Dataset pipeline loading
        config = {
            "data": {
                "img_lmdb_path": str(output_root),
                "max_length": 300,
                "dataset_name": "phoenix",
            }
        }
        dummy_args = SimpleNamespace(input_size=224, resize=256)

        try:
            dataset = S2T_Dataset(
                path=str(labels_file),
                tokenizer=None,
                config=config,
                args=dummy_args,
                phase=split,
                training_refurbish=False,
            )
            # Find an index corresponding to one of our processed UIDs
            sample_idx = 0
            for idx, key in enumerate(dataset.list):
                if key in split_records:
                    sample_idx = idx
                    break

            name_sample, img_sample, tgt_sample, img_len = dataset[sample_idx]
            target_uid = dataset.list[sample_idx]
            expected_sample = split_records[target_uid]

            if name_sample != f"{split}/{target_uid}":
                raise AssertionError(f"Name mismatch: {name_sample} != {split}/{target_uid}")
            if not isinstance(img_sample, torch.Tensor):
                raise AssertionError(f"Image tensor is not torch.Tensor: {type(img_sample)}")
            if img_sample.shape != (img_len, 3, 224, 224):
                raise AssertionError(f"Tensor shape mismatch: {img_sample.shape} != ({img_len}, 3, 224, 224)")
            if tgt_sample != expected_sample["text"]:
                raise AssertionError(f"Text mismatch: {tgt_sample!r} != {expected_sample['text']!r}")

            print(f"  [PASS] S2T_Dataset successfully loaded sample {name_sample} (tensor shape: {img_sample.shape})")
        except Exception as e:
            msg = f"S2T_Dataset pipeline test failed for split {split}: {e}"
            print(f"  [FAIL] {msg}")
            val_report["errors"].append(msg)
            val_report["all_passed"] = False

        val_report["split_results"][split] = split_samples_ok

    return val_report


def print_resource_and_throughput_report(
    summary_stats: Dict[str, Any],
    dataset_root: Path,
) -> None:
    """Print resource consumption, drive E: space analysis, and scaling estimates."""
    total_processed = summary_stats["processed"]
    total_skipped = summary_stats["skipped"]
    total_samples = total_processed + total_skipped
    total_frames = summary_stats["total_frames"]
    total_bytes = summary_stats["total_bytes"]
    total_wall_time = summary_stats["wall_time_s"]

    avg_bytes_per_sample = total_bytes / max(total_samples, 1)
    avg_mb_per_sample = avg_bytes_per_sample / (1024 ** 2)
    avg_frames_per_sample = total_frames / max(total_samples, 1)
    avg_kb_per_frame = (total_bytes / max(total_frames, 1)) / 1024

    # Disk Space Check on drive E:
    try:
        drive_total, drive_used, drive_free = shutil.disk_usage(str(dataset_root))
    except Exception:
        drive_total, drive_used, drive_free = 0, 0, 0

    free_gb = drive_free / (1024 ** 3)
    total_drive_gb = drive_total / (1024 ** 3)
    used_drive_gb = drive_used / (1024 ** 3)

    # Full Dataset Estimates
    estimated_full_bytes = TOTAL_ISIGN_SAMPLES * avg_bytes_per_sample
    estimated_full_gb = estimated_full_bytes / (1024 ** 3)
    free_headroom_gb = free_gb - estimated_full_gb

    # Throughput Estimates
    throughput_samples_per_sec = total_processed / max(total_wall_time, 0.001)
    throughput_frames_per_sec = total_frames / max(total_wall_time, 0.001)

    print("\n" + "=" * 80)
    print("PHASE 2B SUMMARY AND CAPACITY REPORT")
    print("=" * 80)
    print(f"Samples Processed          : {total_processed}")
    print(f"Samples Skipped (Valid)    : {total_skipped}")
    print(f"Samples Failed             : {summary_stats['failed']}")
    print(f"Total Samples Evaluated    : {total_samples}")
    print(f"Total Decoded Frames       : {total_frames:,}")
    print(f"Subset LMDB Disk Usage     : {total_bytes / (1024 ** 2):.2f} MB ({total_bytes / (1024 ** 3):.4f} GB)")
    print(f"Average Frames per Sample  : {avg_frames_per_sample:.1f}")
    print(f"Average Encoded Frame Size : {avg_kb_per_frame:.2f} KB")
    print(f"Average LMDB Size / Sample : {avg_mb_per_sample:.3f} MB")
    print("-" * 80)
    print("STORAGE CAPACITY ASSESSMENT (Drive E:)")
    print(f"Drive Total Capacity       : {total_drive_gb:.2f} GB")
    print(f"Drive Currently Used       : {used_drive_gb:.2f} GB")
    print(f"Drive Available Free Space : {free_gb:.2f} GB")
    print(f"Full Dataset Sample Count  : {TOTAL_ISIGN_SAMPLES:,}")
    print(f"Estimated Full LMDB Size   : {estimated_full_gb:.2f} GB")
    print(f"Projected Free Headroom    : {free_headroom_gb:.2f} GB")

    if free_headroom_gb > 20.0:
        fit_status = "SAFE (Fits with healthy headroom > 20 GB)"
    elif free_headroom_gb > 5.0:
        fit_status = "TIGHT (Fits, but low headroom < 20 GB)"
    else:
        fit_status = "INSUFFICIENT (Will not fit or exceeds safe drive threshold)"

    print(f"Drive Fit Verdict          : {fit_status}")
    print("-" * 80)
    print("THROUGHPUT AND PREPROCESSING TIME ESTIMATE")
    print(f"Measured Wall Clock Time   : {total_wall_time:.2f} s")
    print(f"Measured Throughput (Subset): {throughput_samples_per_sec:.2f} samples/sec ({throughput_frames_per_sec:.1f} frames/sec)")

    if throughput_samples_per_sec > 0:
        for workers in [4, 8, 16]:
            projected_rate = throughput_samples_per_sec * max(1, workers * 0.75)
            est_hours = (TOTAL_ISIGN_SAMPLES / projected_rate) / 3600
            print(f"  Estimated time with {workers} workers: ~{est_hours:.1f} hours ({est_hours * 60:.0f} mins)")
    else:
        print("  All evaluated samples were skipped from existing cache (0 newly processed).")
        print("  (Run with --overwrite to measure fresh encoding throughput.)")
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare iSign v1.1 LMDB dataset.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        help="Splits to process (train, dev, test)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Worker processes")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality (default: 95)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames per sample (default: all)")
    parser.add_argument("--sampling", choices=["all", "uniform"], default="all", help="Frame sampling policy (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="Max samples per split (default: all)")
    parser.add_argument("--limit-train", type=int, default=None, help="Max train samples")
    parser.add_argument("--limit-dev", type=int, default=None, help="Max dev samples")
    parser.add_argument("--limit-test", type=int, default=None, help="Max test samples")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing valid LMDBs")
    parser.add_argument("--validate-only", action="store_true", help="Run validation only without processing")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Path to iSign root")
    parser.add_argument("--output-root", type=Path, default=None, help="Output directory for LMDBs")
    parser.add_argument("--labels-dir", type=Path, default=None, help="Output directory for labels")

    args = parser.parse_args()

    workspace_root = REPO_ROOT.parent
    dataset_root = args.dataset_root or workspace_root / "dataset" / "iSign"
    splits_dir = dataset_root / "splits" / "v1"
    video_dir = dataset_root / "videos" / "iSign-videos_v1.1"
    output_root = args.output_root or dataset_root / "lmdb"
    labels_dir = args.labels_dir or dataset_root / "labels"

    print("=" * 80)
    print("iSign v1.1 LMDB Dataset Preparation Pipeline")
    print("=" * 80)
    print(f"Dataset Root   : {dataset_root}")
    print(f"Splits Dir     : {splits_dir}")
    print(f"Video Dir      : {video_dir}")
    print(f"Output LMDB Dir: {output_root}")
    print(f"Labels Dir     : {labels_dir}")
    print(f"Workers        : {args.workers}")
    print(f"JPEG Quality   : {args.jpeg_quality}")
    print(f"Max Frames     : {args.max_frames}")
    print(f"Sampling       : {args.sampling}")
    print(f"Overwrite      : {args.overwrite}")
    print("-" * 80)

    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be a positive integer")
    if args.sampling == "uniform" and args.max_frames is None:
        parser.error("--max-frames is required when --sampling uniform is used")

    # Determine per-split limits
    split_limits: Dict[str, Optional[int]] = {}
    for split in args.splits:
        if split == "train" and args.limit_train is not None:
            split_limits[split] = args.limit_train
        elif split == "dev" and args.limit_dev is not None:
            split_limits[split] = args.limit_dev
        elif split == "test" and args.limit_test is not None:
            split_limits[split] = args.limit_test
        else:
            split_limits[split] = args.limit

    tasks: List[Dict[str, Any]] = []
    for split in args.splits:
        csv_path = splits_dir / f"{split}.csv"
        rows = read_split_csv(csv_path, limit=split_limits[split])
        print(f"Split [{split}]: Selected {len(rows)} samples (limit={split_limits[split]})")
        for row in rows:
            uid = row["uid"]
            text = row["text"]
            video_path = video_dir / f"{uid}.mp4"
            sample_dir = output_root / split / uid
            tasks.append({
                "split": split,
                "uid": uid,
                "text": text,
                "video_path": str(video_path),
                "sample_dir": str(sample_dir),
                "jpeg_quality": args.jpeg_quality,
                "max_frames": args.max_frames,
                "sampling": args.sampling,
                "overwrite": args.overwrite,
            })

    if args.validate_only:
        print("\nSkipping preprocessing (--validate-only set). Running validation...")
        # Collect existing records
        val_records: Dict[str, Dict[str, Dict[str, Any]]] = {s: {} for s in args.splits}
        for task in tasks:
            s_dir = Path(task["sample_dir"])
            val, n_f, b_len = is_sample_valid(s_dir)
            if val:
                val_records[task["split"]][task["uid"]] = {
                    "text": task["text"],
                    "num_frames": n_f,
                    "disk_bytes": b_len,
                }
        val_results = validate_prepared_dataset(output_root, labels_dir, args.splits, val_records)
        return 0 if val_results["all_passed"] else 1

    # Execute processing
    print(f"\nProcessing {len(tasks)} sample tasks...")
    wall_start = time.time()
    processed_records: Dict[str, Dict[str, Dict[str, Any]]] = {s: {} for s in args.splits}
    stats = {"processed": 0, "skipped": 0, "failed": 0, "total_frames": 0, "total_bytes": 0}

    if args.workers <= 1:
        # Sequential execution
        for task in tasks:
            res = process_single_sample(task)
            split, uid = res["split"], res["uid"]
            if res["status"] in ("PROCESSED", "SKIPPED"):
                if res["status"] == "PROCESSED":
                    stats["processed"] += 1
                else:
                    stats["skipped"] += 1
                stats["total_frames"] += res["num_frames"]
                stats["total_bytes"] += res["disk_bytes"]
                processed_records[split][uid] = {
                    "text": res["text"],
                    "num_frames": res["num_frames"],
                    "original_frame_count": res.get("original_frame_count", res["num_frames"]),
                    "sampling": task["sampling"],
                    "max_frames": task["max_frames"],
                    "disk_bytes": res["disk_bytes"],
                }
                print(f"  [{res['status']}] {split}/{uid}: {res['num_frames']} frames, {res['disk_bytes'] / (1024**2):.2f} MB ({res['duration_s']:.2f}s)")
            else:
                stats["failed"] += 1
                print(f"  [FAILED] {split}/{uid}: {res['error']}")
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(process_single_sample, t): t for t in tasks}
            for future in as_completed(future_to_task):
                res = future.result()
                task = future_to_task[future]
                split, uid = res["split"], res["uid"]
                if res["status"] in ("PROCESSED", "SKIPPED"):
                    if res["status"] == "PROCESSED":
                        stats["processed"] += 1
                    else:
                        stats["skipped"] += 1
                    stats["total_frames"] += res["num_frames"]
                    stats["total_bytes"] += res["disk_bytes"]
                    processed_records[split][uid] = {
                        "text": res["text"],
                        "num_frames": res["num_frames"],
                        "original_frame_count": res.get("original_frame_count", res["num_frames"]),
                        "sampling": task["sampling"],
                        "max_frames": task["max_frames"],
                        "disk_bytes": res["disk_bytes"],
                    }
                    print(f"  [{res['status']}] {split}/{uid}: {res['num_frames']} frames, {res['disk_bytes'] / (1024**2):.2f} MB ({res['duration_s']:.2f}s)")
                else:
                    stats["failed"] += 1
                    print(f"  [FAILED] {split}/{uid}: {res['error']}")

    wall_time_s = time.time() - wall_start
    stats["wall_time_s"] = wall_time_s

    # Update gzipped pickle label files
    print("\nUpdating label files...")
    for split in args.splits:
        split_records = processed_records[split]
        labels_file = labels_dir / f"labels.{split}"
        updated_dict = update_label_file(labels_file, split, split_records)
        print(f"  Updated {labels_file} (contains {len(updated_dict)} samples)")

    # Run validation suite
    val_report = validate_prepared_dataset(output_root, labels_dir, args.splits, processed_records)

    # Print capacity and throughput report
    print_resource_and_throughput_report(stats, dataset_root)

    if not val_report["all_passed"] or stats["failed"] > 0:
        print("\nPhase 2B Finished with ERRORS.")
        return 1

    print("\nPhase 2B Controlled Subset Completed and Validated SUCCESSFULLY!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
