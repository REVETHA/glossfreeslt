"""Phase 2A Compatibility Probe for iSign v1.1 LMDB pipeline.

Tests compatibility of iSign MP4 videos with:
1. ImageDatabase (dataloader/database.py)
2. load_imgs_lmdb() in S2T_Dataset (dataloader/datasets.py)

This script:
- Reads 1 to 3 videos from iSign splits/v1/dev.csv using OpenCV.
- Writes a tiny temporary LMDB test dataset to dataset/iSign/probe_lmdb/.
- Reopens using ImageDatabase and validates frames, order, and shapes.
- Tests sample loading through S2T_Dataset and collate_fn.
- Never modifies existing dataset files or code.
"""

from __future__ import annotations

import csv
import gzip
import io
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import lmdb
import numpy as np
import torch
from PIL import Image

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.database import ImageDatabase
from dataloader.datasets import S2T_Dataset


class DummyTokenizer:
    """Lightweight mock tokenizer to avoid downloading external models."""
    def __init__(self):
        self.pad_token_id = 1

    def __call__(
        self,
        text_target: List[str],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        max_words = max(len(t.split()) for t in text_target) if text_target else 1
        seq_len = max(max_words + 2, 4)
        input_ids = torch.ones((len(text_target), seq_len), dtype=torch.long)
        attention_mask = torch.ones((len(text_target), seq_len), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class ProbeArgs:
    """Minimal args container required by S2T_Dataset and load_imgs_lmdb."""
    input_size: int = 224
    resize: int = 256
    noise_rate: float = 0.15
    noise_type: str = "omit_last"
    random_shuffle: bool = False


def read_mp4_frames_cv2(video_path: Path) -> List[np.ndarray]:
    """Safely read all RGB frames from an MP4 file using OpenCV."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    frames: List[np.ndarray] = []
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    finally:
        cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames could be extracted from: {video_path}")

    return frames


def encode_frame(frame_rgb: np.ndarray, format_type: str = "JPEG", quality: int = 95) -> bytes:
    """Encode RGB frame numpy array into image bytes."""
    pil_img = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    if format_type.upper() == "JPEG":
        pil_img.save(buf, format="JPEG", quality=quality)
    elif format_type.upper() == "PNG":
        pil_img.save(buf, format="PNG")
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    return buf.getvalue()


def write_video_lmdb(
    lmdb_folder: Path,
    frame_bytes_list: List[bytes],
    protocol: int = pickle.DEFAULT_PROTOCOL,
    key_format: str = "pickled_keys_list",
) -> None:
    """Write an LMDB database for a single video matching ImageDatabase expectations.

    Directory structure:
      lmdb_folder/data.mdb
      lmdb_folder/lock.mdb

    Keys:
      b"protocol" -> pickle.dumps(protocol)
      pickle.dumps(i, protocol=protocol) -> image bytes (for i in 0..N-1)
      pickle.dumps("keys", protocol=protocol) -> pickle.dumps(list_of_keys, protocol=protocol)
    """
    lmdb_folder.mkdir(parents=True, exist_ok=True)
    map_size = max(1024 * 1024 * 128, len(frame_bytes_list) * 1024 * 1024 * 2)

    env = lmdb.open(str(lmdb_folder), map_size=map_size, subdir=True, lock=True)
    with env.begin(write=True) as txn:
        # 1. b"protocol"
        proto_key = "protocol".encode("ascii")
        proto_val = pickle.dumps(protocol, protocol=protocol)
        txn.put(key=proto_key, value=proto_val, dupdata=False)

        # 2. Frame keys & values
        list_of_keys = []
        for i, f_bytes in enumerate(frame_bytes_list):
            key = pickle.dumps(i, protocol=protocol)
            txn.put(key=key, value=f_bytes, dupdata=False)
            if key_format == "pickled_keys_list":
                list_of_keys.append(key)
            else:
                list_of_keys.append(i)

        # 3. pickle.dumps("keys")
        keys_key = pickle.dumps("keys", protocol=protocol)
        keys_val = pickle.dumps(list_of_keys, protocol=protocol)
        txn.put(key=keys_key, value=keys_val, dupdata=False)

    env.close()


def run_probe(max_videos: int = 3) -> Dict[str, Any]:
    """Executes the Phase 2A probe on 1-3 videos."""
    workspace_root = REPO_ROOT.parent
    dataset_root = workspace_root / "dataset" / "iSign"
    splits_csv = dataset_root / "splits" / "v1" / "dev.csv"
    video_dir = dataset_root / "videos" / "iSign-videos_v1.1"
    probe_dir = dataset_root / "probe_lmdb"

    results: Dict[str, Any] = {
        "status": "INIT",
        "videos_tested": 0,
        "image_database_checks": [],
        "s2t_dataset_checks": [],
        "collate_fn_check": False,
        "errors": [],
    }

    print("=" * 80)
    print("PHASE 2A: iSign v1.1 LMDB Compatibility Probe")
    print("=" * 80)
    print(f"Repository Root : {REPO_ROOT}")
    print(f"Dataset Root    : {dataset_root}")
    print(f"Dev Split CSV   : {splits_csv}")
    print(f"Video Directory : {video_dir}")
    print(f"Probe LMDB Dir  : {probe_dir}")
    print("-" * 80)

    # 1. Read selected videos from dev.csv
    if not splits_csv.is_file():
        results["errors"].append(f"Dev split CSV not found: {splits_csv}")
        results["status"] = "FAIL"
        return results

    rows: List[Dict[str, str]] = []
    with open(splits_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            if len(rows) >= max_videos:
                break

    print(f"Selected {len(rows)} video(s) for probe testing:")
    for idx, r in enumerate(rows, start=1):
        print(f"  [{idx}] uid: {r['uid']} | text: {r['text'][:50]} | group: {r.get('source_video_id', 'N/A')}")
    print("-" * 80)

    # 2. Extract frames, encode, and write LMDB
    probed_samples: List[Dict[str, Any]] = []
    phase = "dev"
    probe_phase_dir = probe_dir / phase

    for r in rows:
        uid = r["uid"]
        text = r["text"]
        mp4_path = video_dir / f"{uid}.mp4"

        print(f"\nProcessing video: {uid}")
        print(f"  Reading MP4: {mp4_path}")
        raw_frames = read_mp4_frames_cv2(mp4_path)
        num_frames = len(raw_frames)
        h, w, c = raw_frames[0].shape
        print(f"  Extracted {num_frames} frames, resolution: {w}x{h}, channels: {c}")

        # Encode frames as JPEG
        frame_bytes = [encode_frame(f, format_type="JPEG", quality=95) for f in raw_frames]

        # LMDB path for this sample: <probe_dir>/<phase>/<uid>
        sample_lmdb_path = probe_phase_dir / uid
        print(f"  Writing LMDB to: {sample_lmdb_path}")
        write_video_lmdb(sample_lmdb_path, frame_bytes)

        probed_samples.append({
            "uid": uid,
            "text": text,
            "mp4_path": mp4_path,
            "lmdb_path": sample_lmdb_path,
            "num_frames": num_frames,
            "raw_frames": raw_frames,
            "resolution": (w, h),
        })

    results["videos_tested"] = len(probed_samples)

    # 3. Verify with ImageDatabase
    print("\n" + "=" * 80)
    print("STEP 1: Verify with ImageDatabase (dataloader/database.py)")
    print("=" * 80)

    all_db_passed = True
    for sample in probed_samples:
        uid = sample["uid"]
        lmdb_path = sample["lmdb_path"]
        expected_n = sample["num_frames"]
        raw_frames = sample["raw_frames"]

        print(f"\nInspecting ImageDatabase for uid: {uid}")
        db = ImageDatabase(path=str(lmdb_path))

        # Check protocol
        proto = db.protocol
        print(f"  Protocol: {proto} (expected: int >= 2)")

        # Check keys count and length
        keys_len = len(db.keys)
        db_len = len(db)
        print(f"  len(db.keys) = {keys_len}, len(db) = {db_len}, expected = {expected_n}")
        count_ok = (keys_len == expected_n) and (db_len == expected_n)

        # Check single frame fetch
        frame_0 = db[0]
        single_ok = isinstance(frame_0, Image.Image) and (frame_0.size == sample["resolution"])
        print(f"  Single frame fetch db[0]: type={type(frame_0)}, size={frame_0.size}, mode={frame_0.mode}")

        # Check batch frame fetch db[ind] as load_imgs_lmdb does
        ind = list(range(expected_n))
        images = db[ind]
        batch_ok = (len(images) == expected_n) and all(isinstance(img, Image.Image) for img in images)
        print(f"  Multi-frame fetch db[{expected_n} indices]: returned {len(images)} PIL images")

        # Frame ordering check: compare first, middle, last decoded frame with OpenCV
        sample_indices = [0, expected_n // 2, expected_n - 1]
        order_ok = True
        for s_idx in sample_indices:
            decoded_arr = np.array(images[s_idx])
            raw_arr = raw_frames[s_idx]
            # Tolerant difference due to JPEG compression
            mean_diff = np.mean(np.abs(decoded_arr.astype(float) - raw_arr.astype(float)))
            max_diff = np.max(np.abs(decoded_arr.astype(float) - raw_arr.astype(float)))
            if mean_diff > 10.0:  # JPEG compression typically has mean diff < 3.0
                order_ok = False
                print(f"    WARNING: Large difference at frame {s_idx}: mean_diff={mean_diff:.2f}, max_diff={max_diff}")
            else:
                print(f"    Frame {s_idx} verification: mean_diff={mean_diff:.2f}, max_diff={max_diff} -> OK")

        sample_passed = count_ok and single_ok and batch_ok and order_ok
        print(f"  Sample {uid} ImageDatabase Result: {'PASS' if sample_passed else 'FAIL'}")

        results["image_database_checks"].append({
            "uid": uid,
            "count_ok": count_ok,
            "single_ok": single_ok,
            "batch_ok": batch_ok,
            "order_ok": order_ok,
            "passed": sample_passed,
        })
        if not sample_passed:
            all_db_passed = False

    # 4. Verify with S2T_Dataset pipeline
    print("\n" + "=" * 80)
    print("STEP 2: Verify with S2T_Dataset (dataloader/datasets.py)")
    print("=" * 80)

    # Create temporary gzipped label file for S2T_Dataset
    probe_labels_file = probe_dir / "probe_labels.dev"
    probe_labels_data = {}
    for sample in probed_samples:
        uid = sample["uid"]
        # Note: S2T_Dataset.load_imgs_lmdb splits sample['name'] on '/':
        # phase, file_name = file_name.split('/')
        # So sample['name'] MUST be f"{phase}/{uid}"
        probe_labels_data[uid] = {
            "name": f"{phase}/{uid}",
            "text": sample["text"],
        }

    with gzip.open(probe_labels_file, "wb") as f:
        pickle.dump(probe_labels_data, f)
    print(f"Created temporary probe label file: {probe_labels_file}")

    # Configure dataset
    config = {
        "data": {
            "img_lmdb_path": str(probe_dir),
            "max_length": 300,
            "dataset_name": "phoenix",  # triggers standard path resolution
        }
    }
    dummy_tokenizer = DummyTokenizer()
    args = ProbeArgs()

    dataset = S2T_Dataset(
        path=str(probe_labels_file),
        tokenizer=dummy_tokenizer,
        config=config,
        args=args,
        phase=phase,
        training_refurbish=False,
    )
    print(f"Dataset instantiated: {dataset}")
    print(f"Dataset length: {len(dataset)}")

    all_s2t_passed = True
    batch_items = []
    for idx in range(len(dataset)):
        name_sample, img_sample, tgt_sample, img_len = dataset[idx]
        expected_sample = probed_samples[idx]
        uid = expected_sample["uid"]

        print(f"\nChecking dataset[{idx}] (uid: {uid}):")
        print(f"  name_sample: {name_sample}")
        print(f"  img_sample shape: {img_sample.shape}, dtype: {img_sample.dtype}")
        print(f"  tgt_sample : {tgt_sample}")
        print(f"  img_len    : {img_len}")

        # Assertions
        name_ok = name_sample == f"{phase}/{uid}"
        shape_ok = (
            isinstance(img_sample, torch.Tensor)
            and img_sample.shape == (img_len, 3, args.input_size, args.input_size)
            and img_sample.dtype == torch.float32
        )
        text_ok = tgt_sample == expected_sample["text"]
        len_ok = img_len == min(expected_sample["num_frames"], config["data"]["max_length"])

        item_passed = name_ok and shape_ok and text_ok and len_ok
        print(f"  Result: {'PASS' if item_passed else 'FAIL'}")

        results["s2t_dataset_checks"].append({
            "uid": uid,
            "name_ok": name_ok,
            "shape_ok": shape_ok,
            "text_ok": text_ok,
            "len_ok": len_ok,
            "passed": item_passed,
        })
        if not item_passed:
            all_s2t_passed = False

        batch_items.append((name_sample, img_sample, tgt_sample, img_len))

    # Test collate_fn
    print("\nTesting collate_fn on batch:")
    try:
        src_input, tgt_input = dataset.collate_fn(batch_items)
        print(f"  src_input keys: {list(src_input.keys())}")
        print(f"  src_input['input_ids'] shape: {src_input['input_ids'].shape}")
        print(f"  src_input['attention_mask'] shape: {src_input['attention_mask'].shape}")
        print(f"  src_input['src_length_batch']: {src_input['src_length_batch']}")
        print(f"  src_input['new_src_length_batch']: {src_input['new_src_length_batch']}")
        print(f"  tgt_input['input_ids'] shape: {tgt_input['input_ids'].shape}")
        collate_ok = True
        print("  collate_fn Result: PASS")
    except Exception as e:
        print(f"  collate_fn FAILED with error: {e}")
        collate_ok = False
        results["errors"].append(f"collate_fn error: {e}")

    results["collate_fn_check"] = collate_ok

    # Final status
    if all_db_passed and all_s2t_passed and collate_ok:
        results["status"] = "PASS"
    else:
        results["status"] = "FAIL"

    print("\n" + "=" * 80)
    print(f"FINAL PROBE SUMMARY: {results['status']}")
    print("=" * 80)
    print(f"ImageDatabase Checks Passed : {all_db_passed}")
    print(f"S2T_Dataset Checks Passed   : {all_s2t_passed}")
    print(f"collate_fn Checks Passed    : {collate_ok}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    res = run_probe(max_videos=3)
    if res["status"] != "PASS":
        sys.exit(1)
