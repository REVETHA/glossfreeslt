"""Validate iSign MP4 to LMDB preprocessing on exactly five samples.

The generated LMDBs use the ImageDatabase format consumed by
S2T_Dataset.load_imgs_lmdb: pickle-encoded integer keys, a pickled key list,
and JPEG image bytes stored in per-sample LMDB directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import pickle
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import lmdb
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.database import ImageDatabase
from dataloader.datasets import S2T_Dataset


FRAME_SIZE = (256, 256)
SAMPLE_COUNTS = {"train": 3, "dev": 1, "test": 1}


def read_sample_rows(split_dir: Path) -> list[dict[str, str]]:
    selected = []
    for split, count in SAMPLE_COUNTS.items():
        path = split_dir / f"{split}.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) < count:
            raise RuntimeError(f"{path} contains fewer than {count} rows")
        for row in rows[:count]:
            row = dict(row)
            row["split"] = split
            selected.append(row)
    return selected


def frame_digest(image: Image.Image) -> str:
    image = image.convert("RGB")
    return hashlib.sha256(np.asarray(image, dtype=np.uint8).tobytes()).hexdigest()


def jpeg_payload(image: Image.Image, quality: int = 95) -> bytes:
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=quality)
    return encoded.getvalue()


def decoded_jpeg_digest(payload: bytes) -> str:
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    return frame_digest(image)


def decoded_image_matches(reference: Image.Image, decoded: Image.Image) -> bool:
    ref_array = np.asarray(reference.convert("RGB"), dtype=np.uint8)
    dec_array = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
    return ref_array.shape == dec_array.shape and np.array_equal(ref_array, dec_array)


def decode_video(video_path: Path) -> tuple[list[Image.Image], dict[str, object]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb).resize(FRAME_SIZE, Image.Resampling.BILINEAR))
    capture.release()
    if not frames:
        raise RuntimeError(f"OpenCV decoded zero frames from {video_path}")
    return frames, {"fps": fps, "width": width, "height": height}


def write_image_database(path: Path, frames: list[Image.Image], video_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lmdb.open(str(path), map_size=2**32) as environment:
        with environment.begin(write=True) as transaction:
            transaction.put(b"protocol", pickle.dumps(pickle.DEFAULT_PROTOCOL))
            keys = []
            for index, image in enumerate(frames):
                key = pickle.dumps(index, protocol=pickle.DEFAULT_PROTOCOL)
                transaction.put(key, jpeg_payload(image, quality=95))
                keys.append(index)
            transaction.put(
                pickle.dumps("keys", protocol=pickle.DEFAULT_PROTOCOL),
                pickle.dumps(keys, protocol=pickle.DEFAULT_PROTOCOL),
            )
            transaction.put(b"test_video_name", video_name.encode("utf-8"))


def read_database(path: Path) -> tuple[list[Image.Image], list[bytes]]:
    database = ImageDatabase(str(path))
    keys = list(range(len(database.keys)))
    images = database[keys]
    raw_payloads: list[bytes] = []
    with lmdb.open(str(path), readonly=True, lock=False) as environment:
        with environment.begin() as transaction:
            for key in keys:
                raw_key = pickle.dumps(key, protocol=database.protocol)
                raw_payloads.append(transaction.get(raw_key))
    del database
    return images, raw_payloads


def exercise_active_loader(lmdb_root: Path, split: str, uid: str):
    dataset = object.__new__(S2T_Dataset)
    dataset.config = {"data": {"max_length": 10_000}}
    dataset.img_lmdb_path = str(lmdb_root)
    dataset.phase = "dev" if split != "train" else "train"
    dataset.max_length = 10_000
    dataset.args = SimpleNamespace(input_size=224, resize=256)
    dataset.seq = lambda images: images
    tensor = dataset.load_imgs_lmdb(f"{split}/{uid}")
    return tensor, dataset.img_len


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    dataset_root = args.dataset_root or repository_root / "dataset" / "iSign"
    split_dir = dataset_root / "splits" / "v1"
    video_dir = dataset_root / "videos" / "iSign-videos_v1.1"
    output_root = args.output_root or dataset_root / "test_preprocessing_lmdb"

    selected = read_sample_rows(split_dir)
    if len(selected) != 5:
        raise AssertionError(f"Expected exactly five selected samples, got {len(selected)}")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    report_rows = []
    for row in selected:
        uid = row["uid"]
        video_path = video_dir / f"{uid}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing MP4 for {row['split']} UID {uid}: {video_path}")

        frames, metadata = decode_video(video_path)
        source_payloads = [jpeg_payload(image, quality=95) for image in frames]
        source_digests = [frame_digest(image) for image in frames]
        lmdb_path = output_root / row["split"] / uid
        write_image_database(lmdb_path, frames, uid)
        images, raw_payloads = read_database(lmdb_path)
        if len(images) != len(frames):
            raise AssertionError(f"Frame count mismatch for {uid}")
        if source_payloads != raw_payloads:
            raise AssertionError(f"LMDB JPEG payload mismatch for {uid}")
        stored_digests = [decoded_jpeg_digest(payload) for payload in raw_payloads]
        if source_digests != stored_digests:
            mismatch_index = next(
                idx for idx, (src, out) in enumerate(zip(source_digests, stored_digests)) if src != out
            )
            if not all(
                decoded_image_matches(reference, decoded)
                for reference, decoded in zip(frames, images)
            ):
                raise AssertionError(
                    f"LMDB frame content mismatch for {uid} at index {mismatch_index}: "
                    f"source={source_digests[mismatch_index][:12]}, stored={stored_digests[mismatch_index][:12]}"
                )
        if not all(decoded_image_matches(reference, decoded) for reference, decoded in zip(frames, images)):
            raise AssertionError(f"Decoded LMDB frame pixels differ from source frames in {uid}")

        tensor, loader_frame_count = exercise_active_loader(output_root, row["split"], uid)
        if loader_frame_count != len(frames):
            raise AssertionError(f"Active loader frame count mismatch for {uid}")

        report_rows.append(
            {
                "uid": uid,
                "split": row["split"],
                "original_frame_count": len(frames),
                "stored_frame_count": len(images),
                "fps": metadata["fps"],
                "resolution": f"{metadata['width']}x{metadata['height']}",
                "stored_resolution": "256x256",
                "active_loader_tensor_shape": list(tensor.shape),
                "active_loader": "S2T_Dataset.load_imgs_lmdb",
            }
        )

    report_path = output_root / "report.json"
    report_path.write_text(
        __import__("json").dumps(
            {
                "sample_size": len(report_rows),
                "sample_selection": report_rows,
                "lmdb_structure": "<output_root>/<split>/<uid>/ with protocol, pickle('keys'), and pickle(index) JPEG values",
                "active_loader_compatible": True,
                "compressed_script_compatible_with_active_loader": False,
                "compressed_script_note": "create_lmdb_compressed.py writes ASCII frame keys and details, which matches load_imgs_lmdb_preprocessed only; S2T_Dataset.__getitem__ currently calls load_imgs_lmdb.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Created exactly {len(report_rows)} LMDB samples under {output_root}")
    for row in report_rows:
        print(
            f"{row['split']} UID={row['uid']} frames={row['original_frame_count']} "
            f"stored={row['stored_frame_count']} fps={row['fps']:.3f} "
            f"resolution={row['resolution']} tensor_shape={row['active_loader_tensor_shape']}"
        )
    print(f"Read-back validation: PASS; report={report_path}")


if __name__ == "__main__":
    main()
