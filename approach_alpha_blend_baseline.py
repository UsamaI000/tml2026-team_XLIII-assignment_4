#!/usr/bin/env python3
"""
Alpha Blend Baseline Watermark Forgery
======================================

A direct, template-style baseline. For each target image, blend it with the
corresponding watermarked source image from the assigned WM group.

This is intentionally simple and useful mainly as a sanity check. It tends to
copy visible source content, so it should usually be weaker than residual
fingerprint transfer in visual-quality score.

This script does not modify task_template.py or submission.py.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from PIL import Image


CATEGORIES: Tuple[Tuple[str, int, int], ...] = (
    ("WM_1", 1, 25),
    ("WM_2", 26, 50),
    ("WM_3", 51, 75),
    ("WM_4", 76, 100),
    ("WM_5", 101, 125),
    ("WM_6", 126, 150),
    ("WM_7", 151, 175),
    ("WM_8", 176, 200),
)
EXPECTED_IMAGE_NAMES = {f"{i}.png" for i in range(1, 201)}


def numeric_key(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def find_dataset_root(base: Path) -> Path | None:
    for root in (base, base / "Dataset"):
        if (root / "clean_targets").is_dir() and (root / "watermarked_sources").is_dir():
            return root
    return None


def ensure_dataset(zip_file: Path, dataset_dir: Path) -> Path:
    found = find_dataset_root(dataset_dir.resolve()) or find_dataset_root(Path.cwd())
    if found is not None:
        return found
    if not zip_file.is_file():
        raise FileNotFoundError(f"Missing dataset ZIP: {zip_file}")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)
    found = find_dataset_root(dataset_dir.resolve())
    if found is None:
        raise FileNotFoundError("Could not locate clean_targets/ and watermarked_sources/.")
    return found


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32)


def save_rgb(array: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode="RGB").save(path)


def validate_and_zip(out_dir: Path, zip_out: Path) -> None:
    names = {p.name for p in out_dir.glob("*.png")}
    if names != EXPECTED_IMAGE_NAMES:
        missing = sorted(EXPECTED_IMAGE_NAMES - names, key=lambda x: int(x[:-4]))
        extra = sorted(names - EXPECTED_IMAGE_NAMES)
        raise RuntimeError(f"Invalid output set. Missing={missing[:10]}, extra={extra[:10]}")
    if zip_out.exists():
        zip_out.unlink()
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zipf:
        for number in range(1, 201):
            path = out_dir / f"{number}.png"
            zipf.write(path, arcname=path.name)
    with zipfile.ZipFile(zip_out, "r") as zipf:
        if set(zipf.namelist()) != EXPECTED_IMAGE_NAMES:
            raise RuntimeError("ZIP is not a flat 200-image submission.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create alpha-blend baseline submission ZIP.")
    parser.add_argument("--zip_file", type=Path, default=Path("Dataset.zip"))
    parser.add_argument("--dataset_dir", type=Path, default=Path("_dataset_extracted_alpha_blend"))
    parser.add_argument("--out_dir", type=Path, default=Path("submission_temp_alpha_blend"))
    parser.add_argument("--zip_out", type=Path, default=Path("submission_alpha_blend.zip"))
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Source-image blend weight. Try 0.05, 0.10, 0.15, 0.25; 0.50 is the naive template baseline.",
    )
    args = parser.parse_args(argv)

    if not (0.0 <= args.alpha <= 1.0):
        print("ERROR: --alpha must be in [0, 1].", file=sys.stderr)
        return 1

    try:
        root = ensure_dataset(args.zip_file, args.dataset_dir)
        clean_dir = root / "clean_targets"
        source_root = root / "watermarked_sources"
        if args.out_dir.exists():
            shutil.rmtree(args.out_dir)
        args.out_dir.mkdir(parents=True, exist_ok=True)

        total = 0
        for wm_name, start, stop in CATEGORIES:
            source_dir = source_root / wm_name
            source_paths = sorted(source_dir.glob("*.png"), key=numeric_key)
            expected_count = stop - start + 1
            if len(source_paths) != expected_count:
                raise RuntimeError(f"Expected {expected_count} sources in {source_dir}, found {len(source_paths)}")
            for number, source_path in zip(range(start, stop + 1), source_paths):
                target_path = clean_dir / f"{number}.png"
                target = read_rgb(target_path)
                source = read_rgb(source_path)
                if target.shape != source.shape:
                    raise RuntimeError(f"Shape mismatch: {target_path} vs {source_path}")
                forged = (1.0 - args.alpha) * target + args.alpha * source
                save_rgb(forged, args.out_dir / f"{number}.png")
                total += 1
        if total != 200:
            raise RuntimeError(f"Expected 200 outputs, produced {total}")
        validate_and_zip(args.out_dir, args.zip_out)
        print(f"Done. Created {args.zip_out} with alpha={args.alpha}.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
