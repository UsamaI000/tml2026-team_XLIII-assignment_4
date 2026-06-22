#!/usr/bin/env python3
"""
Frequency Band-Pass Fingerprint Watermark Forgery
=================================================

This variant estimates each WM family's common signal using a Fourier-domain
band-pass filter on the watermarked source images, then transfers the aggregated
spatial fingerprint to the mapped clean targets.

Use this when the watermark detector relies on periodic or mid/high-frequency
patterns. It is complementary to approach_residual_fingerprint_forgery.py.

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


def radial_bandpass_mask(height: int, width: int, low: float, high: float) -> np.ndarray:
    if not (0.0 <= low < high <= 0.5):
        raise ValueError("Expected 0 <= low < high <= 0.5 for normalized frequency cutoffs.")
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    return ((radius >= low) & (radius <= high)).astype(np.float32)[:, :, None]


def bandpass_residual(image: np.ndarray, low: float, high: float) -> np.ndarray:
    h, w, _ = image.shape
    centered = image - np.mean(image, axis=(0, 1), keepdims=True)
    mask = radial_bandpass_mask(h, w, low, high)
    spectrum = np.fft.fft2(centered, axes=(0, 1))
    filtered = np.fft.ifft2(spectrum * mask, axes=(0, 1)).real.astype(np.float32)
    return filtered


def robust_scale(array: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    med = np.median(array, axis=(0, 1), keepdims=True)
    centered = array - med
    mad_scale = 1.4826 * np.median(np.abs(centered), axis=(0, 1), keepdims=True)
    p95_scale = np.percentile(np.abs(centered), 95, axis=(0, 1), keepdims=True)
    std_scale = np.std(centered, axis=(0, 1), keepdims=True)
    return np.maximum.reduce([mad_scale, p95_scale / 2.0, std_scale, np.full_like(std_scale, eps)])


def normalize_fingerprint(fingerprint: np.ndarray, strength: float) -> np.ndarray:
    centered = fingerprint - np.mean(fingerprint, axis=(0, 1), keepdims=True)
    normalized = centered / robust_scale(centered) * float(strength)
    max_abs = max(2.0, 3.0 * float(strength))
    return np.clip(normalized, -max_abs, max_abs).astype(np.float32)


def estimate_fingerprint(source_paths: Sequence[Path], low: float, high: float, strength: float) -> np.ndarray:
    residuals = []
    expected_shape = None
    for path in source_paths:
        img = read_rgb(path)
        if expected_shape is None:
            expected_shape = img.shape
        elif img.shape != expected_shape:
            raise RuntimeError(f"Source size mismatch in {path}")
        res = bandpass_residual(img, low=low, high=high)
        res = res - np.mean(res, axis=(0, 1), keepdims=True)
        res = res / robust_scale(res)
        residuals.append(res)
    fingerprint = np.median(np.stack(residuals, axis=0), axis=0)
    return normalize_fingerprint(fingerprint, strength=strength)


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
            p = out_dir / f"{number}.png"
            zipf.write(p, arcname=p.name)
    with zipfile.ZipFile(zip_out, "r") as zipf:
        if set(zipf.namelist()) != EXPECTED_IMAGE_NAMES:
            raise RuntimeError("ZIP is not a flat 200-image submission.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create frequency-bandpass forgery submission ZIP.")
    parser.add_argument("--zip_file", type=Path, default=Path("Dataset.zip"))
    parser.add_argument("--dataset_dir", type=Path, default=Path("_dataset_extracted_frequency_bandpass"))
    parser.add_argument("--out_dir", type=Path, default=Path("submission_temp_frequency_bandpass"))
    parser.add_argument("--zip_out", type=Path, default=Path("submission_frequency_bandpass.zip"))
    parser.add_argument("--low", type=float, default=0.06, help="Low normalized radial frequency cutoff.")
    parser.add_argument("--high", type=float, default=0.40, help="High normalized radial frequency cutoff.")
    parser.add_argument("--strength", type=float, default=4.0, help="Approximate perturbation amplitude in pixel levels.")
    args = parser.parse_args(argv)

    try:
        root = ensure_dataset(args.zip_file, args.dataset_dir)
        clean_dir = root / "clean_targets"
        source_root = root / "watermarked_sources"
        if args.out_dir.exists():
            shutil.rmtree(args.out_dir)
        args.out_dir.mkdir(parents=True, exist_ok=True)

        total = 0
        for wm_name, start, stop in CATEGORIES:
            source_paths = sorted((source_root / wm_name).glob("*.png"), key=numeric_key)
            if len(source_paths) != stop - start + 1:
                raise RuntimeError(f"Wrong source count for {wm_name}: {len(source_paths)}")
            fingerprint = estimate_fingerprint(source_paths, args.low, args.high, args.strength)
            print(f"{wm_name}: fingerprint shape={fingerprint.shape}, range=[{fingerprint.min():.2f}, {fingerprint.max():.2f}]")
            for number in range(start, stop + 1):
                target = read_rgb(clean_dir / f"{number}.png")
                if target.shape != fingerprint.shape:
                    raise RuntimeError(f"Shape mismatch for target {number}.png")
                save_rgb(target + fingerprint, args.out_dir / f"{number}.png")
                total += 1
        if total != 200:
            raise RuntimeError(f"Expected 200 outputs, produced {total}")
        validate_and_zip(args.out_dir, args.zip_out)
        print(f"Done. Created {args.zip_out}.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
