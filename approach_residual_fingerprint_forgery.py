#!/usr/bin/env python3
"""
Residual Fingerprint Watermark Forgery
======================================

This script creates a flat submission ZIP containing 200 forged target images.
It does not modify task_template.py or submission.py.

Approach:
    For each watermark family WM_1 ... WM_8, estimate a reusable watermark
    fingerprint from its 25 watermarked source images. Since the 25 sources share
    the same embedded message but have different image content, averaging robust
    high-pass residuals suppresses image content and keeps signal that is common
    to the watermark family. The estimated fingerprint is then added to the
    mapped clean target images.

Why this is safer than naive 50/50 blending:
    Blending copies large parts of the source image content into the target,
    which severely hurts visual quality. This script transfers only a small
    high-frequency residual pattern, preserving the target image appearance.

Expected dataset layout after extraction:
    clean_targets/1.png ... clean_targets/200.png
    watermarked_sources/WM_1/src_1.png ... watermarked_sources/WM_8/src_200.png

The code is robust to ZIP files that either contain the folders directly at the
archive root or under a top-level Dataset/ folder.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter


# Same target mapping as the provided task template.
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


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    clean_targets: Path
    watermarked_sources: Path


def parse_float_list(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def numeric_key(path: Path) -> int:
    """Sort 1.png, src_1.png, src_25.png numerically, not lexicographically."""
    stem = path.stem
    number_text = stem.split("_")[-1]
    try:
        return int(number_text)
    except ValueError as exc:
        raise ValueError(f"Cannot extract numeric id from filename: {path.name}") from exc


def find_dataset_root(base: Path) -> DatasetPaths | None:
    """Return dataset paths if base contains the expected folders."""
    candidates = [base, base / "Dataset"]
    for root in candidates:
        clean = root / "clean_targets"
        sources = root / "watermarked_sources"
        if clean.is_dir() and sources.is_dir():
            return DatasetPaths(root=root, clean_targets=clean, watermarked_sources=sources)
    return None


def ensure_dataset(zip_file: Path, dataset_dir: Path) -> DatasetPaths:
    """Locate or extract the dataset and return normalized paths."""
    dataset_dir = dataset_dir.resolve()

    existing = find_dataset_root(dataset_dir)
    if existing is not None:
        return existing

    cwd_existing = find_dataset_root(Path.cwd())
    if cwd_existing is not None:
        return cwd_existing

    if not zip_file.is_file():
        raise FileNotFoundError(
            f"Could not find dataset folders or ZIP file. Missing ZIP: {zip_file}"
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_file} -> {dataset_dir}")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    extracted = find_dataset_root(dataset_dir)
    if extracted is None:
        raise FileNotFoundError(
            "Dataset extraction finished, but clean_targets/ and watermarked_sources/ "
            f"were not found under {dataset_dir} or {dataset_dir / 'Dataset'}."
        )
    return extracted


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32)


def save_rgb(array: np.ndarray, path: Path) -> None:
    clipped = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    Image.fromarray(clipped, mode="RGB").save(path)


def gaussian_blur_rgb(array: np.ndarray, sigma: float) -> np.ndarray:
    img = Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode="RGB")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    return np.asarray(blurred, dtype=np.float32)


def highpass_residual(array: np.ndarray, sigmas: Sequence[float]) -> np.ndarray:
    """Average several high-pass residuals to avoid overfitting one blur scale."""
    residuals = []
    for sigma in sigmas:
        if sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {sigma}")
        residuals.append(array - gaussian_blur_rgb(array, sigma))
    return np.mean(np.stack(residuals, axis=0), axis=0)


def robust_channel_scale(array: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    Robust per-channel scale, returned as shape (1, 1, 3).

    Pure MAD can be zero for sparse or quantized fingerprints. We therefore fall
    back to high-percentile absolute deviation and finally standard deviation.
    This prevents accidental division by a tiny number and avoids visible spikes.
    """
    median = np.median(array, axis=(0, 1), keepdims=True)
    centered = array - median
    mad_scale = 1.4826 * np.median(np.abs(centered), axis=(0, 1), keepdims=True)
    p95_scale = np.percentile(np.abs(centered), 95, axis=(0, 1), keepdims=True)
    std_scale = np.std(centered, axis=(0, 1), keepdims=True)
    scale = np.maximum.reduce([mad_scale, p95_scale / 2.0, std_scale, np.full_like(std_scale, eps)])
    return scale


def normalize_fingerprint(fingerprint: np.ndarray, strength: float) -> np.ndarray:
    """Normalize a raw fingerprint and clip outliers for visual safety."""
    centered = fingerprint - np.mean(fingerprint, axis=(0, 1), keepdims=True)
    scale = robust_channel_scale(centered)
    normalized = centered / scale * float(strength)
    max_abs = max(2.0, 3.0 * float(strength))
    return np.clip(normalized, -max_abs, max_abs).astype(np.float32)


def estimate_fingerprint(
    source_paths: Sequence[Path],
    sigmas: Sequence[float],
    aggregation: str,
    whiten_sources: bool,
    strength: float,
) -> np.ndarray:
    """Estimate one watermark-family fingerprint from its watermarked sources."""
    if not source_paths:
        raise ValueError("No source images were provided for fingerprint estimation.")

    residuals = []
    expected_size = None
    for path in source_paths:
        img = read_rgb(path)
        if expected_size is None:
            expected_size = img.shape[:2]
        elif img.shape[:2] != expected_size:
            raise ValueError(
                f"All source images for one WM family must share a size. "
                f"Expected {expected_size}, got {img.shape[:2]} for {path}."
            )

        res = highpass_residual(img, sigmas)
        # Remove per-source bias. Optional whitening prevents highly textured
        # source images from dominating the aggregate fingerprint.
        res = res - np.median(res, axis=(0, 1), keepdims=True)
        if whiten_sources:
            res = res / robust_channel_scale(res)
        residuals.append(res)

    stack = np.stack(residuals, axis=0)
    if aggregation == "median":
        fingerprint = np.median(stack, axis=0)
    elif aggregation == "mean":
        fingerprint = np.mean(stack, axis=0)
    elif aggregation == "trimmed_mean":
        if stack.shape[0] < 5:
            fingerprint = np.mean(stack, axis=0)
        else:
            sorted_stack = np.sort(stack, axis=0)
            trim = max(1, int(round(0.1 * stack.shape[0])))
            fingerprint = np.mean(sorted_stack[trim:-trim], axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    # Zero-center, normalize and clip so --strength has a consistent meaning
    # while avoiding rare extreme pixels from sparse/quantized fingerprints.
    return normalize_fingerprint(fingerprint, strength=strength)


def texture_mask(target: np.ndarray, floor: float = 0.35) -> np.ndarray:
    """
    Compute an optional [floor, 1] mask that puts more signal on textured areas.
    Disabled by default because some detectors prefer unmasked global signal.
    """
    residual = np.mean(np.abs(highpass_residual(target, sigmas=(1.0,))), axis=2, keepdims=True)
    p95 = np.percentile(residual, 95)
    if p95 <= 1e-6:
        return np.ones((*target.shape[:2], 1), dtype=np.float32)
    mask = np.clip(residual / p95, 0.0, 1.0)
    return (floor + (1.0 - floor) * mask).astype(np.float32)


def apply_fingerprint(
    target: np.ndarray,
    fingerprint: np.ndarray,
    use_texture_mask: bool,
    jpeg_safe_rounding: bool,
) -> np.ndarray:
    if target.shape != fingerprint.shape:
        raise ValueError(
            f"Target and fingerprint shapes must match. target={target.shape}, "
            f"fingerprint={fingerprint.shape}"
        )
    perturbation = fingerprint
    if use_texture_mask:
        perturbation = perturbation * texture_mask(target)
    forged = target + perturbation
    if jpeg_safe_rounding:
        # Quantize to even pixel values. This can make weak high-frequency noise
        # slightly more stable under some pipelines, at a small quality cost.
        forged = np.rint(forged / 2.0) * 2.0
    return forged


def validate_dataset(paths: DatasetPaths) -> None:
    clean_files = {p.name for p in paths.clean_targets.glob("*.png")}
    missing_clean = sorted(EXPECTED_IMAGE_NAMES - clean_files, key=lambda x: int(x[:-4]))
    if missing_clean:
        raise FileNotFoundError(f"Missing clean target images: {missing_clean[:10]}")

    for wm_name, start, stop in CATEGORIES:
        wm_dir = paths.watermarked_sources / wm_name
        if not wm_dir.is_dir():
            raise FileNotFoundError(f"Missing source directory: {wm_dir}")
        expected = {f"src_{i}.png" for i in range(start, stop + 1)}
        found = {p.name for p in wm_dir.glob("*.png")}
        missing = sorted(expected - found, key=lambda x: int(x.split("_")[1][:-4]))
        if missing:
            raise FileNotFoundError(f"Missing source images for {wm_name}: {missing[:10]}")


def validate_output_dir(out_dir: Path) -> None:
    files = sorted(out_dir.glob("*.png"), key=numeric_key)
    names = {p.name for p in files}
    if names != EXPECTED_IMAGE_NAMES:
        missing = sorted(EXPECTED_IMAGE_NAMES - names, key=lambda x: int(x[:-4]))
        extra = sorted(names - EXPECTED_IMAGE_NAMES)
        raise RuntimeError(
            f"Output validation failed. Missing={missing[:10]}, extra={extra[:10]}"
        )
    for path in files:
        with Image.open(path) as img:
            if img.mode not in {"RGB", "RGBA"}:
                raise RuntimeError(f"Unexpected image mode for {path.name}: {img.mode}")


def make_flat_zip(out_dir: Path, zip_out: Path) -> None:
    validate_output_dir(out_dir)
    zip_out.parent.mkdir(parents=True, exist_ok=True)
    if zip_out.exists():
        zip_out.unlink()
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for number in range(1, 201):
            img_path = out_dir / f"{number}.png"
            zipf.write(img_path, arcname=img_path.name)

    # Re-open the ZIP to guarantee it contains exactly flat 1.png ... 200.png.
    with zipfile.ZipFile(zip_out, "r") as zipf:
        names = set(zipf.namelist())
    if names != EXPECTED_IMAGE_NAMES:
        raise RuntimeError("ZIP validation failed: archive is not a flat 200-image submission.")


def build_submission(args: argparse.Namespace) -> None:
    dataset = ensure_dataset(args.zip_file, args.dataset_dir)
    validate_dataset(dataset)

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root: {dataset.root}")
    print(f"Output directory: {args.out_dir}")
    print(f"ZIP output: {args.zip_out}")
    print(
        "Config: "
        f"strength={args.strength}, sigmas={args.sigmas}, "
        f"aggregation={args.aggregation}, whiten_sources={args.whiten_sources}, "
        f"texture_mask={args.texture_mask}"
    )

    total = 0
    for wm_name, start, stop in CATEGORIES:
        source_dir = dataset.watermarked_sources / wm_name
        source_paths = sorted(source_dir.glob("*.png"), key=numeric_key)
        fingerprint = estimate_fingerprint(
            source_paths=source_paths,
            sigmas=args.sigmas,
            aggregation=args.aggregation,
            whiten_sources=args.whiten_sources,
            strength=args.strength,
        )
        print(
            f"{wm_name}: estimated fingerprint from {len(source_paths)} sources; "
            f"shape={fingerprint.shape}; perturbation range="
            f"[{fingerprint.min():.2f}, {fingerprint.max():.2f}]"
        )

        for number in range(start, stop + 1):
            target_path = dataset.clean_targets / f"{number}.png"
            target = read_rgb(target_path)
            if target.shape != fingerprint.shape:
                raise ValueError(
                    f"Size mismatch for {target_path.name}: target shape {target.shape}, "
                    f"fingerprint shape {fingerprint.shape}."
                )
            forged = apply_fingerprint(
                target=target,
                fingerprint=fingerprint,
                use_texture_mask=args.texture_mask,
                jpeg_safe_rounding=args.jpeg_safe_rounding,
            )
            save_rgb(forged, args.out_dir / f"{number}.png")
            total += 1

    if total != 200:
        raise RuntimeError(f"Expected to process 200 images, processed {total}.")
    make_flat_zip(args.out_dir, args.zip_out)
    print(f"Done. Created {args.zip_out} with exactly 200 flat PNG files.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a watermark-forgery submission using residual fingerprint transfer."
    )
    parser.add_argument("--zip_file", type=Path, default=Path("Dataset.zip"))
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("_dataset_extracted_residual_fingerprint"),
        help="Extraction/location directory. The script also detects folders in the CWD.",
    )
    parser.add_argument(
        "--out_dir", type=Path, default=Path("submission_temp_residual_fingerprint")
    )
    parser.add_argument(
        "--zip_out", type=Path, default=Path("submission_residual_fingerprint.zip")
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=6.0,
        help="Perturbation strength in approximate pixel levels. Try 3, 6, 9, 12.",
    )
    parser.add_argument(
        "--sigmas",
        type=parse_float_list,
        default=[1.0, 2.0, 4.0],
        help="Comma-separated Gaussian blur radii used to extract source residuals.",
    )
    parser.add_argument(
        "--aggregation",
        choices=("median", "mean", "trimmed_mean"),
        default="median",
    )
    parser.add_argument(
        "--whiten_sources",
        action="store_true",
        help="Normalize each source residual before aggregation. Useful when sources have uneven texture.",
    )
    parser.add_argument(
        "--texture_mask",
        action="store_true",
        help="Hide more perturbation in textured target regions; may reduce detector strength.",
    )
    parser.add_argument(
        "--jpeg_safe_rounding",
        action="store_true",
        help="Optional even-value quantization. Usually keep disabled for PNG submissions.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        build_submission(args)
    except Exception as exc:  # noqa: BLE001 - user-facing CLI should print clean errors.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
