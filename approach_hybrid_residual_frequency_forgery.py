#!/usr/bin/env python3
"""
Hybrid Residual + Frequency Fingerprint Watermark Forgery
=========================================================

This script creates a flat 200-image submission ZIP for the watermark-forgery
assignment. It does NOT modify task_template.py or submission.py.

Why this variant exists:
    Your best result so far came from residual fingerprint transfer with
    strength=4 and sigmas=4.0. That suggests the useful watermark signal is in a
    broader mid-frequency residual band. This hybrid keeps that winning residual
    component and adds a small Fourier band-pass component that may capture
    periodic or frequency-domain watermark traces missed by Gaussian residuals.

Default configuration:
    residual_strength=4.0, residual_sigmas=4.0
    frequency_strength=0.75, frequency low/high = 0.04/0.18

Recommended first command:
    python approach_hybrid_residual_frequency_forgery.py \
        --zip_file Dataset.zip \
        --dataset_dir Dataset \
        --out_dir submission_temp_hybrid_r4_f075_b0418 \
        --zip_out submission_hybrid_r4_f075_b0418.zip \
        --residual_strength 4 \
        --residual_sigmas 4.0 \
        --frequency_strength 0.75 \
        --freq_low 0.04 \
        --freq_high 0.18
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter


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
    stem = path.stem
    number_text = stem.split("_")[-1]
    try:
        return int(number_text)
    except ValueError as exc:
        raise ValueError(f"Cannot extract numeric id from filename: {path.name}") from exc


def find_dataset_root(base: Path) -> DatasetPaths | None:
    for root in (base, base / "Dataset"):
        clean = root / "clean_targets"
        sources = root / "watermarked_sources"
        if clean.is_dir() and sources.is_dir():
            return DatasetPaths(root=root, clean_targets=clean, watermarked_sources=sources)
    return None


def ensure_dataset(zip_file: Path, dataset_dir: Path) -> DatasetPaths:
    found = find_dataset_root(dataset_dir.resolve()) or find_dataset_root(Path.cwd())
    if found is not None:
        return found

    if not zip_file.is_file():
        raise FileNotFoundError(f"Missing dataset ZIP: {zip_file}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_file} -> {dataset_dir}")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    found = find_dataset_root(dataset_dir.resolve())
    if found is None:
        raise FileNotFoundError(
            "Could not locate clean_targets/ and watermarked_sources/ after extraction."
        )
    return found


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


def robust_channel_scale(array: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    median = np.median(array, axis=(0, 1), keepdims=True)
    centered = array - median
    mad_scale = 1.4826 * np.median(np.abs(centered), axis=(0, 1), keepdims=True)
    p95_scale = np.percentile(np.abs(centered), 95, axis=(0, 1), keepdims=True)
    std_scale = np.std(centered, axis=(0, 1), keepdims=True)
    return np.maximum.reduce(
        [mad_scale, p95_scale / 2.0, std_scale, np.full_like(std_scale, eps)]
    )


def normalize_fingerprint(fingerprint: np.ndarray, strength: float) -> np.ndarray:
    centered = fingerprint - np.mean(fingerprint, axis=(0, 1), keepdims=True)
    normalized = centered / robust_channel_scale(centered) * float(strength)
    max_abs = max(2.0, 3.0 * float(strength))
    return np.clip(normalized, -max_abs, max_abs).astype(np.float32)


def highpass_residual(image: np.ndarray, sigmas: Sequence[float]) -> np.ndarray:
    residuals = []
    for sigma in sigmas:
        if sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {sigma}")
        residuals.append(image - gaussian_blur_rgb(image, sigma))
    return np.mean(np.stack(residuals, axis=0), axis=0)


def estimate_residual_fingerprint(
    source_paths: Sequence[Path], sigmas: Sequence[float], strength: float
) -> np.ndarray:
    residuals = []
    expected_shape = None
    for path in source_paths:
        image = read_rgb(path)
        if expected_shape is None:
            expected_shape = image.shape
        elif image.shape != expected_shape:
            raise RuntimeError(f"Source size mismatch in {path}")

        residual = highpass_residual(image, sigmas)
        residual = residual - np.median(residual, axis=(0, 1), keepdims=True)
        residuals.append(residual)

    fingerprint = np.median(np.stack(residuals, axis=0), axis=0)
    return normalize_fingerprint(fingerprint, strength=strength)


def radial_bandpass_mask(height: int, width: int, low: float, high: float) -> np.ndarray:
    if not (0.0 <= low < high <= 0.5):
        raise ValueError("Expected 0 <= low < high <= 0.5 for normalized frequency cutoffs")
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    return ((radius >= low) & (radius <= high)).astype(np.float32)[:, :, None]


def bandpass_residual(image: np.ndarray, low: float, high: float) -> np.ndarray:
    height, width, _ = image.shape
    centered = image - np.mean(image, axis=(0, 1), keepdims=True)
    spectrum = np.fft.fft2(centered, axes=(0, 1))
    mask = radial_bandpass_mask(height, width, low=low, high=high)
    filtered = np.fft.ifft2(spectrum * mask, axes=(0, 1)).real.astype(np.float32)
    return filtered


def estimate_frequency_fingerprint(
    source_paths: Sequence[Path], low: float, high: float, strength: float
) -> np.ndarray:
    residuals = []
    expected_shape = None
    for path in source_paths:
        image = read_rgb(path)
        if expected_shape is None:
            expected_shape = image.shape
        elif image.shape != expected_shape:
            raise RuntimeError(f"Source size mismatch in {path}")

        residual = bandpass_residual(image, low=low, high=high)
        residual = residual - np.mean(residual, axis=(0, 1), keepdims=True)
        residual = residual / robust_channel_scale(residual)
        residuals.append(residual)

    fingerprint = np.median(np.stack(residuals, axis=0), axis=0)
    return normalize_fingerprint(fingerprint, strength=strength)


def apply_channel_mode(perturbation: np.ndarray, channel_mode: str) -> np.ndarray:
    if channel_mode == "rgb":
        return perturbation
    if channel_mode == "luma":
        # Convert RGB perturbation to one luminance-like channel and add the same
        # perturbation to all RGB channels. This can reduce chroma artifacts.
        y = (
            0.299 * perturbation[:, :, 0:1]
            + 0.587 * perturbation[:, :, 1:2]
            + 0.114 * perturbation[:, :, 2:3]
        )
        return np.repeat(y, 3, axis=2).astype(np.float32)
    raise ValueError(f"Unknown channel mode: {channel_mode}")


def validate_dataset(paths: DatasetPaths) -> None:
    clean_names = {p.name for p in paths.clean_targets.glob("*.png")}
    if clean_names != EXPECTED_IMAGE_NAMES:
        missing = sorted(EXPECTED_IMAGE_NAMES - clean_names, key=lambda x: int(x[:-4]))
        extra = sorted(clean_names - EXPECTED_IMAGE_NAMES)
        raise FileNotFoundError(f"Clean target mismatch. Missing={missing[:10]}, extra={extra[:10]}")

    for wm_name, start, stop in CATEGORIES:
        wm_dir = paths.watermarked_sources / wm_name
        if not wm_dir.is_dir():
            raise FileNotFoundError(f"Missing source directory: {wm_dir}")
        expected_count = stop - start + 1
        found_count = len(list(wm_dir.glob("*.png")))
        if found_count != expected_count:
            raise FileNotFoundError(
                f"Expected {expected_count} PNGs in {wm_dir}, found {found_count}"
            )


def validate_and_zip(out_dir: Path, zip_out: Path) -> None:
    names = {p.name for p in out_dir.glob("*.png")}
    if names != EXPECTED_IMAGE_NAMES:
        missing = sorted(EXPECTED_IMAGE_NAMES - names, key=lambda x: int(x[:-4]))
        extra = sorted(names - EXPECTED_IMAGE_NAMES)
        raise RuntimeError(f"Output mismatch. Missing={missing[:10]}, extra={extra[:10]}")

    zip_out.parent.mkdir(parents=True, exist_ok=True)
    if zip_out.exists():
        zip_out.unlink()

    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zipf:
        for number in range(1, 201):
            p = out_dir / f"{number}.png"
            zipf.write(p, arcname=p.name)

    with zipfile.ZipFile(zip_out, "r") as zipf:
        if set(zipf.namelist()) != EXPECTED_IMAGE_NAMES:
            raise RuntimeError("ZIP is not a flat 200-image submission")


def mean_psnr(clean_dir: Path, out_dir: Path) -> float:
    psnrs = []
    for number in range(1, 201):
        clean = read_rgb(clean_dir / f"{number}.png")
        forged = read_rgb(out_dir / f"{number}.png")
        mse = np.mean((clean - forged) ** 2)
        if mse <= 1e-12:
            psnrs.append(float("inf"))
        else:
            psnrs.append(20.0 * np.log10(255.0 / np.sqrt(mse)))
    return float(np.mean(psnrs))


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
        f"residual_strength={args.residual_strength}, residual_sigmas={args.residual_sigmas}, "
        f"frequency_strength={args.frequency_strength}, freq_low={args.freq_low}, "
        f"freq_high={args.freq_high}, channel_mode={args.channel_mode}, "
        f"perturbation_clip={args.perturbation_clip}"
    )

    total = 0
    for wm_name, start, stop in CATEGORIES:
        source_dir = dataset.watermarked_sources / wm_name
        source_paths = sorted(source_dir.glob("*.png"), key=numeric_key)

        residual_fp = estimate_residual_fingerprint(
            source_paths=source_paths,
            sigmas=args.residual_sigmas,
            strength=args.residual_strength,
        )
        frequency_fp = estimate_frequency_fingerprint(
            source_paths=source_paths,
            low=args.freq_low,
            high=args.freq_high,
            strength=args.frequency_strength,
        )

        perturbation = residual_fp + frequency_fp
        perturbation = apply_channel_mode(perturbation, args.channel_mode)
        perturbation = np.clip(
            perturbation,
            -float(args.perturbation_clip),
            float(args.perturbation_clip),
        ).astype(np.float32)

        print(
            f"{wm_name}: residual_range=[{residual_fp.min():.2f},{residual_fp.max():.2f}], "
            f"freq_range=[{frequency_fp.min():.2f},{frequency_fp.max():.2f}], "
            f"combined_range=[{perturbation.min():.2f},{perturbation.max():.2f}]"
        )

        for number in range(start, stop + 1):
            target = read_rgb(dataset.clean_targets / f"{number}.png")
            if target.shape != perturbation.shape:
                raise RuntimeError(
                    f"Shape mismatch for target {number}.png: target={target.shape}, "
                    f"perturbation={perturbation.shape}"
                )
            save_rgb(target + perturbation, args.out_dir / f"{number}.png")
            total += 1

    if total != 200:
        raise RuntimeError(f"Expected 200 outputs, produced {total}")

    validate_and_zip(args.out_dir, args.zip_out)
    if args.print_psnr:
        print(f"Mean PSNR vs clean targets: {mean_psnr(dataset.clean_targets, args.out_dir):.2f} dB")
    print(f"Done. Created {args.zip_out} with exactly 200 flat PNG files.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a hybrid residual+frequency watermark-forgery submission ZIP."
    )
    parser.add_argument("--zip_file", type=Path, default=Path("Dataset.zip"))
    parser.add_argument("--dataset_dir", type=Path, default=Path("_dataset_extracted_hybrid"))
    parser.add_argument("--out_dir", type=Path, default=Path("submission_temp_hybrid"))
    parser.add_argument("--zip_out", type=Path, default=Path("submission_hybrid.zip"))
    parser.add_argument("--residual_strength", type=float, default=4.0)
    parser.add_argument("--residual_sigmas", type=parse_float_list, default=[4.0])
    parser.add_argument("--frequency_strength", type=float, default=0.75)
    parser.add_argument("--freq_low", type=float, default=0.04)
    parser.add_argument("--freq_high", type=float, default=0.18)
    parser.add_argument(
        "--channel_mode",
        choices=("rgb", "luma"),
        default="rgb",
        help="Use RGB perturbation or luminance-only perturbation.",
    )
    parser.add_argument(
        "--perturbation_clip",
        type=float,
        default=14.0,
        help="Clip final combined perturbation to +/- this many pixel levels.",
    )
    parser.add_argument("--print_psnr", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        build_submission(args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
