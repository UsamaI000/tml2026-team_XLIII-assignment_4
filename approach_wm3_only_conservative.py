#!/usr/bin/env python3
"""
WM_3-only Watermark Forgery Experiments
=======================================

This script is for one-WM-at-a-time diagnosis. It starts from an existing
best submission ZIP, keeps images 51.png ... 75.png unchanged, and regenerates
ONLY the WM_3 target batch: 51.png ... 75.png.

Why:
    The assignment has 8 watermark families. A global recipe can hide which
    family is helping or hurting. By changing only WM_3, leaderboard movement
    tells us whether the WM_3 configuration is a bottleneck.

It does not modify task_template.py or submission.py.
Output is validated: exactly 1.png ... 200.png at the ZIP root.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

EXPECTED_IMAGE_NAMES = {f"{i}.png" for i in range(1, 201)}
WM3_TARGETS = range(51, 76)


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    clean_targets: Path
    watermarked_sources: Path


@dataclass(frozen=True)
class WM3Config:
    residual_strength: float
    residual_sigmas: Tuple[float, ...]
    frequency_strength: float = 0.75
    freq_low: float = 0.04
    freq_high: float = 0.20
    aggregation: str = "median"      # median, mean, trimmed_mean
    channel_mode: str = "rgb"        # rgb, luma
    perturbation_clip: float = 14.0


# These profiles change only WM_3. Use current-best submission ZIP as --base_zip.
# Baseline context:
#   - per_wm_consistency_v1 uses WM_3: residual_strength=4, sigma=6, f=0.75, band=0.04-0.20.
#   - global hybrid used sigma=4. These profiles test whether WM_3 prefers global, broader,
#     lower frequency, no frequency, or different aggregation.
PROFILES: Dict[str, WM3Config] = {
    # Diagnostic: revert only WM_3 to the old global-hybrid scale.
    "wm3_global_revert_sig4": WM3Config(4.0, (4.0,), 0.75, 0.04, 0.20),

    # Local strength sweep around the current per-WM setting sigma=6.
    "wm3_sig6_s35": WM3Config(3.5, (6.0,), 0.75, 0.04, 0.20),
    "wm3_sig6_s45": WM3Config(4.5, (6.0,), 0.75, 0.04, 0.20),

    # Broader residual tests. If WM_3 is more low/mid-frequency, these may help.
    "wm3_sig8": WM3Config(4.0, (8.0,), 0.75, 0.04, 0.20),
    "wm3_sig10": WM3Config(4.0, (10.0,), 0.75, 0.04, 0.20),
    "wm3_sig12": WM3Config(4.0, (12.0,), 0.75, 0.04, 0.20),

    # Frequency diagnostics for WM_3.
    "wm3_no_freq": WM3Config(4.0, (6.0,), 0.00, 0.04, 0.20),
    "wm3_freq_narrow": WM3Config(4.0, (6.0,), 0.75, 0.04, 0.16),
    "wm3_freq_wide": WM3Config(4.0, (6.0,), 0.75, 0.03, 0.25),
    "wm3_freq_lowmid": WM3Config(4.0, (6.0,), 0.75, 0.02, 0.12),

    # Aggregation diagnostics.
    "wm3_mean": WM3Config(4.0, (6.0,), 0.75, 0.04, 0.20, aggregation="mean"),
    "wm3_trimmed": WM3Config(4.0, (6.0,), 0.75, 0.04, 0.20, aggregation="trimmed_mean"),

    # Channel diagnostic.
    "wm3_luma": WM3Config(4.0, (6.0,), 0.75, 0.04, 0.20, channel_mode="luma"),
    "wm3_freq_high014": WM3Config(4.0, (6.0,), 0.75, 0.04, 0.14),
    "wm3_freq_high015": WM3Config(4.0, (6.0,), 0.75, 0.04, 0.15),
    "wm3_freq_high017": WM3Config(4.0, (6.0,), 0.75, 0.04, 0.17),
    "wm3_freq_narrow_f060": WM3Config(4.0, (6.0,), 0.60, 0.04, 0.16),
    "wm3_freq_narrow_f090": WM3Config(4.0, (6.0,), 0.90, 0.04, 0.16),
    "wm3_sig4_freq_narrow": WM3Config(4.0, (4.0,), 0.75, 0.04, 0.16),
}


def parse_float_list(text: str) -> Tuple[float, ...]:
    values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def numeric_key(path: Path) -> int:
    stem = path.stem
    n = stem.split("_")[-1]
    try:
        return int(n)
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
        raise FileNotFoundError("Could not locate clean_targets/ and watermarked_sources/.")
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
    return np.maximum.reduce([
        mad_scale,
        p95_scale / 2.0,
        std_scale,
        np.full_like(std_scale, eps),
    ])


def normalize_fingerprint(fingerprint: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return np.zeros_like(fingerprint, dtype=np.float32)
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


def aggregate_stack(stack: np.ndarray, method: str) -> np.ndarray:
    if method == "median":
        return np.median(stack, axis=0)
    if method == "mean":
        return np.mean(stack, axis=0)
    if method == "trimmed_mean":
        if stack.shape[0] < 5:
            return np.mean(stack, axis=0)
        sorted_stack = np.sort(stack, axis=0)
        trim = max(1, int(round(0.1 * stack.shape[0])))
        return np.mean(sorted_stack[trim:-trim], axis=0)
    raise ValueError(f"Unknown aggregation method: {method}")


def estimate_residual_fingerprint(source_paths: Sequence[Path], cfg: WM3Config) -> np.ndarray:
    residuals = []
    expected_shape = None
    for path in source_paths:
        image = read_rgb(path)
        if expected_shape is None:
            expected_shape = image.shape
        elif image.shape != expected_shape:
            raise RuntimeError(f"Source size mismatch in {path}")
        residual = highpass_residual(image, cfg.residual_sigmas)
        residual = residual - np.median(residual, axis=(0, 1), keepdims=True)
        residuals.append(residual)
    raw = aggregate_stack(np.stack(residuals, axis=0), cfg.aggregation)
    return normalize_fingerprint(raw, cfg.residual_strength)


def radial_bandpass_mask(height: int, width: int, low: float, high: float) -> np.ndarray:
    if not (0.0 <= low < high <= 0.5):
        raise ValueError("Expected 0 <= low < high <= 0.5")
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    return ((radius >= low) & (radius <= high)).astype(np.float32)[:, :, None]


def bandpass_residual(image: np.ndarray, low: float, high: float) -> np.ndarray:
    h, w, _ = image.shape
    centered = image - np.mean(image, axis=(0, 1), keepdims=True)
    spectrum = np.fft.fft2(centered, axes=(0, 1))
    filtered = np.fft.ifft2(spectrum * radial_bandpass_mask(h, w, low, high), axes=(0, 1)).real
    return filtered.astype(np.float32)


def estimate_frequency_fingerprint(source_paths: Sequence[Path], cfg: WM3Config) -> np.ndarray:
    if cfg.frequency_strength <= 0:
        image = read_rgb(source_paths[0])
        return np.zeros_like(image, dtype=np.float32)
    residuals = []
    expected_shape = None
    for path in source_paths:
        image = read_rgb(path)
        if expected_shape is None:
            expected_shape = image.shape
        elif image.shape != expected_shape:
            raise RuntimeError(f"Source size mismatch in {path}")
        residual = bandpass_residual(image, cfg.freq_low, cfg.freq_high)
        residual = residual - np.mean(residual, axis=(0, 1), keepdims=True)
        residual = residual / robust_channel_scale(residual)
        residuals.append(residual)
    raw = aggregate_stack(np.stack(residuals, axis=0), cfg.aggregation)
    return normalize_fingerprint(raw, cfg.frequency_strength)


def apply_channel_mode(perturbation: np.ndarray, channel_mode: str) -> np.ndarray:
    if channel_mode == "rgb":
        return perturbation.astype(np.float32)
    if channel_mode == "luma":
        y = 0.299 * perturbation[:, :, 0:1] + 0.587 * perturbation[:, :, 1:2] + 0.114 * perturbation[:, :, 2:3]
        return np.repeat(y, 3, axis=2).astype(np.float32)
    raise ValueError(f"Unknown channel mode: {channel_mode}")


def validate_base_zip(base_zip: Path) -> None:
    if not base_zip.is_file():
        raise FileNotFoundError(
            f"Missing base ZIP: {base_zip}. Use your current best ZIP here, e.g. "
            "submission_wm4_edge_only.zip or submission_per_wm_consistency_v1.zip."
        )
    with zipfile.ZipFile(base_zip, "r") as zipf:
        names = set(zipf.namelist())
    if names != EXPECTED_IMAGE_NAMES:
        missing = sorted(EXPECTED_IMAGE_NAMES - names, key=lambda x: int(x[:-4]))
        extra = sorted(names - EXPECTED_IMAGE_NAMES)
        raise RuntimeError(f"Base ZIP is not a flat 200-image submission. Missing={missing[:10]}, extra={extra[:10]}")


def extract_base_zip(base_zip: Path, out_dir: Path) -> None:
    validate_base_zip(base_zip)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip, "r") as zipf:
        zipf.extractall(out_dir)


def make_flat_zip(out_dir: Path, zip_out: Path) -> None:
    names = {p.name for p in out_dir.glob("*.png")}
    if names != EXPECTED_IMAGE_NAMES:
        missing = sorted(EXPECTED_IMAGE_NAMES - names, key=lambda x: int(x[:-4]))
        extra = sorted(names - EXPECTED_IMAGE_NAMES)
        raise RuntimeError(f"Output mismatch. Missing={missing[:10]}, extra={extra[:10]}")
    zip_out.parent.mkdir(parents=True, exist_ok=True)
    if zip_out.exists():
        zip_out.unlink()
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zipf:
        for i in range(1, 201):
            p = out_dir / f"{i}.png"
            zipf.write(p, arcname=p.name)
    with zipfile.ZipFile(zip_out, "r") as zipf:
        if set(zipf.namelist()) != EXPECTED_IMAGE_NAMES:
            raise RuntimeError("ZIP validation failed: output is not flat 1.png ... 200.png")


def mean_psnr_for_wm3(clean_dir: Path, out_dir: Path) -> float:
    values = []
    for i in WM3_TARGETS:
        clean = read_rgb(clean_dir / f"{i}.png")
        forged = read_rgb(out_dir / f"{i}.png")
        mse = np.mean((clean - forged) ** 2)
        values.append(float("inf") if mse <= 1e-12 else 20.0 * np.log10(255.0 / np.sqrt(mse)))
    return float(np.mean(values))


def build_submission(args: argparse.Namespace) -> None:
    dataset = ensure_dataset(args.zip_file, args.dataset_dir)
    cfg = PROFILES[args.profile]

    source_dir = dataset.watermarked_sources / "WM_3"
    source_paths = sorted(source_dir.glob("*.png"), key=numeric_key)
    if len(source_paths) != 25:
        raise RuntimeError(f"Expected 25 WM_3 source images in {source_dir}, found {len(source_paths)}")

    print(f"Base ZIP: {args.base_zip}")
    print(f"Output directory: {args.out_dir}")
    print(f"ZIP output: {args.zip_out}")
    print(f"Profile: {args.profile} -> {cfg}")

    extract_base_zip(args.base_zip, args.out_dir)

    residual_fp = estimate_residual_fingerprint(source_paths, cfg)
    frequency_fp = estimate_frequency_fingerprint(source_paths, cfg)
    perturbation = apply_channel_mode(residual_fp + frequency_fp, cfg.channel_mode)
    perturbation = np.clip(perturbation, -cfg.perturbation_clip, cfg.perturbation_clip).astype(np.float32)

    print(
        f"WM_3 perturbation range=[{perturbation.min():.2f},{perturbation.max():.2f}], "
        f"std={perturbation.std():.3f}"
    )

    for i in WM3_TARGETS:
        target = read_rgb(dataset.clean_targets / f"{i}.png")
        if target.shape != perturbation.shape:
            raise RuntimeError(f"Shape mismatch for target {i}.png: target={target.shape}, perturbation={perturbation.shape}")
        save_rgb(target + perturbation, args.out_dir / f"{i}.png")

    make_flat_zip(args.out_dir, args.zip_out)
    if args.print_psnr:
        print(f"WM_3 mean PSNR vs clean targets: {mean_psnr_for_wm3(dataset.clean_targets, args.out_dir):.2f} dB")
    print(f"Done. Created {args.zip_out}. Only 51.png ... 75.png were regenerated.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run WM_3-only forging experiments on top of an existing base ZIP.")
    parser.add_argument("--zip_file", type=Path, default=Path("Dataset.zip"))
    parser.add_argument("--dataset_dir", type=Path, default=Path("_dataset_extracted_wm3_only"))
    parser.add_argument(
        "--base_zip",
        type=Path,
        default=Path("submission_per_wm_consistency_v1.zip"),
        help="Existing 200-image submission ZIP to keep for all non-WM3 images. Prefer current best ZIP.",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("submission_temp_wm3_only"))
    parser.add_argument("--zip_out", type=Path, default=Path("submission_wm3_only.zip"))
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
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
