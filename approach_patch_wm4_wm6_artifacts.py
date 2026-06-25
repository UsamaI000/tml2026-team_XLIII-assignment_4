#!/usr/bin/env python3
"""
Patch WM_4 Edge Artifact + WM_6 FFT Peak Artifact
=================================================

This script is intentionally designed after the per-WM adaptive profile improved
the leaderboard. Instead of recomputing all 8 watermark groups, it starts from a
known good complete submission ZIP, such as submission_per_wm_consistency_v1.zip,
and only patches the two groups where visual/FFT diagnostics revealed specific
reusable artifacts:

- WM_4 / targets 76-100: visible border/corner black-white grid artifacts.
- WM_6 / targets 126-150: sparse FFT peak / repeated grid pattern.

It produces a new flat ZIP containing exactly 1.png ... 200.png.
It does not modify task_template.py or submission.py.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

EXPECTED_IMAGE_NAMES = {f"{i}.png" for i in range(1, 201)}


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    clean_targets: Path
    watermarked_sources: Path


@dataclass(frozen=True)
class PatchProfile:
    edge_strength: float
    edge_width_ratio: float
    edge_highpass_sigma: float
    edge_clip: float
    peak_strength: float
    peak_low: float
    peak_high: float
    peak_top_fraction: float
    peak_input_sigma: float
    peak_clip: float
    final_clip: float


PROFILES = {
    # First submit this: least risk, small patch on top of current best v1.
    "patch_conservative": PatchProfile(
        edge_strength=0.55, edge_width_ratio=0.075, edge_highpass_sigma=0.8, edge_clip=6.0,
        peak_strength=0.45, peak_low=0.035, peak_high=0.45, peak_top_fraction=0.004,
        peak_input_sigma=1.0, peak_clip=6.0, final_clip=16.0,
    ),
    # Stronger targeted signal, still controlled.
    "patch_balanced": PatchProfile(
        edge_strength=0.95, edge_width_ratio=0.080, edge_highpass_sigma=0.7, edge_clip=8.0,
        peak_strength=0.80, peak_low=0.035, peak_high=0.45, peak_top_fraction=0.006,
        peak_input_sigma=1.0, peak_clip=8.0, final_clip=18.0,
    ),
    # Aggressive. Use only if balanced helps.
    "patch_aggressive": PatchProfile(
        edge_strength=1.35, edge_width_ratio=0.090, edge_highpass_sigma=0.7, edge_clip=10.0,
        peak_strength=1.15, peak_low=0.030, peak_high=0.47, peak_top_fraction=0.008,
        peak_input_sigma=1.0, peak_clip=10.0, final_clip=20.0,
    ),
    # Isolates WM_4 border effect.
    "wm4_edge_only": PatchProfile(
        edge_strength=0.95, edge_width_ratio=0.080, edge_highpass_sigma=0.7, edge_clip=8.0,
        peak_strength=0.0, peak_low=0.035, peak_high=0.45, peak_top_fraction=0.006,
        peak_input_sigma=1.0, peak_clip=8.0, final_clip=18.0,
    ),
    # Isolates WM_6 sparse FFT effect.
    "wm6_peak_only": PatchProfile(
        edge_strength=0.0, edge_width_ratio=0.080, edge_highpass_sigma=0.7, edge_clip=8.0,
        peak_strength=0.80, peak_low=0.035, peak_high=0.45, peak_top_fraction=0.006,
        peak_input_sigma=1.0, peak_clip=8.0, final_clip=18.0,
    ),
}


def numeric_key(path: Path) -> int:
    return int(path.stem.split("_")[-1])


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
    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(dataset_dir)
    found = find_dataset_root(dataset_dir.resolve())
    if found is None:
        raise FileNotFoundError("Could not find clean_targets/ and watermarked_sources/.")
    return found


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32)


def save_rgb(array: np.ndarray, path: Path) -> None:
    clipped = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    Image.fromarray(clipped, mode="RGB").save(path)


def gaussian_blur_rgb(array: np.ndarray, sigma: float) -> np.ndarray:
    img = Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius=float(sigma))), dtype=np.float32)


def smooth_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    img = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return (np.asarray(blurred, dtype=np.float32) / 255.0)[:, :, None]


def border_mask(height: int, width: int, ratio: float) -> np.ndarray:
    bw = max(4, int(round(min(height, width) * ratio)))
    mask = np.zeros((height, width), dtype=np.float32)
    mask[:bw, :] = 1
    mask[-bw:, :] = 1
    mask[:, :bw] = 1
    mask[:, -bw:] = 1
    return smooth_mask(mask, max(1.0, bw / 3.0)).astype(np.float32)


def robust_channel_scale(array: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    median = np.median(array, axis=(0, 1), keepdims=True)
    centered = array - median
    mad = 1.4826 * np.median(np.abs(centered), axis=(0, 1), keepdims=True)
    p95 = np.percentile(np.abs(centered), 95, axis=(0, 1), keepdims=True)
    std = np.std(centered, axis=(0, 1), keepdims=True)
    return np.maximum.reduce([mad, p95 / 2.0, std, np.full_like(std, eps)])


def robust_masked_scale(array: np.ndarray, mask: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    selected = mask[:, :, 0] > 0.15
    if selected.sum() < 16:
        return robust_channel_scale(array, eps)
    vals = array[selected, :]
    med = np.median(vals, axis=0, keepdims=True)
    centered = vals - med
    mad = 1.4826 * np.median(np.abs(centered), axis=0, keepdims=True)
    p95 = np.percentile(np.abs(centered), 95, axis=0, keepdims=True)
    std = np.std(centered, axis=0, keepdims=True)
    scale = np.maximum.reduce([mad, p95 / 2.0, std, np.full_like(std, eps)])
    return scale.reshape(1, 1, 3)


def normalize_full(fp: np.ndarray, strength: float, clip_value: float) -> np.ndarray:
    if strength <= 0:
        return np.zeros_like(fp, dtype=np.float32)
    centered = fp - np.mean(fp, axis=(0, 1), keepdims=True)
    out = centered / robust_channel_scale(centered) * float(strength)
    return np.clip(out, -clip_value, clip_value).astype(np.float32)


def normalize_masked(fp: np.ndarray, mask: np.ndarray, strength: float, clip_value: float) -> np.ndarray:
    if strength <= 0:
        return np.zeros_like(fp, dtype=np.float32)
    selected = mask[:, :, 0] > 0.15
    centered = fp.copy().astype(np.float32)
    if selected.sum() >= 16:
        centered -= np.median(centered[selected, :], axis=0).reshape(1, 1, 3)
    else:
        centered -= np.median(centered, axis=(0, 1), keepdims=True)
    out = centered / robust_masked_scale(centered, mask) * float(strength)
    return np.clip(out * mask, -clip_value, clip_value).astype(np.float32)


def estimate_wm4_edge_patch(source_dir: Path, p: PatchProfile) -> np.ndarray:
    source_paths = sorted(source_dir.glob("*.png"), key=numeric_key)
    residuals = []
    for path in source_paths:
        img = read_rgb(path)
        res = img - gaussian_blur_rgb(img, p.edge_highpass_sigma)
        res -= np.median(res, axis=(0, 1), keepdims=True)
        residuals.append(res.astype(np.float32))
    raw = np.median(np.stack(residuals, axis=0), axis=0)
    h, w, _ = raw.shape
    mask = border_mask(h, w, p.edge_width_ratio)
    return normalize_masked(raw, mask, p.edge_strength, p.edge_clip)


def radial_mask(height: int, width: int, low: float, high: float) -> np.ndarray:
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    r = np.sqrt(fx * fx + fy * fy)
    return (r >= low) & (r <= high)


def estimate_wm6_peak_patch(source_dir: Path, p: PatchProfile) -> np.ndarray:
    source_paths = sorted(source_dir.glob("*.png"), key=numeric_key)
    if p.peak_strength <= 0:
        return np.zeros_like(read_rgb(source_paths[0]), dtype=np.float32)
    spectra = []
    for path in source_paths:
        img = read_rgb(path)
        res = img - gaussian_blur_rgb(img, p.peak_input_sigma)
        res -= np.mean(res, axis=(0, 1), keepdims=True)
        res /= robust_channel_scale(res)
        spectra.append(np.fft.fft2(res, axes=(0, 1)))
    avg_spectrum = np.mean(np.stack(spectra, axis=0), axis=0)
    h, w, _ = avg_spectrum.shape
    eligible = radial_mask(h, w, p.peak_low, p.peak_high)
    mag = np.mean(np.abs(avg_spectrum), axis=2)
    threshold = np.quantile(mag[eligible], max(0.0, min(0.9999, 1.0 - p.peak_top_fraction)))
    peak_mask = ((mag >= threshold) & eligible).astype(np.float32)[:, :, None]
    sparse = np.fft.ifft2(avg_spectrum * peak_mask, axes=(0, 1)).real.astype(np.float32)
    return normalize_full(sparse, p.peak_strength, p.peak_clip)


def extract_base_zip(base_zip: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not base_zip.is_file():
        raise FileNotFoundError(f"Missing base ZIP: {base_zip}")
    with zipfile.ZipFile(base_zip, "r") as zf:
        names = set(zf.namelist())
        if names != EXPECTED_IMAGE_NAMES:
            raise RuntimeError("Base ZIP must contain exactly flat 1.png ... 200.png")
        zf.extractall(out_dir)


def validate_and_zip(out_dir: Path, zip_out: Path) -> None:
    names = {p.name for p in out_dir.glob("*.png")}
    if names != EXPECTED_IMAGE_NAMES:
        missing = sorted(EXPECTED_IMAGE_NAMES - names, key=lambda x: int(x[:-4]))
        extra = sorted(names - EXPECTED_IMAGE_NAMES)
        raise RuntimeError(f"Output mismatch. Missing={missing[:10]}, extra={extra[:10]}")
    if zip_out.exists():
        zip_out.unlink()
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(1, 201):
            img_path = out_dir / f"{i}.png"
            zf.write(img_path, arcname=img_path.name)
    with zipfile.ZipFile(zip_out, "r") as zf:
        if set(zf.namelist()) != EXPECTED_IMAGE_NAMES:
            raise RuntimeError("ZIP validation failed")


def mean_psnr(clean_dir: Path, out_dir: Path) -> float:
    vals = []
    for i in range(1, 201):
        clean = read_rgb(clean_dir / f"{i}.png")
        forged = read_rgb(out_dir / f"{i}.png")
        mse = np.mean((clean - forged) ** 2)
        vals.append(float("inf") if mse <= 1e-12 else 20 * np.log10(255.0 / np.sqrt(mse)))
    return float(np.mean(vals))


def build(args: argparse.Namespace) -> None:
    dataset = ensure_dataset(args.zip_file, args.dataset_dir)
    profile = PROFILES[args.profile]
    extract_base_zip(args.base_zip, args.out_dir)

    print(f"Dataset root: {dataset.root}")
    print(f"Base ZIP: {args.base_zip}")
    print(f"Profile: {args.profile}")
    print(f"Output ZIP: {args.zip_out}")

    wm4_patch = estimate_wm4_edge_patch(dataset.watermarked_sources / "WM_4", profile)
    wm6_patch = estimate_wm6_peak_patch(dataset.watermarked_sources / "WM_6", profile)
    print(f"WM_4 edge patch range: [{wm4_patch.min():.2f}, {wm4_patch.max():.2f}]")
    print(f"WM_6 peak patch range: [{wm6_patch.min():.2f}, {wm6_patch.max():.2f}]")

    for i in range(76, 101):
        img = read_rgb(args.out_dir / f"{i}.png")
        save_rgb(img + np.clip(wm4_patch, -profile.final_clip, profile.final_clip), args.out_dir / f"{i}.png")

    for i in range(126, 151):
        img = read_rgb(args.out_dir / f"{i}.png")
        save_rgb(img + np.clip(wm6_patch, -profile.final_clip, profile.final_clip), args.out_dir / f"{i}.png")

    validate_and_zip(args.out_dir, args.zip_out)
    if args.print_psnr:
        print(f"Mean PSNR vs clean targets: {mean_psnr(dataset.clean_targets, args.out_dir):.2f} dB")
    print(f"Done. Created {args.zip_out}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Patch a known good submission ZIP with WM_4/WM_6 artifacts.")
    p.add_argument("--zip_file", type=Path, default=Path("Dataset.zip"))
    p.add_argument("--dataset_dir", type=Path, default=Path("Dataset"))
    p.add_argument("--base_zip", type=Path, default=Path("submission_per_wm_consistency_v1.zip"))
    p.add_argument("--out_dir", type=Path, default=Path("submission_temp_patch_artifacts"))
    p.add_argument("--zip_out", type=Path, default=Path("submission_patch_artifacts.zip"))
    p.add_argument("--profile", choices=tuple(PROFILES.keys()), default="patch_conservative")
    p.add_argument("--print_psnr", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        build(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
