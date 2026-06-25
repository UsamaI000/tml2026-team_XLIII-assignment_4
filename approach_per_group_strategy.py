#!/usr/bin/env python3
"""
Per-Group Strategy Watermark Forgery
=====================================

Based on diagnostic results, each WM family gets a tailored strategy:

  WM1, WM2, WM3, WM8 : Low-frequency spread-spectrum
                         → Residual with large sigmas (8, 16, 32)
  WM4                 : DWT subband (structured 3x3 FFT grid)
                         → Bandpass targeting subband boundaries
  WM5                 : Spatial template / mid-frequency
                         → Residual with mid sigmas (4, 8)
  WM6                 : Tiled/periodic spatial pattern
                         → Bandpass with low-frequency band
  WM7                 : Vertical stripe frequency watermark
                         → Directional vertical-only FFT mask

Usage (recommended first run — mirrors your best residual config per group):
    python approach_per_group_strategy.py \
        --zip_file Dataset.zip \
        --dataset_dir Dataset \
        --out_dir submission_temp_pergroup_v1 \
        --zip_out submission_pergroup_v1.zip

To tune a specific group without re-running everything, use --only:
    python approach_per_group_strategy.py ... --only WM4 WM7
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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


# ── Per-group config ──────────────────────────────────────────────────────────
# Edit these to tune individual groups without touching anything else.

@dataclass
class GroupConfig:
    strategy: str           # "residual", "bandpass", "vertical"
    strength: float
    # residual params
    sigmas: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    aggregation: str = "median"
    whiten_sources: bool = False
    # bandpass params
    freq_low: float = 0.06
    freq_high: float = 0.40
    # vertical bandpass params (strategy="vertical")
    vert_low: float = 0.06
    vert_high: float = 0.40
    vert_horiz_suppress: float = 0.05  # suppress horizontal freq components below this


# Default per-group configs derived from diagnostic
GROUP_CONFIGS: dict[str, GroupConfig] = {
    # Low-freq spread-spectrum: large sigmas to capture low-frequency watermark signal
    "WM_1": GroupConfig(strategy="residual", strength=6.0, sigmas=[8.0, 16.0, 32.0]),
    "WM_2": GroupConfig(strategy="residual", strength=6.0, sigmas=[8.0, 16.0, 32.0]),
    "WM_3": GroupConfig(strategy="residual", strength=6.0, sigmas=[8.0, 16.0, 32.0]),
    "WM_8": GroupConfig(strategy="residual", strength=6.0, sigmas=[8.0, 16.0, 32.0]),
    # DWT subband: bandpass targeting subband frequency boundaries
    "WM_4": GroupConfig(strategy="bandpass", strength=8.0, freq_low=0.10, freq_high=0.25),
    # Spatial template / mid-frequency
    "WM_5": GroupConfig(strategy="residual", strength=7.0, sigmas=[4.0, 8.0]),
    # Tiled/periodic spatial: low-frequency bandpass
    "WM_6": GroupConfig(strategy="bandpass", strength=6.0, freq_low=0.01, freq_high=0.15),
    # Vertical stripe: directional vertical-only FFT mask
    "WM_7": GroupConfig(strategy="vertical", strength=6.0, vert_low=0.06, vert_high=0.40,
                        vert_horiz_suppress=0.04),
}


# ── Dataset helpers ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    clean_targets: Path
    watermarked_sources: Path


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
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(dataset_dir)
    found = find_dataset_root(dataset_dir.resolve())
    if found is None:
        raise FileNotFoundError("Could not locate clean_targets/ and watermarked_sources/.")
    return found


def numeric_key(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32)


def save_rgb(array: np.ndarray, path: Path) -> None:
    Image.fromarray(
        np.clip(np.rint(array), 0, 255).astype(np.uint8), mode="RGB"
    ).save(path)


# ── Signal processing ─────────────────────────────────────────────────────────

def gaussian_blur(array: np.ndarray, sigma: float) -> np.ndarray:
    img = Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.float32)


def robust_scale(array: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    median = np.median(array, axis=(0, 1), keepdims=True)
    centered = array - median
    mad = 1.4826 * np.median(np.abs(centered), axis=(0, 1), keepdims=True)
    p95 = np.percentile(np.abs(centered), 95, axis=(0, 1), keepdims=True)
    std = np.std(centered, axis=(0, 1), keepdims=True)
    return np.maximum.reduce([mad, p95 / 2.0, std, np.full_like(std, eps)])


def normalize_fingerprint(fp: np.ndarray, strength: float) -> np.ndarray:
    centered = fp - np.mean(fp, axis=(0, 1), keepdims=True)
    normalized = centered / robust_scale(centered) * strength
    return np.clip(normalized, -3.0 * strength, 3.0 * strength).astype(np.float32)


# ── Strategy 1: Residual (Gaussian high-pass) ─────────────────────────────────

def residual_fingerprint(image: np.ndarray, sigmas: Sequence[float]) -> np.ndarray:
    residuals = [image - gaussian_blur(image, s) for s in sigmas]
    res = np.mean(np.stack(residuals, axis=0), axis=0)
    return res - np.median(res, axis=(0, 1), keepdims=True)


def estimate_residual(
    source_paths: Sequence[Path],
    cfg: GroupConfig,
) -> np.ndarray:
    residuals = []
    for path in source_paths:
        img = read_rgb(path)
        res = residual_fingerprint(img, cfg.sigmas)
        if cfg.whiten_sources:
            res = res / robust_scale(res)
        residuals.append(res)

    stack = np.stack(residuals, axis=0)
    if cfg.aggregation == "median":
        fp = np.median(stack, axis=0)
    elif cfg.aggregation == "trimmed_mean":
        trim = max(1, int(round(0.1 * stack.shape[0])))
        fp = np.mean(np.sort(stack, axis=0)[trim:-trim], axis=0)
    else:
        fp = np.mean(stack, axis=0)

    return normalize_fingerprint(fp, cfg.strength)


# ── Strategy 2: Radial bandpass (FFT) ────────────────────────────────────────

def radial_bandpass_mask(h: int, w: int, low: float, high: float) -> np.ndarray:
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fx**2 + fy**2)
    return ((r >= low) & (r <= high)).astype(np.float32)[:, :, None]


def bandpass_residual_fft(image: np.ndarray, low: float, high: float) -> np.ndarray:
    h, w, _ = image.shape
    centered = image - np.mean(image, axis=(0, 1), keepdims=True)
    mask = radial_bandpass_mask(h, w, low, high)
    spectrum = np.fft.fft2(centered, axes=(0, 1))
    filtered = np.fft.ifft2(spectrum * mask, axes=(0, 1)).real.astype(np.float32)
    return filtered - np.mean(filtered, axis=(0, 1), keepdims=True)


def estimate_bandpass(
    source_paths: Sequence[Path],
    cfg: GroupConfig,
) -> np.ndarray:
    residuals = []
    for path in source_paths:
        img = read_rgb(path)
        res = bandpass_residual_fft(img, cfg.freq_low, cfg.freq_high)
        res = res / robust_scale(res)
        residuals.append(res)
    fp = np.median(np.stack(residuals, axis=0), axis=0)
    return normalize_fingerprint(fp, cfg.strength)


# ── Strategy 3: Vertical directional bandpass ─────────────────────────────────

def vertical_bandpass_mask(h: int, w: int, low: float, high: float,
                            horiz_suppress: float) -> np.ndarray:
    """
    Keep only vertical frequency components (high |fy|, low |fx|).
    This targets watermarks that appear as vertical stripes in the FFT.

    - Keeps frequencies where |fy| is in [low, high]
    - Suppresses horizontal components where |fx| > horiz_suppress
    """
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    abs_fy = np.abs(fy)
    abs_fx = np.abs(fx)
    vertical_band = (abs_fy >= low) & (abs_fy <= high)
    horiz_limited = abs_fx <= horiz_suppress
    mask = (vertical_band & horiz_limited).astype(np.float32)[:, :, None]
    return mask


def vertical_residual_fft(image: np.ndarray, low: float, high: float,
                           horiz_suppress: float) -> np.ndarray:
    h, w, _ = image.shape
    centered = image - np.mean(image, axis=(0, 1), keepdims=True)
    mask = vertical_bandpass_mask(h, w, low, high, horiz_suppress)
    spectrum = np.fft.fft2(centered, axes=(0, 1))
    filtered = np.fft.ifft2(spectrum * mask, axes=(0, 1)).real.astype(np.float32)
    return filtered - np.mean(filtered, axis=(0, 1), keepdims=True)


def estimate_vertical(
    source_paths: Sequence[Path],
    cfg: GroupConfig,
) -> np.ndarray:
    residuals = []
    for path in source_paths:
        img = read_rgb(path)
        res = vertical_residual_fft(img, cfg.vert_low, cfg.vert_high, cfg.vert_horiz_suppress)
        res = res / robust_scale(res)
        residuals.append(res)
    fp = np.median(np.stack(residuals, axis=0), axis=0)
    return normalize_fingerprint(fp, cfg.strength)


# ── Dispatch ──────────────────────────────────────────────────────────────────

def estimate_fingerprint(source_paths: Sequence[Path], cfg: GroupConfig) -> np.ndarray:
    if cfg.strategy == "residual":
        return estimate_residual(source_paths, cfg)
    elif cfg.strategy == "bandpass":
        return estimate_bandpass(source_paths, cfg)
    elif cfg.strategy == "vertical":
        return estimate_vertical(source_paths, cfg)
    else:
        raise ValueError(f"Unknown strategy: {cfg.strategy}")


# ── Zip output ────────────────────────────────────────────────────────────────

def validate_and_zip(out_dir: Path, zip_out: Path) -> None:
    names = {p.name for p in out_dir.glob("*.png")}
    if names != EXPECTED_IMAGE_NAMES:
        missing = sorted(EXPECTED_IMAGE_NAMES - names, key=lambda x: int(x[:-4]))
        extra = sorted(names - EXPECTED_IMAGE_NAMES)
        raise RuntimeError(f"Output mismatch. Missing={missing[:5]}, extra={extra[:5]}")
    zip_out.parent.mkdir(parents=True, exist_ok=True)
    if zip_out.exists():
        zip_out.unlink()
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zipf:
        for number in range(1, 201):
            p = out_dir / f"{number}.png"
            zipf.write(p, arcname=p.name)
    with zipfile.ZipFile(zip_out, "r") as zipf:
        if set(zipf.namelist()) != EXPECTED_IMAGE_NAMES:
            raise RuntimeError("ZIP is not a flat 200-image submission.")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_submission(args: argparse.Namespace) -> None:
    dataset = ensure_dataset(args.zip_file, args.dataset_dir)

    if args.out_dir.exists() and not args.only:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # If --only is set, copy existing outputs first so we have a full 200-image dir
    if args.only:
        existing = args.existing_dir
        if existing is None or not existing.exists():
            raise ValueError(
                "--only requires --existing_dir pointing to a full previous output directory."
            )
        print(f"Copying baseline from {existing} ...")
        for p in existing.glob("*.png"):
            dest = args.out_dir / p.name
            if not dest.exists():
                shutil.copy2(p, dest)

    only_set = set(args.only) if args.only else None

    print(f"Dataset root : {dataset.root}")
    print(f"Output dir   : {args.out_dir}")
    print(f"ZIP output   : {args.zip_out}")
    if only_set:
        print(f"Only groups  : {sorted(only_set)}")
    print()

    total = 0
    for wm_name, start, stop in CATEGORIES:
        if only_set and wm_name not in only_set:
            total += stop - start + 1
            continue

        cfg = GROUP_CONFIGS[wm_name]
        source_paths = sorted(
            (dataset.watermarked_sources / wm_name).glob("*.png"), key=numeric_key
        )

        print(f"{wm_name}: strategy={cfg.strategy}, strength={cfg.strength}", end="")
        if cfg.strategy == "residual":
            print(f", sigmas={cfg.sigmas}")
        elif cfg.strategy == "bandpass":
            print(f", freq=[{cfg.freq_low}, {cfg.freq_high}]")
        elif cfg.strategy == "vertical":
            print(f", vert=[{cfg.vert_low}, {cfg.vert_high}], horiz_suppress={cfg.vert_horiz_suppress}")

        fp = estimate_fingerprint(source_paths, cfg)
        print(f"  fingerprint range=[{fp.min():.2f}, {fp.max():.2f}]")

        for number in range(start, stop + 1):
            target = read_rgb(dataset.clean_targets / f"{number}.png")
            if target.shape != fp.shape:
                raise RuntimeError(f"Shape mismatch: target {number}.png")
            save_rgb(target + fp, args.out_dir / f"{number}.png")
            total += 1

    if total != 200:
        raise RuntimeError(f"Expected 200 outputs, produced {total}")

    validate_and_zip(args.out_dir, args.zip_out)
    print(f"\nDone. Created {args.zip_out} with exactly 200 flat PNG files.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-group strategy watermark forgery submission."
    )
    parser.add_argument("--zip_file", type=Path, default=Path("Dataset.zip"))
    parser.add_argument("--dataset_dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--out_dir", type=Path, default=Path("submission_temp_pergroup"))
    parser.add_argument("--zip_out", type=Path, default=Path("submission_pergroup.zip"))
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="WM_N",
        help="Process only these groups (e.g. --only WM_4 WM_7). "
             "Requires --existing_dir for the other 175 images.",
    )
    parser.add_argument(
        "--existing_dir",
        type=Path,
        default=None,
        help="Output directory from a previous full run to copy non-targeted groups from.",
    )
    args = parser.parse_args(argv)

    try:
        build_submission(args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())