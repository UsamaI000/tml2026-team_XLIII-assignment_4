#!/usr/bin/env python3
"""
WM_5 / WM_6 / WM_7 / WM_8 Single-Group Experiments
=====================================================

Same pattern as approach_wm1_only_experiments.py and approach_wm2_only_experiments.py.
Starts from the current best ZIP, patches only the specified WM group, writes a new ZIP.

Covers WM_5 (101-125), WM_6 (126-150), WM_7 (151-175), WM_8 (176-200).

Key insight from WM2/WM3 experiments:
  - WM2 prefers sigma=4 (global revert helped)
  - WM3 prefers sigma=8 (broader residual helped)
  - WM1 sig8 did NOT help
  → Different groups need different sigma values. We test a range for each.

Usage:
  python approach_wm5678_experiments.py \
      --zip_file Dataset.zip --dataset_dir Dataset \
      --base_zip submission_wm3_sig8.zip \
      --wm WM_5 \
      --profile wm5_sig4 \
      --zip_out submission_wm5_sig4.zip \
      --print_psnr

Run all candidate profiles for one group, pick the best, then move to the next group.
Suggested order: WM_5, WM_8, WM_6, WM_7 (roughly by expected recoverability).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

EXPECTED_IMAGE_NAMES = {f"{i}.png" for i in range(1, 201)}

WM_RANGES = {
    "WM_5": range(101, 126),
    "WM_6": range(126, 151),
    "WM_7": range(151, 176),
    "WM_8": range(176, 201),
}


@dataclass(frozen=True)
class WMConfig:
    residual_strength: float
    residual_sigmas: Tuple[float, ...]
    frequency_strength: float = 0.75
    freq_low: float = 0.04
    freq_high: float = 0.20
    aggregation: str = "median"
    channel_mode: str = "rgb"
    perturbation_clip: float = 14.0


# Profiles apply to whichever --wm group is selected.
# Named with the group prefix for clarity in experiment logs.
# Base config for all groups in current best: residual_strength=4, sigma=4, f=0.75, band=0.04-0.20
# (per_wm_consistency_v1 used sigma=1 for WM5/WM6; global hybrid used sigma=4)

PROFILES: Dict[str, WMConfig] = {
    # ── WM_5 profiles ─────────────────────────────────────────────────────────
    # Baseline: global hybrid config (sigma=4)
    "wm5_sig4":        WMConfig(4.0, (4.0,), 0.75, 0.04, 0.20),
    # Smaller sigma: WM5 showed mid-frequency spread, smaller sigma captures that
    "wm5_sig1":        WMConfig(4.0, (1.0,), 0.75, 0.04, 0.20),
    "wm5_sig2":        WMConfig(4.0, (2.0,), 0.75, 0.04, 0.20),
    "wm5_sig1_2":      WMConfig(4.0, (1.0, 2.0), 0.75, 0.04, 0.20),
    # Broader sigma
    "wm5_sig6":        WMConfig(4.0, (6.0,), 0.75, 0.04, 0.20),
    "wm5_sig8":        WMConfig(4.0, (8.0,), 0.75, 0.04, 0.20),
    # Frequency band variants
    "wm5_no_freq":     WMConfig(4.0, (4.0,), 0.00, 0.04, 0.20),
    "wm5_freq_mid":    WMConfig(4.0, (2.0,), 0.75, 0.08, 0.30),
    "wm5_freq_narrow": WMConfig(4.0, (2.0,), 0.75, 0.04, 0.16),
    "wm5_no_freq_sig4": WMConfig(4.0, (4.0,), 0.00, 0.04, 0.20),
    # Strength variants
    "wm5_sig2_s35":    WMConfig(3.5, (2.0,), 0.75, 0.04, 0.20),
    "wm5_sig2_s45":    WMConfig(4.5, (2.0,), 0.75, 0.04, 0.20),

    # ── WM_6 profiles ─────────────────────────────────────────────────────────
    # WM6 showed oval grid pattern (tiled spatial). Frequency analysis showed
    # current perturbation energy in mid/high — possibly wrong band.
    "wm6_sig4":        WMConfig(4.0, (4.0,), 0.75, 0.04, 0.20),
    "wm6_sig1":        WMConfig(4.0, (1.0,), 0.75, 0.04, 0.20),
    "wm6_sig8":        WMConfig(4.0, (8.0,), 0.75, 0.04, 0.20),
    "wm6_no_freq":     WMConfig(4.0, (4.0,), 0.00, 0.04, 0.20),
    "wm6_freq_low":    WMConfig(4.0, (4.0,), 0.75, 0.02, 0.12),
    "wm6_freq_wide":   WMConfig(4.0, (4.0,), 0.75, 0.03, 0.30),
    "wm6_sig6":        WMConfig(4.0, (6.0,), 0.75, 0.04, 0.20),
    "wm6_sig4_s35":    WMConfig(3.5, (4.0,), 0.75, 0.04, 0.20),
    "wm6_sig4_s45":    WMConfig(4.5, (4.0,), 0.75, 0.04, 0.20),
    "wm6_luma":        WMConfig(4.0, (4.0,), 0.75, 0.04, 0.20, channel_mode="luma"),
    "wm6_sig10":       WMConfig(4.0, (10.0,), 0.75, 0.04, 0.20),
    "wm6_sig6_no_freq": WMConfig(4.0, (6.0,),  0.00, 0.04, 0.20),
    "wm6_sig8_no_freq": WMConfig(4.0, (8.0,),  0.00, 0.04, 0.20),
    "wm6_sig10_no_freq": WMConfig(4.0, (10.0,), 0.00, 0.04, 0.20),

    # ── WM_7 profiles ─────────────────────────────────────────────────────────
    # WM7 showed vertical stripe in FFT. very_low dominant (60%).
    # Current approach: sigma=4 with standard hybrid. Try various sigmas.
    "wm7_sig4":        WMConfig(4.0, (4.0,), 0.75, 0.04, 0.20),
    "wm7_sig1":        WMConfig(4.0, (1.0,), 0.75, 0.04, 0.20),
    "wm7_sig6":        WMConfig(4.0, (6.0,), 0.75, 0.04, 0.20),
    "wm7_sig8":        WMConfig(4.0, (8.0,), 0.75, 0.04, 0.20),
    "wm7_sig10":       WMConfig(4.0, (10.0,), 0.75, 0.04, 0.20),
    "wm7_no_freq":     WMConfig(4.0, (4.0,), 0.00, 0.04, 0.20),
    "wm7_freq_low":    WMConfig(4.0, (4.0,), 0.75, 0.02, 0.12),
    "wm7_freq_wide":   WMConfig(4.0, (4.0,), 0.75, 0.03, 0.30),
    "wm7_sig4_s35":    WMConfig(3.5, (4.0,), 0.75, 0.04, 0.20),
    "wm7_sig4_s45":    WMConfig(4.5, (4.0,), 0.75, 0.04, 0.20),
    "wm7_sig8_s35":  WMConfig(3.5, (8.0,),  0.75, 0.04, 0.20),
    "wm7_sig8_s45":  WMConfig(4.5, (8.0,),  0.75, 0.04, 0.20),
    "wm7_sig8_no_freq": WMConfig(4.0, (8.0,), 0.00, 0.04, 0.20),

    # ── WM_8 profiles ─────────────────────────────────────────────────────────
    # WM8 similar to WM1/WM3 (very_low dominant). WM3 benefited from sig8.
    # WM2 benefited from sig4. Try the full range.
    "wm8_sig4":        WMConfig(4.0, (4.0,), 0.75, 0.04, 0.20),
    "wm8_sig6":        WMConfig(4.0, (6.0,), 0.75, 0.04, 0.20),
    "wm8_sig8":        WMConfig(4.0, (8.0,), 0.75, 0.04, 0.20),
    "wm8_sig10":       WMConfig(4.0, (10.0,), 0.75, 0.04, 0.20),
    "wm8_sig2":        WMConfig(4.0, (2.0,), 0.75, 0.04, 0.20),
    "wm8_no_freq":     WMConfig(4.0, (4.0,), 0.00, 0.04, 0.20),
    "wm8_freq_narrow": WMConfig(4.0, (4.0,), 0.75, 0.04, 0.16),
    "wm8_freq_low":    WMConfig(4.0, (4.0,), 0.75, 0.02, 0.12),
    "wm8_sig8_s35":    WMConfig(3.5, (8.0,), 0.75, 0.04, 0.20),
    "wm8_sig8_s45":    WMConfig(4.5, (8.0,), 0.75, 0.04, 0.20),
    "wm8_luma":        WMConfig(4.0, (4.0,), 0.75, 0.04, 0.20, channel_mode="luma"),
    "wm8_sig12":   WMConfig(4.0, (12.0,), 0.75, 0.04, 0.20),
    "wm8_sig10_s35": WMConfig(3.5, (10.0,), 0.75, 0.04, 0.20),
    "wm8_sig10_s45": WMConfig(4.5, (10.0,), 0.75, 0.04, 0.20),
    "wm8_sig10_no_freq": WMConfig(4.0, (10.0,), 0.00, 0.04, 0.20),
}


# ── Dataset helpers ───────────────────────────────────────────────────────────

def numeric_key(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def find_dataset_root(base: Path):
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
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(dataset_dir)
    found = find_dataset_root(dataset_dir.resolve())
    if found is None:
        raise FileNotFoundError("Could not locate dataset folders.")
    return found


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32)


def save_rgb(array: np.ndarray, path: Path) -> None:
    Image.fromarray(
        np.clip(np.rint(array), 0, 255).astype(np.uint8), mode="RGB"
    ).save(path)


def gaussian_blur_rgb(array: np.ndarray, sigma: float) -> np.ndarray:
    img = Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.float32)


def robust_channel_scale(array: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    median = np.median(array, axis=(0, 1), keepdims=True)
    centered = array - median
    mad = 1.4826 * np.median(np.abs(centered), axis=(0, 1), keepdims=True)
    p95 = np.percentile(np.abs(centered), 95, axis=(0, 1), keepdims=True)
    std = np.std(centered, axis=(0, 1), keepdims=True)
    return np.maximum.reduce([mad, p95 / 2.0, std, np.full_like(std, eps)])


def normalize_fingerprint(fp: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return np.zeros_like(fp, dtype=np.float32)
    centered = fp - np.mean(fp, axis=(0, 1), keepdims=True)
    normalized = centered / robust_channel_scale(centered) * float(strength)
    max_abs = max(2.0, 3.0 * float(strength))
    return np.clip(normalized, -max_abs, max_abs).astype(np.float32)


def highpass_residual(image: np.ndarray, sigmas: Sequence[float]) -> np.ndarray:
    return np.mean(
        np.stack([image - gaussian_blur_rgb(image, s) for s in sigmas], axis=0), axis=0
    )


def aggregate(stack: np.ndarray, method: str) -> np.ndarray:
    if method == "median":
        return np.median(stack, axis=0)
    if method == "mean":
        return np.mean(stack, axis=0)
    if method == "trimmed_mean":
        if stack.shape[0] < 5:
            return np.mean(stack, axis=0)
        trim = max(1, int(round(0.1 * stack.shape[0])))
        return np.mean(np.sort(stack, axis=0)[trim:-trim], axis=0)
    raise ValueError(f"Unknown aggregation: {method}")


def radial_bandpass_mask(h: int, w: int, low: float, high: float) -> np.ndarray:
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fx**2 + fy**2)
    return ((r >= low) & (r <= high)).astype(np.float32)[:, :, None]


def estimate_fingerprint(
    source_paths: Sequence[Path], cfg: WMConfig
) -> np.ndarray:
    # Residual component
    res_list = []
    for p in source_paths:
        img = read_rgb(p)
        res = highpass_residual(img, cfg.residual_sigmas)
        res = res - np.median(res, axis=(0, 1), keepdims=True)
        res_list.append(res)
    residual_fp = normalize_fingerprint(
        aggregate(np.stack(res_list, axis=0), cfg.aggregation),
        cfg.residual_strength,
    )

    # Frequency component
    if cfg.frequency_strength <= 0:
        freq_fp = np.zeros_like(residual_fp)
    else:
        freq_list = []
        for p in source_paths:
            img = read_rgb(p)
            h, w, _ = img.shape
            centered = img - np.mean(img, axis=(0, 1), keepdims=True)
            spectrum = np.fft.fft2(centered, axes=(0, 1))
            filtered = np.fft.ifft2(
                spectrum * radial_bandpass_mask(h, w, cfg.freq_low, cfg.freq_high),
                axes=(0, 1)
            ).real.astype(np.float32)
            filtered = filtered - np.mean(filtered, axis=(0, 1), keepdims=True)
            filtered = filtered / robust_channel_scale(filtered)
            freq_list.append(filtered)
        freq_fp = normalize_fingerprint(
            aggregate(np.stack(freq_list, axis=0), cfg.aggregation),
            cfg.frequency_strength,
        )

    perturbation = residual_fp + freq_fp

    if cfg.channel_mode == "luma":
        y = (0.299 * perturbation[:, :, 0:1]
             + 0.587 * perturbation[:, :, 1:2]
             + 0.114 * perturbation[:, :, 2:3])
        perturbation = np.repeat(y, 3, axis=2).astype(np.float32)

    return np.clip(perturbation, -cfg.perturbation_clip, cfg.perturbation_clip).astype(np.float32)


# ── ZIP helpers ───────────────────────────────────────────────────────────────

def extract_base_zip(base_zip: Path, out_dir: Path) -> None:
    if not base_zip.is_file():
        raise FileNotFoundError(f"Missing base ZIP: {base_zip}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip, "r") as zf:
        names = set(zf.namelist())
        if names != EXPECTED_IMAGE_NAMES:
            raise RuntimeError("Base ZIP must be flat 1.png...200.png")
        zf.extractall(out_dir)


def validate_and_zip(out_dir: Path, zip_out: Path) -> None:
    names = {p.name for p in out_dir.glob("*.png")}
    if names != EXPECTED_IMAGE_NAMES:
        missing = sorted(EXPECTED_IMAGE_NAMES - names, key=lambda x: int(x[:-4]))
        extra = sorted(names - EXPECTED_IMAGE_NAMES)
        raise RuntimeError(f"Output mismatch. Missing={missing[:5]}, extra={extra[:5]}")
    if zip_out.exists():
        zip_out.unlink()
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(1, 201):
            p = out_dir / f"{i}.png"
            zf.write(p, arcname=p.name)
    with zipfile.ZipFile(zip_out, "r") as zf:
        if set(zf.namelist()) != EXPECTED_IMAGE_NAMES:
            raise RuntimeError("ZIP validation failed.")


def group_psnr(clean_dir: Path, out_dir: Path, img_range: range) -> float:
    vals = []
    for i in img_range:
        clean = read_rgb(clean_dir / f"{i}.png")
        forged = read_rgb(out_dir / f"{i}.png")
        mse = np.mean((clean - forged) ** 2)
        vals.append(float("inf") if mse < 1e-12 else 20 * np.log10(255.0 / np.sqrt(mse)))
    return float(np.mean(vals))


# ── Main ──────────────────────────────────────────────────────────────────────

def build(args: argparse.Namespace) -> None:
    root = ensure_dataset(args.zip_file, args.dataset_dir)
    clean_dir = root / "clean_targets"
    source_dir = root / "watermarked_sources" / args.wm

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    img_range = WM_RANGES[args.wm]
    cfg = PROFILES[args.profile]
    source_paths = sorted(source_dir.glob("*.png"), key=numeric_key)

    if len(source_paths) != 25:
        raise RuntimeError(
            f"Expected 25 source images for {args.wm}, found {len(source_paths)}"
        )

    print(f"Group        : {args.wm} (images {img_range.start}-{img_range.stop - 1})")
    print(f"Base ZIP     : {args.base_zip}")
    print(f"Profile      : {args.profile} → {cfg}")
    print(f"Output ZIP   : {args.zip_out}")

    extract_base_zip(args.base_zip, args.out_dir)

    fp = estimate_fingerprint(source_paths, cfg)
    print(f"Fingerprint  : range=[{fp.min():.2f}, {fp.max():.2f}], std={fp.std():.3f}")

    for i in img_range:
        target = read_rgb(clean_dir / f"{i}.png")
        if target.shape != fp.shape:
            raise RuntimeError(f"Shape mismatch {i}.png")
        save_rgb(target + fp, args.out_dir / f"{i}.png")

    validate_and_zip(args.out_dir, args.zip_out)

    if args.print_psnr:
        psnr = group_psnr(clean_dir, args.out_dir, img_range)
        print(f"PSNR {args.wm}    : {psnr:.2f} dB")

    print(f"Done. Created {args.zip_out}. Only {args.wm} images were regenerated.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Single-group experiments for WM_5, WM_6, WM_7, WM_8."
    )
    parser.add_argument("--zip_file", type=Path, default=Path("Dataset.zip"))
    parser.add_argument("--dataset_dir", type=Path, default=Path("Dataset"))
    parser.add_argument(
        "--base_zip", type=Path,
        default=Path("submission_wm3_sig8.zip"),
        help="Current best ZIP. Only the selected --wm group is replaced.",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("submission_temp_wm5678"))
    parser.add_argument("--zip_out", type=Path, default=Path("submission_wm5678.zip"))
    parser.add_argument(
        "--wm", choices=list(WM_RANGES.keys()), required=True,
        help="Which WM group to experiment on.",
    )
    parser.add_argument(
        "--profile", choices=sorted(PROFILES.keys()), required=True,
        help="Which config profile to apply.",
    )
    parser.add_argument("--print_psnr", action="store_true")
    args = parser.parse_args(argv)

    try:
        build(args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
