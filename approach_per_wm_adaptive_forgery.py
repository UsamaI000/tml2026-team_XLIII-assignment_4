#!/usr/bin/env python3
"""
Per-WM Adaptive Watermark Forgery
=================================

Creates a flat 200-image submission ZIP for the watermark-forgery assignment.
This file does not modify task_template.py or submission.py.

Why this approach exists
------------------------
The 8 watermark folders can correspond to 8 different watermarking methods.
Therefore one global extraction recipe can be suboptimal. This script treats
WM_1 ... WM_8 separately and allows each class to use a different residual
scale, residual strength, frequency strength, and frequency band.

The default profile `per_wm_consistency_v1` is based on source-set consistency:
WM_4, WM_5, and WM_6 showed stronger fine-scale agreement, while WM_1, WM_2,
WM_3, WM_7, and WM_8 looked more useful in broader residual bands.

Output format is validated: exactly 1.png ... 200.png at the root of the ZIP.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


@dataclass(frozen=True)
class WMConfig:
    residual_strength: float
    residual_sigmas: Tuple[float, ...]
    frequency_strength: float = 0.75
    freq_low: float = 0.04
    freq_high: float = 0.20
    aggregation: str = "median"  # median, mean, trimmed_mean
    channel_mode: str = "rgb"    # rgb, luma
    perturbation_clip: float = 14.0


# Current best global profile from your experiments: hybrid r4 + f0.75 + band 0.04-0.20.
GLOBAL_HYBRID_BEST: Dict[str, WMConfig] = {
    wm: WMConfig(
        residual_strength=4.0,
        residual_sigmas=(4.0,),
        frequency_strength=0.75,
        freq_low=0.04,
        freq_high=0.20,
        perturbation_clip=14.0,
    )
    for wm, _, _ in CATEGORIES
}


# New profile: each WM uses its own residual scale. This is intentionally moderate.
# It keeps the hybrid component that already improved the leaderboard, but changes
# the residual extractor per class.
PER_WM_CONSISTENCY_V1: Dict[str, WMConfig] = {
    # Broader residual classes. Keep close to known winner, with slightly broader extraction.
    "WM_1": WMConfig(4.0, (6.0,), 0.75, 0.04, 0.20, perturbation_clip=14.0),
    "WM_2": WMConfig(4.0, (6.0,), 0.75, 0.04, 0.20, perturbation_clip=14.0),
    "WM_3": WMConfig(4.0, (6.0,), 0.75, 0.04, 0.20, perturbation_clip=14.0),

    # Fine-scale class: consistency proxy was highest at very small sigma.
    "WM_4": WMConfig(3.5, (0.5, 1.0), 0.50, 0.08, 0.30, perturbation_clip=12.0),

    # Fine/high-frequency classes: strong source agreement near sigma=1.
    "WM_5": WMConfig(4.0, (1.0,), 0.50, 0.06, 0.25, perturbation_clip=14.0),
    "WM_6": WMConfig(4.0, (1.0,), 0.50, 0.06, 0.25, perturbation_clip=14.0),

    # Larger-resolution classes: keep broad residual signal and current useful band.
    "WM_7": WMConfig(4.0, (6.0,), 0.75, 0.04, 0.20, perturbation_clip=14.0),
    "WM_8": WMConfig(4.0, (6.0,), 0.75, 0.04, 0.20, perturbation_clip=14.0),
}


# Conservative variant: only changes the strongest suspected fine-scale classes.
PER_WM_CONSISTENCY_V2: Dict[str, WMConfig] = dict(GLOBAL_HYBRID_BEST)
PER_WM_CONSISTENCY_V2.update(
    {
        "WM_4": WMConfig(3.5, (0.5, 1.0), 0.50, 0.08, 0.30, perturbation_clip=12.0),
        "WM_5": WMConfig(4.0, (1.0,), 0.50, 0.06, 0.25, perturbation_clip=14.0),
        "WM_6": WMConfig(4.0, (1.0,), 0.50, 0.06, 0.25, perturbation_clip=14.0),
    }
)


# Another conservative variant: only WM_5 and WM_6 use sigma=1. Useful if WM_4 fine-scale hurts.
PER_WM_CONSISTENCY_V3: Dict[str, WMConfig] = dict(GLOBAL_HYBRID_BEST)
PER_WM_CONSISTENCY_V3.update(
    {
        "WM_5": WMConfig(4.0, (1.0,), 0.50, 0.06, 0.25, perturbation_clip=14.0),
        "WM_6": WMConfig(4.0, (1.0,), 0.50, 0.06, 0.25, perturbation_clip=14.0),
    }
)


PROFILES: Dict[str, Dict[str, WMConfig]] = {
    "global_hybrid_best": GLOBAL_HYBRID_BEST,
    "per_wm_consistency_v1": PER_WM_CONSISTENCY_V1,
    "per_wm_consistency_v2": PER_WM_CONSISTENCY_V2,
    "per_wm_consistency_v3": PER_WM_CONSISTENCY_V3,
}


def parse_float_list(text: str) -> Tuple[float, ...]:
    values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def numeric_key(path: Path) -> int:
    number_text = path.stem.split("_")[-1]
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
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius=float(sigma))), dtype=np.float32)


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


def estimate_residual_fingerprint(
    source_paths: Sequence[Path], sigmas: Sequence[float], strength: float, aggregation: str
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
        residuals.append(residual.astype(np.float32))
    fp = aggregate_stack(np.stack(residuals, axis=0), aggregation)
    return normalize_fingerprint(fp, strength=strength)


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
    filtered = np.fft.ifft2(
        spectrum * radial_bandpass_mask(height, width, low=low, high=high), axes=(0, 1)
    ).real.astype(np.float32)
    return filtered


def estimate_frequency_fingerprint(
    source_paths: Sequence[Path], low: float, high: float, strength: float, aggregation: str
) -> np.ndarray:
    if strength <= 0:
        # Use the first source image to infer shape.
        return np.zeros_like(read_rgb(source_paths[0]), dtype=np.float32)

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
        residuals.append(residual.astype(np.float32))
    fp = aggregate_stack(np.stack(residuals, axis=0), aggregation)
    return normalize_fingerprint(fp, strength=strength)


def apply_channel_mode(perturbation: np.ndarray, channel_mode: str) -> np.ndarray:
    if channel_mode == "rgb":
        return perturbation
    if channel_mode == "luma":
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
            raise FileNotFoundError(f"Expected {expected_count} PNGs in {wm_dir}, found {found_count}")


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
        psnrs.append(float("inf") if mse <= 1e-12 else 20.0 * np.log10(255.0 / np.sqrt(mse)))
    return float(np.mean(psnrs))


def load_config_from_json(path: Path) -> Dict[str, WMConfig]:
    data = json.loads(path.read_text())
    cfg = dict(GLOBAL_HYBRID_BEST)
    for wm_name, values in data.items():
        if wm_name not in cfg:
            raise ValueError(f"Unknown WM name in JSON config: {wm_name}")
        current = asdict(cfg[wm_name])
        current.update(values)
        current["residual_sigmas"] = tuple(current["residual_sigmas"])
        cfg[wm_name] = WMConfig(**current)
    return cfg


def print_profile(config: Dict[str, WMConfig]) -> None:
    print("Per-WM configuration:")
    for wm_name, _, _ in CATEGORIES:
        c = config[wm_name]
        print(
            f"  {wm_name}: residual_strength={c.residual_strength}, "
            f"sigmas={c.residual_sigmas}, freq_strength={c.frequency_strength}, "
            f"band=({c.freq_low},{c.freq_high}), agg={c.aggregation}, "
            f"channel={c.channel_mode}, clip={c.perturbation_clip}"
        )


def build_submission(args: argparse.Namespace) -> None:
    dataset = ensure_dataset(args.zip_file, args.dataset_dir)
    validate_dataset(dataset)

    if args.config_json:
        profile = load_config_from_json(args.config_json)
        profile_name = f"json:{args.config_json}"
    else:
        profile = PROFILES[args.profile]
        profile_name = args.profile

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root: {dataset.root}")
    print(f"Profile: {profile_name}")
    print(f"Output directory: {args.out_dir}")
    print(f"ZIP output: {args.zip_out}")
    print_profile(profile)

    total = 0
    for wm_name, start, stop in CATEGORIES:
        c = profile[wm_name]
        source_dir = dataset.watermarked_sources / wm_name
        source_paths = sorted(source_dir.glob("*.png"), key=numeric_key)

        residual_fp = estimate_residual_fingerprint(
            source_paths=source_paths,
            sigmas=c.residual_sigmas,
            strength=c.residual_strength,
            aggregation=c.aggregation,
        )
        frequency_fp = estimate_frequency_fingerprint(
            source_paths=source_paths,
            low=c.freq_low,
            high=c.freq_high,
            strength=c.frequency_strength,
            aggregation=c.aggregation,
        )

        perturbation = apply_channel_mode(residual_fp + frequency_fp, c.channel_mode)
        perturbation = np.clip(perturbation, -c.perturbation_clip, c.perturbation_clip).astype(np.float32)

        print(
            f"{wm_name}: residual_range=[{residual_fp.min():.2f},{residual_fp.max():.2f}], "
            f"freq_range=[{frequency_fp.min():.2f},{frequency_fp.max():.2f}], "
            f"combined_range=[{perturbation.min():.2f},{perturbation.max():.2f}]"
        )

        for number in range(start, stop + 1):
            target = read_rgb(dataset.clean_targets / f"{number}.png")
            if target.shape != perturbation.shape:
                raise RuntimeError(
                    f"Shape mismatch for {number}.png: target={target.shape}, perturbation={perturbation.shape}"
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
    parser = argparse.ArgumentParser(description="Per-WM adaptive watermark-forgery submission builder.")
    parser.add_argument("--zip_file", type=Path, default=Path("Dataset.zip"))
    parser.add_argument("--dataset_dir", type=Path, default=Path("_dataset_extracted_per_wm"))
    parser.add_argument("--out_dir", type=Path, default=Path("submission_temp_per_wm"))
    parser.add_argument("--zip_out", type=Path, default=Path("submission_per_wm.zip"))
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES.keys()),
        default="per_wm_consistency_v1",
        help="Built-in per-WM profile to use unless --config_json is provided.",
    )
    parser.add_argument(
        "--config_json",
        type=Path,
        default=None,
        help="Optional JSON file with per-WM overrides. Keys: WM_1 ... WM_8.",
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
