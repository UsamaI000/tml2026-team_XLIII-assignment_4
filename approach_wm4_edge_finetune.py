#!/usr/bin/env python3
"""
WM_4 Edge/Grid Artifact Fine-Tuning
===================================

This script starts from an existing best submission ZIP and modifies only the
WM_4 target batch (76.png ... 100.png). It is intended for the watermark-forgery
assignment where WM_4 source images visibly contain a repeated black/white edge
or corner grid artifact.

It does NOT modify task_template.py or submission.py.

Idea:
    1. Keep the current best forged images for all classes.
    2. Estimate the stable WM_4 edge/grid artifact from the 25 WM_4 source images.
    3. Add only this edge artifact to targets 76.png ... 100.png with a small,
       tunable strength.

Output ZIP format is validated: exactly 1.png ... 200.png at the archive root.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

EXPECTED_IMAGE_NAMES = {f"{i}.png" for i in range(1, 201)}
WM4_START, WM4_STOP = 76, 100


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    clean_targets: Path
    watermarked_sources: Path


def numeric_key(path: Path) -> int:
    txt = path.stem.split("_")[-1]
    return int(txt)


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
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(dataset_dir)
    found = find_dataset_root(dataset_dir.resolve())
    if found is None:
        raise FileNotFoundError("Could not locate clean_targets/ and watermarked_sources/.")
    return found


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32)


def save_rgb(arr: np.ndarray, path: Path) -> None:
    out = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    Image.fromarray(out, mode="RGB").save(path)


def gaussian_blur_rgb(arr: np.ndarray, sigma: float) -> np.ndarray:
    img = Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius=float(sigma))), dtype=np.float32)


def robust_channel_scale(arr: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    median = np.median(arr, axis=(0, 1), keepdims=True)
    centered = arr - median
    mad = 1.4826 * np.median(np.abs(centered), axis=(0, 1), keepdims=True)
    p95 = np.percentile(np.abs(centered), 95, axis=(0, 1), keepdims=True)
    std = np.std(centered, axis=(0, 1), keepdims=True)
    return np.maximum.reduce([mad, p95 / 2.0, std, np.full_like(std, eps)])


def parse_sigmas(text: str) -> Tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not vals:
        raise argparse.ArgumentTypeError("Need at least one sigma.")
    return vals


def highpass(arr: np.ndarray, sigmas: Sequence[float]) -> np.ndarray:
    parts = []
    for s in sigmas:
        parts.append(arr - gaussian_blur_rgb(arr, s))
    return np.mean(np.stack(parts, axis=0), axis=0).astype(np.float32)


def make_edge_mask(h: int, w: int, width: int, mode: str, softness: int) -> np.ndarray:
    """Return mask shape (h, w, 1)."""
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]

    left = x < width
    right = x >= (w - width)
    top = y < width
    bottom = y >= (h - width)

    if mode == "all_edges":
        mask2d = left | right | top | bottom
    elif mode == "sides_bottom":
        mask2d = left | right | bottom
    elif mode == "bottom_only":
        mask2d = bottom
    elif mode == "corners":
        corner_h = max(width * 3, 32)
        corner_w = max(width * 3, 32)
        bl = (x < corner_w) & (y >= h - corner_h)
        br = (x >= w - corner_w) & (y >= h - corner_h)
        tl = (x < corner_w) & (y < corner_h)
        tr = (x >= w - corner_w) & (y < corner_h)
        mask2d = bl | br | tl | tr
    elif mode == "bottom_corners":
        corner_h = max(width * 3, 32)
        corner_w = max(width * 3, 32)
        bl = (x < corner_w) & (y >= h - corner_h)
        br = (x >= w - corner_w) & (y >= h - corner_h)
        mask2d = bl | br
    else:
        raise ValueError(f"Unknown edge mask mode: {mode}")

    mask = mask2d.astype(np.float32)
    if softness > 0:
        img = Image.fromarray(np.uint8(mask * 255), mode="L")
        img = img.filter(ImageFilter.GaussianBlur(radius=float(softness)))
        mask = np.asarray(img, dtype=np.float32) / 255.0
    return mask[:, :, None].astype(np.float32)


def estimate_wm4_edge_artifact(
    wm4_dir: Path,
    residual_sigmas: Sequence[float],
    strength: float,
    edge_width: int,
    mask_mode: str,
    softness: int,
    aggregation: str,
) -> np.ndarray:
    source_paths = sorted(wm4_dir.glob("*.png"), key=numeric_key)
    if len(source_paths) != 25:
        raise FileNotFoundError(f"Expected 25 WM_4 source images in {wm4_dir}, got {len(source_paths)}")

    residuals = []
    expected_shape = None
    for p in source_paths:
        img = read_rgb(p)
        if expected_shape is None:
            expected_shape = img.shape
        elif img.shape != expected_shape:
            raise RuntimeError(f"WM_4 source shape mismatch: {p}")
        res = highpass(img, residual_sigmas)
        res = res - np.median(res, axis=(0, 1), keepdims=True)
        residuals.append(res)

    stack = np.stack(residuals, axis=0)
    if aggregation == "median":
        fp = np.median(stack, axis=0)
    elif aggregation == "mean":
        fp = np.mean(stack, axis=0)
    else:
        raise ValueError("aggregation must be median or mean")

    fp = fp - np.mean(fp, axis=(0, 1), keepdims=True)
    fp = fp / robust_channel_scale(fp) * float(strength)
    # Keep the grid artifact bounded; stronger variants should use strength, not unlimited clipping.
    fp = np.clip(fp, -max(2.0, 3.0 * strength), max(2.0, 3.0 * strength)).astype(np.float32)

    h, w, _ = fp.shape
    mask = make_edge_mask(h, w, width=edge_width, mode=mask_mode, softness=softness)
    return fp * mask


def extract_base_zip(base_zip: Path, out_dir: Path) -> None:
    if not base_zip.is_file():
        raise FileNotFoundError(f"Missing base submission zip: {base_zip}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip, "r") as z:
        names = set(z.namelist())
        if names != EXPECTED_IMAGE_NAMES:
            raise RuntimeError("Base ZIP is not exactly flat 1.png ... 200.png")
        z.extractall(out_dir)


def validate_and_zip(out_dir: Path, zip_out: Path) -> None:
    names = {p.name for p in out_dir.glob("*.png")}
    if names != EXPECTED_IMAGE_NAMES:
        missing = sorted(EXPECTED_IMAGE_NAMES - names, key=lambda s: int(s[:-4]))
        extra = sorted(names - EXPECTED_IMAGE_NAMES)
        raise RuntimeError(f"Output mismatch. Missing={missing[:10]}, extra={extra[:10]}")
    zip_out.parent.mkdir(parents=True, exist_ok=True)
    if zip_out.exists():
        zip_out.unlink()
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
        for i in range(1, 201):
            p = out_dir / f"{i}.png"
            z.write(p, arcname=p.name)
    with zipfile.ZipFile(zip_out, "r") as z:
        if set(z.namelist()) != EXPECTED_IMAGE_NAMES:
            raise RuntimeError("ZIP validation failed")


def mean_psnr(clean_dir: Path, out_dir: Path) -> float:
    vals = []
    for i in range(1, 201):
        clean = read_rgb(clean_dir / f"{i}.png")
        forged = read_rgb(out_dir / f"{i}.png")
        mse = np.mean((clean - forged) ** 2)
        vals.append(float("inf") if mse <= 1e-12 else float(20.0 * np.log10(255.0 / np.sqrt(mse))))
    return float(np.mean(vals))


def build_submission(args: argparse.Namespace) -> None:
    dataset = ensure_dataset(args.zip_file, args.dataset_dir)
    extract_base_zip(args.base_zip, args.out_dir)

    artifact = estimate_wm4_edge_artifact(
        wm4_dir=dataset.watermarked_sources / "WM_4",
        residual_sigmas=args.residual_sigmas,
        strength=args.strength,
        edge_width=args.edge_width,
        mask_mode=args.mask_mode,
        softness=args.softness,
        aggregation=args.aggregation,
    )
    print(
        f"WM_4 artifact: shape={artifact.shape}, range=[{artifact.min():.2f},{artifact.max():.2f}], "
        f"strength={args.strength}, edge_width={args.edge_width}, mode={args.mask_mode}, sigmas={args.residual_sigmas}"
    )

    for i in range(WM4_START, WM4_STOP + 1):
        p = args.out_dir / f"{i}.png"
        base = read_rgb(p)
        if base.shape != artifact.shape:
            raise RuntimeError(f"Shape mismatch for {p.name}: {base.shape} vs {artifact.shape}")
        save_rgb(base + artifact, p)

    validate_and_zip(args.out_dir, args.zip_out)
    if args.print_psnr:
        print(f"Mean PSNR vs clean targets: {mean_psnr(dataset.clean_targets, args.out_dir):.2f} dB")
    print(f"Done. Created {args.zip_out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine tune WM_4 edge/grid artifact patch on top of a base ZIP.")
    p.add_argument("--zip_file", type=Path, default=Path("Dataset.zip"))
    p.add_argument("--dataset_dir", type=Path, default=Path("_dataset_extracted_wm4_edge"))
    p.add_argument("--base_zip", type=Path, default=Path("submission_per_wm_consistency_v1.zip"))
    p.add_argument("--out_dir", type=Path, default=Path("submission_temp_wm4_edge"))
    p.add_argument("--zip_out", type=Path, default=Path("submission_wm4_edge_finetuned.zip"))
    p.add_argument("--strength", type=float, default=0.45)
    p.add_argument("--residual_sigmas", type=parse_sigmas, default=(0.5, 1.0))
    p.add_argument("--edge_width", type=int, default=18)
    p.add_argument("--mask_mode", choices=("all_edges", "sides_bottom", "bottom_only", "corners", "bottom_corners"), default="sides_bottom")
    p.add_argument("--softness", type=int, default=2)
    p.add_argument("--aggregation", choices=("median", "mean"), default="median")
    p.add_argument("--print_psnr", action="store_true")
    return p


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
