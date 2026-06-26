#!/usr/bin/env python3
"""
WM Family Diagnostic
====================

For each WM family, this script:
1. Estimates the fingerprint (same logic as the residual script)
2. Saves the fingerprint as a visible PNG so you can inspect it
3. Computes mean pairwise cosine similarity between per-source residuals
4. Prints a per-group classification: FIXED or ADAPTIVE

Run this BEFORE deciding which strategy to use per group.

Usage:
    python diagnose_wm_families.py --dataset_dir Dataset

Output:
    diag_fingerprints/WM_1_fingerprint.png  ... (one per group, scaled for visibility)
    diag_fingerprints/WM_1_residuals.png    ... (montage of first 5 per-source residuals)
    A printed table with pairwise cosine similarity and classification per group.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont

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


# ── I/O helpers ──────────────────────────────────────────────────────────────

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
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(dataset_dir)
    found = find_dataset_root(dataset_dir.resolve())
    if found is None:
        raise FileNotFoundError("Could not locate clean_targets/ and watermarked_sources/.")
    return found


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32)


def numeric_key(path: Path) -> int:
    return int(path.stem.split("_")[-1])


# ── Signal processing ─────────────────────────────────────────────────────────

def gaussian_blur(array: np.ndarray, sigma: float) -> np.ndarray:
    img = Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.float32)


def highpass_residual(array: np.ndarray, sigmas: Sequence[float] = (1.0, 2.0, 4.0)) -> np.ndarray:
    return np.mean(
        np.stack([array - gaussian_blur(array, s) for s in sigmas], axis=0), axis=0
    )


def bandpass_residual(array: np.ndarray, low: float = 0.06, high: float = 0.40) -> np.ndarray:
    h, w, _ = array.shape
    centered = array - np.mean(array, axis=(0, 1), keepdims=True)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    mask = ((np.sqrt(fx**2 + fy**2) >= low) & (np.sqrt(fx**2 + fy**2) <= high)).astype(np.float32)[:, :, None]
    spectrum = np.fft.fft2(centered, axes=(0, 1))
    return np.fft.ifft2(spectrum * mask, axes=(0, 1)).real.astype(np.float32)


def normalize_for_display(array: np.ndarray) -> np.ndarray:
    """Scale an arbitrary float array to [0, 255] uint8 for visual inspection."""
    a_min, a_max = array.min(), array.max()
    if a_max - a_min < 1e-8:
        return np.full_like(array, 128, dtype=np.uint8)
    scaled = (array - a_min) / (a_max - a_min) * 255.0
    return scaled.astype(np.uint8)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def mean_pairwise_cosine(residuals: List[np.ndarray]) -> float:
    """Average cosine similarity over all unique pairs."""
    n = len(residuals)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine_similarity(residuals[i], residuals[j]))
    return float(np.mean(sims)) if sims else 0.0


def fft_peak_ratio(fingerprint: np.ndarray) -> float:
    """
    Ratio of energy in top-1% of FFT bins to total energy.
    High ratio → concentrated frequency peaks → classical/spread-spectrum scheme.
    Low ratio → diffuse energy → neural/learned scheme.
    """
    gray = np.mean(fingerprint, axis=2)
    spectrum = np.abs(np.fft.fft2(gray))
    spectrum[0, 0] = 0  # ignore DC
    flat = spectrum.ravel()
    threshold = np.percentile(flat, 99)
    top_energy = np.sum(flat[flat >= threshold] ** 2)
    total_energy = np.sum(flat ** 2)
    if total_energy < 1e-12:
        return 0.0
    return float(top_energy / total_energy)


# ── Visualisation helpers ─────────────────────────────────────────────────────

def save_fingerprint_image(fingerprint: np.ndarray, path: Path) -> None:
    """Save fingerprint scaled to [0,255] for visual inspection."""
    Image.fromarray(normalize_for_display(fingerprint), mode="RGB").save(path)


def save_residual_montage(residuals: List[np.ndarray], path: Path, n: int = 5) -> None:
    """Save a side-by-side montage of the first n per-source residuals."""
    to_show = residuals[:n]
    imgs = [Image.fromarray(normalize_for_display(r), mode="RGB") for r in to_show]
    w, h = imgs[0].size
    canvas = Image.new("RGB", (w * len(imgs), h), color=(200, 200, 200))
    for i, img in enumerate(imgs):
        canvas.paste(img, (i * w, 0))
    canvas.save(path)


def save_fft_magnitude(fingerprint: np.ndarray, path: Path) -> None:
    """Save log-magnitude FFT spectrum of the fingerprint (grayscale)."""
    gray = np.mean(fingerprint, axis=2)
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(gray)))
    log_spec = np.log1p(spectrum)
    log_spec_uint8 = normalize_for_display(log_spec[:, :, None])[:, :, 0]
    Image.fromarray(log_spec_uint8, mode="L").save(path)


# ── Per-group analysis ────────────────────────────────────────────────────────

def analyse_group(
    wm_name: str,
    source_paths: List[Path],
    out_dir: Path,
    sigmas: Sequence[float],
) -> dict:
    residuals_hp = []   # high-pass residuals
    residuals_bp = []   # band-pass residuals

    expected_shape = None
    for path in source_paths:
        img = read_rgb(path)
        if expected_shape is None:
            expected_shape = img.shape
        hp = highpass_residual(img, sigmas)
        hp = hp - np.median(hp, axis=(0, 1), keepdims=True)
        residuals_hp.append(hp)

        bp = bandpass_residual(img)
        bp = bp - np.mean(bp, axis=(0, 1), keepdims=True)
        residuals_bp.append(bp)

    # Aggregate fingerprints
    fp_hp = np.median(np.stack(residuals_hp, axis=0), axis=0)
    fp_bp = np.median(np.stack(residuals_bp, axis=0), axis=0)

    # Metrics
    sim_hp = mean_pairwise_cosine(residuals_hp)
    sim_bp = mean_pairwise_cosine(residuals_bp)
    peak_ratio = fft_peak_ratio(fp_hp)
    fp_std = float(np.std(fp_hp))

    # Fingerprint energy relative to per-source residual energy.
    # High ratio = averaging reinforced signal (fixed pattern).
    # Low ratio = averaging cancelled signal (adaptive).
    mean_src_energy = float(np.mean([np.std(r) for r in residuals_hp]))
    fp_energy_ratio = fp_std / (mean_src_energy + 1e-8)

    # Classification heuristic (tune thresholds after visual inspection)
    is_fixed = (sim_hp > 0.05) or (fp_energy_ratio > 0.25) or (peak_ratio > 0.15)
    classification = "FIXED   " if is_fixed else "ADAPTIVE"

    # Save visuals
    save_fingerprint_image(fp_hp, out_dir / f"{wm_name}_fingerprint_highpass.png")
    save_fingerprint_image(fp_bp, out_dir / f"{wm_name}_fingerprint_bandpass.png")
    save_residual_montage(residuals_hp, out_dir / f"{wm_name}_residuals_montage.png")
    save_fft_magnitude(fp_hp, out_dir / f"{wm_name}_fft_magnitude.png")

    return {
        "wm_name": wm_name,
        "classification": classification,
        "sim_hp": sim_hp,
        "sim_bp": sim_bp,
        "peak_ratio": peak_ratio,
        "fp_energy_ratio": fp_energy_ratio,
        "fp_std": fp_std,
        "mean_src_energy": mean_src_energy,
        "n_sources": len(source_paths),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose WM families: fixed vs adaptive.")
    parser.add_argument("--zip_file", type=Path, default=Path("Dataset.zip"))
    parser.add_argument("--dataset_dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--out_dir", type=Path, default=Path("diag_fingerprints"))
    parser.add_argument(
        "--sigmas",
        type=lambda s: [float(x) for x in s.split(",")],
        default=[1.0, 2.0, 4.0],
        help="Comma-separated Gaussian sigmas for high-pass residual.",
    )
    args = parser.parse_args()

    root = ensure_dataset(args.zip_file, args.dataset_dir)
    source_root = root / "watermarked_sources"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset root : {root}")
    print(f"Output dir   : {args.out_dir}")
    print(f"Sigmas       : {args.sigmas}\n")

    results = []
    for wm_name, start, stop in CATEGORIES:
        source_paths = sorted((source_root / wm_name).glob("*.png"), key=numeric_key)
        print(f"Analysing {wm_name} ({len(source_paths)} sources)...", flush=True)
        result = analyse_group(wm_name, source_paths, args.out_dir, args.sigmas)
        results.append(result)

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Group':<8} {'Class':<10} {'sim_hp':>8} {'sim_bp':>8} "
          f"{'peak_r':>8} {'fp/src':>8} {'fp_std':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['wm_name']:<8} {r['classification']:<10} "
            f"{r['sim_hp']:>8.4f} {r['sim_bp']:>8.4f} "
            f"{r['peak_ratio']:>8.4f} {r['fp_energy_ratio']:>8.4f} "
            f"{r['fp_std']:>8.4f}"
        )
    print("=" * 80)
    print("\nColumn guide:")
    print("  sim_hp      : mean pairwise cosine similarity of high-pass residuals")
    print("                > 0.05 strongly suggests a fixed watermark pattern")
    print("  sim_bp      : same for band-pass residuals")
    print("  peak_r      : fraction of FFT energy in top-1% bins")
    print("                > 0.15 suggests periodic/spread-spectrum scheme")
    print("  fp/src      : fingerprint std / mean per-source residual std")
    print("                > 0.25 means averaging reinforced signal (fixed pattern)")
    print("  fp_std      : absolute fingerprint pixel std (scale reference)")
    print("\nVisual outputs saved to:", args.out_dir)
    print("  *_fingerprint_highpass.png  — aggregated high-pass fingerprint (scaled)")
    print("  *_fingerprint_bandpass.png  — aggregated band-pass fingerprint (scaled)")
    print("  *_residuals_montage.png     — first 5 per-source residuals side-by-side")
    print("  *_fft_magnitude.png         — log-magnitude FFT of high-pass fingerprint")
    print("\nNext step: inspect the images and table, then share results to decide")
    print("per-group strategy (FIXED → residual/bandpass, ADAPTIVE → PCA or blending).")


if __name__ == "__main__":
    main()