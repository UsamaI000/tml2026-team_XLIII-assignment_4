#!/usr/bin/env python3
"""
Patch WM_7 and WM_8 using FFT Peak-Picking
===========================================

This script follows the same pattern as approach_patch_wm4_wm6_artifacts.py:
it takes a known-good full submission ZIP as input (your current best at 0.444516,
which is the output of the wm4_edge_only patch on top of per_wm_consistency_v1),
and only overwrites images 151-175 (WM_7) and 176-200 (WM_8).

Why WM_7 and WM_8:
    Diagnostic showed these have the highest peak_ratio (0.39, 0.285) meaning
    their watermark energy is concentrated at specific FFT frequencies — a strong
    signal that the watermark is classical/spread-spectrum. But fp/src was near
    zero (0.049, 0.041), meaning the current averaging approach barely recovers
    any signal. This contradiction means the watermark IS recoverable but lives
    at specific frequency bins that a smooth bandpass smears over.

Why FFT peak-picking works here:
    Instead of a smooth radial bandpass, we:
    1. Compute per-source residuals and their FFT spectra
    2. Average the magnitude spectra across all 25 sources
       → Bins where the watermark lives get reinforced (consistent across sources)
       → Bins with content energy cancel out (varies per source)
    3. Keep only the top-K magnitude bins in the averaged spectrum
    4. Reconstruct a spatial fingerprint using the averaged complex spectrum
       masked to those K bins, then inverse FFT
    This finds the exact frequency locations of the watermark rather than
    guessing a broad band.

Usage (recommended order):

    # Step 1: conservative, lowest risk
    python approach_patch_wm7_wm8_fftpeak.py \
        --zip_file Dataset.zip \
        --dataset_dir Dataset \
        --base_zip submission_temp_wm4_edge_only/... \
        --zip_out submission_patch_wm78_conservative.zip \
        --profile conservative

    # Step 2: if conservative helps, try balanced
    python approach_patch_wm7_wm8_fftpeak.py \
        --zip_file Dataset.zip \
        --dataset_dir Dataset \
        --base_zip submission_temp_wm4_edge_only/... \
        --zip_out submission_patch_wm78_balanced.zip \
        --profile balanced

    # Step 3: isolate which group is driving improvement
    python approach_patch_wm7_wm8_fftpeak.py ... --profile wm7_only
    python approach_patch_wm7_wm8_fftpeak.py ... --profile wm8_only

    # Step 4: try source-mean residual variant if peak-picking plateaus
    python approach_patch_wm7_wm8_fftpeak.py ... --profile source_mean_conservative

Note on --base_zip:
    Point this at whatever ZIP is currently your best submission.
    The script reads all 200 images from it, patches WM_7 and WM_8, and
    writes a new flat ZIP. Images for all other groups are passed through unchanged.
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

WM7_RANGE = range(151, 176)   # targets 151-175
WM8_RANGE = range(176, 201)   # targets 176-200


# ── Profile definitions ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class PatchProfile:
    # --- shared residual extraction params (applied before FFT) ---
    input_sigma: float        # Gaussian blur sigma for pre-FFT high-pass residual
                              # Use small sigma (0.5-1.5) to avoid bleeding source content

    # --- FFT peak-picking params ---
    peak_top_fraction: float  # fraction of FFT bins to keep (0.001 = top 0.1%)
                              # Lower = more targeted, higher = captures more signal
    freq_low: float           # ignore bins below this normalized radius (exclude DC region)
    freq_high: float          # ignore bins above this radius (exclude extreme high freq noise)

    # --- per-group strength and clip ---
    wm7_strength: float       # perturbation amplitude for WM_7 targets
    wm8_strength: float       # perturbation amplitude for WM_8 targets
    wm7_clip: float           # hard pixel-level clip for WM_7 perturbation
    wm8_clip: float           # hard pixel-level clip for WM_8 perturbation
    final_clip: float         # additional safety clip on final perturbed image delta

    # --- alternative strategy flag ---
    use_source_mean: bool     # if True, use residual-of-mean instead of mean-of-residuals
                              # Better when per-source residuals have opposite-phase cancellation


PROFILES = {
    # Safest first attempt. Small fraction, moderate strength.
    "conservative": PatchProfile(
        input_sigma=1.0,
        peak_top_fraction=0.002,
        freq_low=0.04, freq_high=0.45,
        wm7_strength=4.0, wm8_strength=4.0,
        wm7_clip=10.0, wm8_clip=10.0,
        final_clip=14.0,
        use_source_mean=False,
    ),
    # Moderate. Wider peak selection, higher strength.
    "balanced": PatchProfile(
        input_sigma=1.0,
        peak_top_fraction=0.004,
        freq_low=0.04, freq_high=0.45,
        wm7_strength=5.0, wm8_strength=5.0,
        wm7_clip=12.0, wm8_clip=12.0,
        final_clip=16.0,
        use_source_mean=False,
    ),
    # Aggressive. Use only if balanced helps.
    "aggressive": PatchProfile(
        input_sigma=0.8,
        peak_top_fraction=0.006,
        freq_low=0.03, freq_high=0.47,
        wm7_strength=6.5, wm8_strength=6.5,
        wm7_clip=14.0, wm8_clip=14.0,
        final_clip=18.0,
        use_source_mean=False,
    ),
    # Isolate WM_7 only. WM_8 images are left as-is from base ZIP.
    "wm7_only": PatchProfile(
        input_sigma=1.0,
        peak_top_fraction=0.004,
        freq_low=0.04, freq_high=0.45,
        wm7_strength=5.0, wm8_strength=0.0,  # 0.0 = skip WM_8
        wm7_clip=12.0, wm8_clip=0.0,
        final_clip=16.0,
        use_source_mean=False,
    ),
    # Isolate WM_8 only. WM_7 images are left as-is from base ZIP.
    "wm8_only": PatchProfile(
        input_sigma=1.0,
        peak_top_fraction=0.004,
        freq_low=0.04, freq_high=0.45,
        wm7_strength=0.0, wm8_strength=5.0,  # 0.0 = skip WM_7
        wm7_clip=0.0, wm8_clip=12.0,
        final_clip=16.0,
        use_source_mean=False,
    ),
    # Alternative: residual-of-mean instead of mean-of-residuals.
    # Avoids phase cancellation when sources embed with varying phase.
    "source_mean_conservative": PatchProfile(
        input_sigma=1.0,
        peak_top_fraction=0.002,
        freq_low=0.04, freq_high=0.45,
        wm7_strength=4.0, wm8_strength=4.0,
        wm7_clip=10.0, wm8_clip=10.0,
        final_clip=14.0,
        use_source_mean=True,
    ),
    "source_mean_balanced": PatchProfile(
        input_sigma=1.0,
        peak_top_fraction=0.004,
        freq_low=0.04, freq_high=0.45,
        wm7_strength=5.0, wm8_strength=5.0,
        wm7_clip=12.0, wm8_clip=12.0,
        final_clip=16.0,
        use_source_mean=True,
    ),
    # Tighter frequency band for WM_7 (diagnostic showed concentrated vertical stripe)
    "wm7_vertical_focus": PatchProfile(
        input_sigma=1.0,
        peak_top_fraction=0.003,
        freq_low=0.06, freq_high=0.40,  # matches WM_7 vertical stripe band
        wm7_strength=5.0, wm8_strength=0.0,
        wm7_clip=12.0, wm8_clip=0.0,
        final_clip=16.0,
        use_source_mean=False,
        ),
    "micro": PatchProfile(
        input_sigma=1.0,
        peak_top_fraction=0.0002,   # 10x fewer bins than conservative
        freq_low=0.04, freq_high=0.45,
        wm7_strength=1.5, wm8_strength=1.5,  # much lower strength
        wm7_clip=4.0, wm8_clip=4.0,
        final_clip=6.0,
        use_source_mean=False,
    ),
    "micro_srcmean": PatchProfile(
        input_sigma=1.0,
        peak_top_fraction=0.0002,
        freq_low=0.04, freq_high=0.45,
        wm7_strength=1.5, wm8_strength=1.5,
        wm7_clip=4.0, wm8_clip=4.0,
        final_clip=6.0,
        use_source_mean=True,   # averages sources first, avoids phase noise
    ),
    "micro_strength2": PatchProfile(
        input_sigma=1.0,
        peak_top_fraction=0.0002,
        freq_low=0.04, freq_high=0.45,
        wm7_strength=2.5, wm8_strength=2.5,
        wm7_clip=6.0, wm8_clip=6.0,
        final_clip=8.0,
        use_source_mean=True,
    ),
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
    Image.fromarray(
        np.clip(np.rint(array), 0, 255).astype(np.uint8), mode="RGB"
    ).save(path)


def gaussian_blur_rgb(array: np.ndarray, sigma: float) -> np.ndarray:
    img = Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(
        img.filter(ImageFilter.GaussianBlur(radius=float(sigma))), dtype=np.float32
    )


# ── Signal processing ─────────────────────────────────────────────────────────

def robust_channel_scale(array: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    median = np.median(array, axis=(0, 1), keepdims=True)
    centered = array - median
    mad = 1.4826 * np.median(np.abs(centered), axis=(0, 1), keepdims=True)
    p95 = np.percentile(np.abs(centered), 95, axis=(0, 1), keepdims=True)
    std = np.std(centered, axis=(0, 1), keepdims=True)
    return np.maximum.reduce([mad, p95 / 2.0, std, np.full_like(std, eps)])


def normalize_fingerprint(fp: np.ndarray, strength: float, clip: float) -> np.ndarray:
    if strength <= 0:
        return np.zeros_like(fp, dtype=np.float32)
    centered = fp - np.mean(fp, axis=(0, 1), keepdims=True)
    normalized = centered / robust_channel_scale(centered) * strength
    return np.clip(normalized, -clip, clip).astype(np.float32)


def highpass_residual(image: np.ndarray, sigma: float) -> np.ndarray:
    res = image - gaussian_blur_rgb(image, sigma)
    return res - np.median(res, axis=(0, 1), keepdims=True)


def fft_peak_fingerprint(
    source_paths: Sequence[Path],
    p: PatchProfile,
    strength: float,
    clip: float,
    wm_name: str,
) -> np.ndarray:
    """
    Estimate a watermark fingerprint by finding the specific FFT bins that are
    consistently energetic across all source images.

    Strategy A (use_source_mean=False): mean-of-residuals
        - Extract a high-pass residual from each source independently
        - Average their complex FFT spectra
        - Pick the top-K magnitude bins from the averaged spectrum
        - Inverse FFT to get the spatial fingerprint
        Pros: robust to per-source content variation
        Cons: per-source phase noise can cause partial cancellation

    Strategy B (use_source_mean=True): residual-of-mean
        - Average all source images first (content cancels, watermark reinforces)
        - Extract high-pass residual from that mean image
        - Pick top-K FFT bins from its spectrum
        Pros: no phase cancellation
        Cons: requires content to be diverse enough to cancel (25 sources helps)
    """
    if strength <= 0:
        img0 = read_rgb(source_paths[0])
        return np.zeros_like(img0, dtype=np.float32)

    source_paths = list(source_paths)
    h, w, c = read_rgb(source_paths[0]).shape

    if p.use_source_mean:
        # Strategy B: residual of the mean image
        mean_img = np.mean(
            np.stack([read_rgb(path) for path in source_paths], axis=0), axis=0
        )
        res = highpass_residual(mean_img, p.input_sigma)
        res = res / robust_channel_scale(res)
        avg_spectrum = np.fft.fft2(res, axes=(0, 1))
    else:
        # Strategy A: mean of per-source complex spectra
        spectra = []
        for path in source_paths:
            img = read_rgb(path)
            res = highpass_residual(img, p.input_sigma)
            res = res / robust_channel_scale(res)
            spectra.append(np.fft.fft2(res, axes=(0, 1)))
        avg_spectrum = np.mean(np.stack(spectra, axis=0), axis=0)

    # Build eligible frequency mask (annular band, same logic as bandpass scripts)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fx**2 + fy**2)
    eligible = (r >= p.freq_low) & (r <= p.freq_high)  # shape (h, w)

    # Average magnitude across channels to get a single 2D magnitude map
    avg_magnitude = np.mean(np.abs(avg_spectrum), axis=2)  # (h, w)

    # Find threshold for top-K bins within the eligible band
    eligible_magnitudes = avg_magnitude[eligible]
    if eligible_magnitudes.size == 0:
        raise ValueError(f"{wm_name}: no FFT bins in freq range [{p.freq_low}, {p.freq_high}]")

    n_keep = max(1, int(round(p.peak_top_fraction * eligible_magnitudes.size)))
    threshold = np.sort(eligible_magnitudes)[-n_keep]

    # Binary mask: keep only top-K eligible bins
    peak_mask = ((avg_magnitude >= threshold) & eligible).astype(np.float32)[:, :, None]

    n_selected = int(peak_mask.sum())
    total_eligible = int(eligible.sum())
    print(f"  {wm_name}: selected {n_selected} / {total_eligible} eligible FFT bins "
          f"(top {p.peak_top_fraction*100:.2f}% of eligible band)")

    # Reconstruct spatial fingerprint from peaked spectrum
    filtered = np.fft.ifft2(avg_spectrum * peak_mask, axes=(0, 1)).real.astype(np.float32)
    fp = normalize_fingerprint(filtered, strength, clip)
    return fp


# ── Base ZIP extraction ───────────────────────────────────────────────────────

def extract_base_zip(base_zip: Path, out_dir: Path) -> None:
    """Extract the base submission ZIP into out_dir, validating it is a flat 200-image set."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not base_zip.is_file():
        raise FileNotFoundError(f"Missing base ZIP: {base_zip}")
    with zipfile.ZipFile(base_zip, "r") as zf:
        names = set(zf.namelist())
        if names != EXPECTED_IMAGE_NAMES:
            missing = sorted(EXPECTED_IMAGE_NAMES - names, key=lambda x: int(x[:-4]))
            extra = sorted(names - EXPECTED_IMAGE_NAMES)
            raise RuntimeError(
                f"Base ZIP must contain exactly flat 1.png...200.png. "
                f"Missing={missing[:5]}, extra={extra[:5]}"
            )
        zf.extractall(out_dir)


# ── Output ZIP ────────────────────────────────────────────────────────────────

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
            raise RuntimeError("ZIP validation failed: not a flat 200-image submission.")


def mean_psnr(clean_dir: Path, out_dir: Path) -> float:
    vals = []
    for i in range(1, 201):
        clean = read_rgb(clean_dir / f"{i}.png")
        forged = read_rgb(out_dir / f"{i}.png")
        mse = np.mean((clean - forged) ** 2)
        vals.append(float("inf") if mse <= 1e-12 else 20 * np.log10(255.0 / np.sqrt(mse)))
    return float(np.mean(vals))


# ── Main ──────────────────────────────────────────────────────────────────────

def build(args: argparse.Namespace) -> None:
    dataset_root = ensure_dataset(args.zip_file, args.dataset_dir)
    clean_dir = dataset_root / "clean_targets"
    source_root = dataset_root / "watermarked_sources"
    profile = PROFILES[args.profile]

    print(f"Dataset root : {dataset_root}")
    print(f"Base ZIP     : {args.base_zip}")
    print(f"Profile      : {args.profile}")
    print(f"  input_sigma={profile.input_sigma}, "
          f"peak_top_fraction={profile.peak_top_fraction}, "
          f"freq=[{profile.freq_low}, {profile.freq_high}], "
          f"use_source_mean={profile.use_source_mean}")
    print(f"  WM_7: strength={profile.wm7_strength}, clip={profile.wm7_clip}")
    print(f"  WM_8: strength={profile.wm8_strength}, clip={profile.wm8_clip}")
    print(f"Output ZIP   : {args.zip_out}")
    print()

    # Start from the current best submission
    extract_base_zip(args.base_zip, args.out_dir)

    # Estimate fingerprints
    wm7_sources = sorted((source_root / "WM_7").glob("*.png"), key=numeric_key)
    wm8_sources = sorted((source_root / "WM_8").glob("*.png"), key=numeric_key)

    wm7_fp = fft_peak_fingerprint(
        wm7_sources, profile,
        strength=profile.wm7_strength, clip=profile.wm7_clip, wm_name="WM_7"
    )
    wm8_fp = fft_peak_fingerprint(
        wm8_sources, profile,
        strength=profile.wm8_strength, clip=profile.wm8_clip, wm_name="WM_8"
    )

    print(f"  WM_7 fingerprint range=[{wm7_fp.min():.2f}, {wm7_fp.max():.2f}], "
          f"std={wm7_fp.std():.3f}")
    print(f"  WM_8 fingerprint range=[{wm8_fp.min():.2f}, {wm8_fp.max():.2f}], "
          f"std={wm8_fp.std():.3f}")

    # Patch WM_7 targets (151-175)
    if profile.wm7_strength > 0:
        for i in WM7_RANGE:
            img = read_rgb(args.out_dir / f"{i}.png")
            if img.shape != wm7_fp.shape:
                raise RuntimeError(f"Shape mismatch for {i}.png: img={img.shape}, fp={wm7_fp.shape}")
            delta = np.clip(wm7_fp, -profile.final_clip, profile.final_clip)
            save_rgb(img + delta, args.out_dir / f"{i}.png")
        print(f"Patched WM_7 targets: {list(WM7_RANGE)[0]}-{list(WM7_RANGE)[-1]}")
    else:
        print("WM_7: skipped (strength=0)")

    # Patch WM_8 targets (176-200)
    if profile.wm8_strength > 0:
        for i in WM8_RANGE:
            img = read_rgb(args.out_dir / f"{i}.png")
            if img.shape != wm8_fp.shape:
                raise RuntimeError(f"Shape mismatch for {i}.png: img={img.shape}, fp={wm8_fp.shape}")
            delta = np.clip(wm8_fp, -profile.final_clip, profile.final_clip)
            save_rgb(img + delta, args.out_dir / f"{i}.png")
        print(f"Patched WM_8 targets: {list(WM8_RANGE)[0]}-{list(WM8_RANGE)[-1]}")
    else:
        print("WM_8: skipped (strength=0)")

    validate_and_zip(args.out_dir, args.zip_out)

    if args.print_psnr:
        psnr = mean_psnr(clean_dir, args.out_dir)
        print(f"Mean PSNR vs clean targets: {psnr:.2f} dB")

    print(f"\nDone. Created {args.zip_out}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Patch WM_7 and WM_8 in a submission ZIP using FFT peak-picking."
    )
    parser.add_argument("--zip_file", type=Path, default=Path("Dataset.zip"),
                        help="Original dataset ZIP (for source images).")
    parser.add_argument("--dataset_dir", type=Path, default=Path("Dataset"),
                        help="Extracted dataset directory.")
    parser.add_argument(
        "--base_zip", type=Path,
        default=Path("submission_temp_wm4_edge_only.zip"),
        help="Your current best submission ZIP (200 flat PNGs). "
             "WM_7 and WM_8 images in this ZIP will be patched.",
    )
    parser.add_argument("--out_dir", type=Path,
                        default=Path("submission_temp_patch_wm78_fftpeak"))
    parser.add_argument("--zip_out", type=Path,
                        default=Path("submission_patch_wm78_fftpeak.zip"))
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES.keys()),
        default="conservative",
        help="Which patch profile to use. Start with 'conservative'.",
    )
    parser.add_argument("--print_psnr", action="store_true",
                        help="Print mean PSNR vs clean targets after patching.")
    args = parser.parse_args(argv)

    try:
        build(args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
