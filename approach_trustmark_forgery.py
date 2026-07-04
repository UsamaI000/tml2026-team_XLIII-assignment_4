#!/usr/bin/env python3
"""
Real-Scheme Forgery: WM_7 = TrustMark 'Q' variant
=====================================================

try_trustmark.py found WM_7 decodes with TrustMark's 'Q' variant at PERFECT
cross-source agreement (all 25 sources -> the exact same 100-bit raw decoder
output), vs a same-resolution clean-image control baseline of ~0.61 -- the
strongest signature found in this project (stronger than even WM_1/WM_2).

This recovers that exact 100-bit message and re-encodes it into WM_7's clean
targets using TrustMark's own real encoder network, bypassing its optional
BCH error-correction wrapper on both ends (use_ECC=False) so we operate
directly at the same raw-bit interface used for detection -- not an
approximation of the scheme, the same mechanism.

IMPORTANT: this must run with venv_trustmark's python, not the main venv's
(trustmark's numpy<2.0 requirement conflicts with the main venv's scipy/
torchmetrics stack -- see the memory note on this). Example:
    venv_trustmark/Scripts/python.exe approach_trustmark_forgery.py \\
        --dataset_dir Dataset --base_zip latest_best_results_070.zip \\
        --out_dir temp_wm7 --zip_out submission_wm7_trustmark.zip --print_psnr

Same base_zip-patch convention as every other approach_*.py script here:
only regenerates WM_7's targets (151-175), copies everything else unchanged.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from trustmark import TrustMark

EXPECTED_IMAGE_NAMES = {f"{i}.png" for i in range(1, 201)}
WM7_TARGETS = range(151, 176)


def numeric_key(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def find_dataset_root(base: Path) -> Path | None:
    for root in (base, base / "Dataset"):
        if (root / "clean_targets").is_dir() and (root / "watermarked_sources").is_dir():
            return root
    return None


def ensure_dataset(dataset_dir: Path) -> Path:
    found = find_dataset_root(dataset_dir.resolve()) or find_dataset_root(Path.cwd())
    if found is None:
        raise FileNotFoundError("Could not locate clean_targets/ and watermarked_sources/.")
    return found


def raw_decode(tm: TrustMark, img: Image.Image) -> np.ndarray:
    """Bypass BCH error-correction and read the neural decoder's raw bits directly."""
    resized = img.resize((tm.model_resolution_dec, tm.model_resolution_dec), Image.BILINEAR)
    stego = transforms.ToTensor()(resized).unsqueeze(0).to(tm.decoder.device) * 2.0 - 1.0
    with torch.no_grad():
        secret_binaryarray = (tm.decoder.decoder(stego) > 0).cpu().numpy()
    return secret_binaryarray[0].astype(np.uint8)


def recover_majority_message(tm: TrustMark, source_paths: Sequence[Path]):
    bits = []
    for p in source_paths:
        img = Image.open(p).convert("RGB")
        bits.append(raw_decode(tm, img))
    stacked = np.stack(bits, axis=0)
    majority = (stacked.mean(axis=0) >= 0.5).astype(np.uint8)
    agreement = (stacked == majority[None, :]).mean()
    return majority, agreement


def encode_targets(tm: TrustMark, clean_dir: Path, targets: range, message_bits: np.ndarray, out_dir: Path) -> None:
    secret_str = "".join(str(b) for b in message_bits.tolist())
    tm.use_ECC = False  # operate at the same raw-bit interface used for detection
    for i in targets:
        img = Image.open(clean_dir / f"{i}.png").convert("RGB")
        forged = tm.encode(img, secret_str, MODE="binary")
        forged.save(out_dir / f"{i}.png")


def round_trip_check(tm: TrustMark, out_dir: Path, targets: range, message_bits: np.ndarray) -> float:
    agreements = []
    for i in targets:
        img = Image.open(out_dir / f"{i}.png").convert("RGB")
        decoded = raw_decode(tm, img)
        agreements.append((decoded == message_bits).mean())
    return float(np.mean(agreements))


def validate_base_zip(base_zip: Path) -> None:
    if not base_zip.is_file():
        raise FileNotFoundError(f"Missing base ZIP: {base_zip}")
    with zipfile.ZipFile(base_zip, "r") as zipf:
        names = set(zipf.namelist())
    if names != EXPECTED_IMAGE_NAMES:
        raise RuntimeError("Base ZIP is not a flat 1.png...200.png submission.")


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
        raise RuntimeError(f"Output mismatch. Missing={missing[:10]}")
    zip_out.parent.mkdir(parents=True, exist_ok=True)
    if zip_out.exists():
        zip_out.unlink()
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zipf:
        for i in range(1, 201):
            p = out_dir / f"{i}.png"
            zipf.write(p, arcname=p.name)
    with zipfile.ZipFile(zip_out, "r") as zipf:
        if set(zipf.namelist()) != EXPECTED_IMAGE_NAMES:
            raise RuntimeError("ZIP validation failed")


def mean_psnr(clean_dir: Path, out_dir: Path, targets: range) -> float:
    values = []
    for i in targets:
        clean = np.asarray(Image.open(clean_dir / f"{i}.png").convert("RGB"), dtype=np.float32)
        forged = np.asarray(Image.open(out_dir / f"{i}.png").convert("RGB"), dtype=np.float32)
        mse = np.mean((clean - forged) ** 2)
        values.append(float("inf") if mse <= 1e-12 else 20.0 * np.log10(255.0 / np.sqrt(mse)))
    return float(np.mean(values))


def build_submission(args: argparse.Namespace) -> None:
    dataset_root = ensure_dataset(args.dataset_dir)
    clean_dir = dataset_root / "clean_targets"
    source_dir = dataset_root / "watermarked_sources" / "WM_7"
    source_paths = sorted(source_dir.glob("*.png"), key=numeric_key)
    if len(source_paths) != 25:
        raise RuntimeError(f"Expected 25 WM_7 sources, found {len(source_paths)}")

    print(f"Base ZIP: {args.base_zip}")
    print(f"Loading TrustMark variant {args.variant}...")
    tm = TrustMark(verbose=False, model_type=args.variant)

    print("Recovering majority-vote message from 25 WM_7 sources...")
    message_bits, agreement = recover_majority_message(tm, source_paths)
    print(f"  cross-source bit agreement = {agreement:.4f}")

    extract_base_zip(args.base_zip, args.out_dir)

    print("Encoding recovered message into WM_7 targets (151-175.png)...")
    encode_targets(tm, clean_dir, WM7_TARGETS, message_bits, args.out_dir)

    rt_agreement = round_trip_check(tm, args.out_dir, WM7_TARGETS, message_bits)
    print(f"  round-trip decode agreement on freshly forged targets = {rt_agreement:.4f} (should be ~1.0)")

    make_flat_zip(args.out_dir, args.zip_out)
    if args.print_psnr:
        print(f"WM_7 mean PSNR vs clean targets: {mean_psnr(clean_dir, args.out_dir, WM7_TARGETS):.2f} dB")
    print(f"Done: {args.zip_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real TrustMark-scheme forgery for WM_7 (single-WM patch)")
    parser.add_argument("--dataset_dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--base_zip", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--zip_out", type=Path, required=True)
    parser.add_argument("--variant", choices=("Q", "P", "C"), default="Q")
    parser.add_argument("--print_psnr", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        build_submission(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
