#!/usr/bin/env python3
"""
Real-Scheme Forgery: WM_1 = dwtDct (invisible-watermark library)
====================================================================

Every earlier approach in this project ESTIMATED an approximate perturbation
from source images and added it to targets. try_known_schemes.py instead
tested whether any of the 8 "unidentified" WM groups is simply a STANDARD,
publicly documented watermarking scheme -- specifically the classical
dwtDct/dwtDctSvd methods from the `invisible-watermark` package (the same
library used as Stable Diffusion's default watermarker).

Result: WM_1 decodes with dwtDct at ~90-96% cross-source bit agreement across
a WIDE range of tested bit-lengths (24-79 bits), while a clean-image control
at the same resolution gives only ~55-59% (pure majority-vote noise). That
~35-point gap, stable across bit-length choices, is a strong, real signature
-- not a statistical artifact (verified the same way as every other
consistency claim in this project: compare against unwatermarked images of
matching resolution). No other group showed a comparable gap for either
dwtDct or dwtDctSvd.

This script:
  1. Decodes the majority-vote N-bit message from all 25 real WM_1 sources.
  2. Re-encodes that EXACT recovered message into the WM_1 clean targets using
     the SAME library's real encoder (not an approximation of one).
  3. Round-trip verifies: decoding the freshly forged targets should recover
     the same message with high fidelity, confirming the encode step worked
     mechanically (this does not by itself prove the real evaluation decoder
     agrees, but it is the strongest, most specific local evidence available
     anywhere in this project so far).

Same base_zip-patch convention: only regenerates 1.png...25.png (WM_1's
range), copies everything else unchanged from --base_zip.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
from imwatermark import WatermarkDecoder, WatermarkEncoder
from PIL import Image

EXPECTED_IMAGE_NAMES = {f"{i}.png" for i in range(1, 201)}
WM1_TARGETS = range(1, 26)


def numeric_key(path: Path) -> int:
    return int(path.stem.split("_")[-1])


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
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)
    found = find_dataset_root(dataset_dir.resolve())
    if found is None:
        raise FileNotFoundError("Could not locate clean_targets/ and watermarked_sources/.")
    return found


def recover_majority_message(source_paths: Sequence[Path], n_bits: int, method: str) -> np.ndarray:
    decoder = WatermarkDecoder("bits", n_bits)
    bits = []
    for p in source_paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise RuntimeError(f"Failed to read {p}")
        bits.append(np.asarray(decoder.decode(bgr, method), dtype=np.uint8))
    stacked = np.stack(bits, axis=0)
    majority = (stacked.mean(axis=0) >= 0.5).astype(np.uint8)
    agreement = (stacked == majority[None, :]).mean()
    return majority, agreement


def encode_targets(clean_dir: Path, targets: range, message_bits: np.ndarray, method: str, out_dir: Path) -> None:
    encoder = WatermarkEncoder()
    encoder.set_watermark("bits", message_bits.tolist())
    for i in targets:
        bgr = cv2.imread(str(clean_dir / f"{i}.png"))
        if bgr is None:
            raise RuntimeError(f"Failed to read clean target {i}.png")
        forged = encoder.encode(bgr, method)
        cv2.imwrite(str(out_dir / f"{i}.png"), forged)


def round_trip_check(out_dir: Path, targets: range, message_bits: np.ndarray, method: str) -> float:
    decoder = WatermarkDecoder("bits", len(message_bits))
    agreements = []
    for i in targets:
        bgr = cv2.imread(str(out_dir / f"{i}.png"))
        decoded = np.asarray(decoder.decode(bgr, method), dtype=np.uint8)
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
    dataset_root = ensure_dataset(args.zip_file, args.dataset_dir)
    clean_dir = dataset_root / "clean_targets"
    source_dir = dataset_root / "watermarked_sources" / "WM_1"
    source_paths = sorted(source_dir.glob("*.png"), key=numeric_key)
    if len(source_paths) != 25:
        raise RuntimeError(f"Expected 25 WM_1 sources, found {len(source_paths)}")

    print(f"Base ZIP: {args.base_zip}")
    print(f"Recovering majority-vote {args.n_bits}-bit message via {args.method} from 25 WM_1 sources...")
    message_bits, agreement = recover_majority_message(source_paths, args.n_bits, args.method)
    print(f"  cross-source bit agreement = {agreement:.4f}  (clean-image baseline is ~0.55-0.59)")
    print(f"  recovered message bits = {message_bits.tolist()}")

    extract_base_zip(args.base_zip, args.out_dir)

    print(f"Encoding recovered message into WM_1 targets (1-25.png) via {args.method}...")
    encode_targets(clean_dir, WM1_TARGETS, message_bits, args.method, args.out_dir)

    rt_agreement = round_trip_check(args.out_dir, WM1_TARGETS, message_bits, args.method)
    print(f"  round-trip decode agreement on freshly forged targets = {rt_agreement:.4f} (should be ~1.0)")

    make_flat_zip(args.out_dir, args.zip_out)
    if args.print_psnr:
        print(f"WM_1 mean PSNR vs clean targets: {mean_psnr(clean_dir, args.out_dir, WM1_TARGETS):.2f} dB")
    print(f"Done: {args.zip_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real dwtDct-scheme forgery for WM_1 (single-WM patch)")
    parser.add_argument("--zip_file", type=Path, default=Path("Dataset.zip"))
    parser.add_argument("--dataset_dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--base_zip", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--zip_out", type=Path, required=True)
    parser.add_argument("--method", choices=("dwtDct", "dwtDctSvd"), default="dwtDct")
    parser.add_argument("--n_bits", type=int, default=32)
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
