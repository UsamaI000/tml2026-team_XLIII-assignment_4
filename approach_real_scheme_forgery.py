#!/usr/bin/env python3
"""
Real-Scheme Forgery: Generalized (dwtDct / dwtDctSvd / rivaGan)
===================================================================

Generalizes approach_dwtdct_real_scheme.py (which found WM_1 = dwtDct,
0.466486 -> 0.577703 on the leaderboard) to any WM group and any method
supported by the `invisible-watermark` library.

try_known_schemes.py / try_known_schemes_v2.py identify a real scheme match by
cross-source bit agreement (do all 25 same-message sources decode to the same
value) being far above a same-resolution clean-image control baseline:
  WM_1 -> dwtDct   (agreement ~0.83-0.96 vs control ~0.55-0.59, CONFIRMED on leaderboard)
  WM_2 -> rivaGan  (agreement 0.989 vs control 0.626, round-trip 0.989 -- even cleaner than WM_1)

This script: recovers the majority-vote N-bit message from a WM group's real
sources using a given method, re-encodes that exact message into the group's
clean targets with the same library's real encoder, and round-trip verifies.
Same base_zip-patch convention as every other approach_*.py script here.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from imwatermark import WatermarkDecoder, WatermarkEncoder
from PIL import Image

EXPECTED_IMAGE_NAMES = {f"{i}.png" for i in range(1, 201)}

WM_RANGES = {
    "WM_1": range(1, 26), "WM_2": range(26, 51), "WM_3": range(51, 76), "WM_4": range(76, 101),
    "WM_5": range(101, 126), "WM_6": range(126, 151), "WM_7": range(151, 176), "WM_8": range(176, 201),
}

NEEDS_LOAD_MODEL = {"rivaGan"}


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


def recover_majority_message(source_paths: Sequence[Path], decoder: WatermarkDecoder, method: str):
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


def encode_targets(clean_dir: Path, targets: range, message_bits: np.ndarray, method: str,
                    encoder: WatermarkEncoder, out_dir: Path) -> None:
    encoder.set_watermark("bits", message_bits.tolist())
    for i in targets:
        bgr = cv2.imread(str(clean_dir / f"{i}.png"))
        if bgr is None:
            raise RuntimeError(f"Failed to read clean target {i}.png")
        forged = encoder.encode(bgr, method)
        cv2.imwrite(str(out_dir / f"{i}.png"), forged)


def round_trip_check(out_dir: Path, targets: range, message_bits: np.ndarray,
                      decoder: WatermarkDecoder, method: str) -> float:
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
    targets = WM_RANGES[args.wm]
    source_dir = dataset_root / "watermarked_sources" / args.wm
    source_paths = sorted(source_dir.glob("*.png"), key=numeric_key)
    if len(source_paths) != len(targets):
        raise RuntimeError(f"Expected {len(targets)} sources for {args.wm}, found {len(source_paths)}")

    decoder = WatermarkDecoder("bits", args.n_bits)
    encoder = WatermarkEncoder()
    if args.method in NEEDS_LOAD_MODEL:
        decoder.loadModel()
        encoder.loadModel()

    print(f"Base ZIP: {args.base_zip}")
    print(f"Recovering majority-vote {args.n_bits}-bit message via {args.method} from {len(source_paths)} {args.wm} sources...")
    message_bits, agreement = recover_majority_message(source_paths, decoder, args.method)
    print(f"  cross-source bit agreement = {agreement:.4f}")
    print(f"  recovered message bits = {message_bits.tolist()}")

    extract_base_zip(args.base_zip, args.out_dir)

    print(f"Encoding recovered message into {args.wm} targets via {args.method}...")
    encode_targets(clean_dir, targets, message_bits, args.method, encoder, args.out_dir)

    rt_agreement = round_trip_check(args.out_dir, targets, message_bits, decoder, args.method)
    print(f"  round-trip decode agreement on freshly forged targets = {rt_agreement:.4f} (should be ~1.0)")

    make_flat_zip(args.out_dir, args.zip_out)
    if args.print_psnr:
        print(f"{args.wm} mean PSNR vs clean targets: {mean_psnr(clean_dir, args.out_dir, targets):.2f} dB")
    print(f"Done: {args.zip_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-scheme forgery for any WM group (single-WM patch)")
    parser.add_argument("--zip_file", type=Path, default=Path("Dataset.zip"))
    parser.add_argument("--dataset_dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--base_zip", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--zip_out", type=Path, required=True)
    parser.add_argument("--wm", choices=sorted(WM_RANGES), required=True)
    parser.add_argument("--method", choices=("dwtDct", "dwtDctSvd", "rivaGan"), required=True)
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
