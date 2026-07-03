#!/usr/bin/env python3
"""
Try Known Open-Source Watermarking Schemes
=============================================

Every custom signal-processing approximation tried so far (Gaussian residual,
FFT peak-lock, wavelet denoise, content-adaptive scaling, DCT quantization
scan) has failed to move the leaderboard score. Before inventing yet another
approximation, check the much simpler possibility: one or more of these 8
"unidentified" schemes might just be a standard, publicly documented
watermarking method (the `invisible-watermark` package implements dwtDct and
dwtDctSvd, the same default watermarker used by Stable Diffusion and widely
used in coursework/research datasets).

If a scheme matches, its DECODER will recover the SAME message bits
consistently across all 25 same-message sources in a group (real decode,
not an approximation). That is a much stronger, more specific signature than
any of our earlier consistency metrics -- it's not "these images are
statistically similar", it's "this decoder extracts the exact same N-bit
string from all 25 of them". If we find that, we can forge with the SAME
library's real encoder using the recovered message: a near-exact reproduction
instead of an estimate.

This script tries dwtDct and dwtDctSvd (classical, fast, no extra model
downloads) across several common bit-lengths for all 8 groups, and reports
per-group bit-agreement-rate (fraction of the 25 decodes that agree with the
majority-vote message, per bit) -- high agreement (>>50%, ideally ~100%) is
the signature we're looking for.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
from imwatermark import WatermarkDecoder

CATEGORIES = {
    "WM_1": (1, 25), "WM_2": (26, 50), "WM_3": (51, 75), "WM_4": (76, 100),
    "WM_5": (101, 125), "WM_6": (126, 150), "WM_7": (151, 175), "WM_8": (176, 200),
}

METHODS = ("dwtDct", "dwtDctSvd")
BIT_LENGTHS = (32, 48, 64)


def find_dataset_root(base: Path) -> Path | None:
    for root in (base, base / "Dataset"):
        if (root / "clean_targets").is_dir() and (root / "watermarked_sources").is_dir():
            return root
    return None


def numeric_key(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def decode_all(source_paths: List[Path], method: str, n_bits: int) -> np.ndarray:
    decoder = WatermarkDecoder("bits", n_bits)
    bits = []
    for p in source_paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise RuntimeError(f"Failed to read {p}")
        decoded = decoder.decode(bgr, method)
        bits.append(np.asarray(decoded, dtype=np.uint8))
    return np.stack(bits, axis=0)


def bit_agreement(bits: np.ndarray) -> float:
    """Mean, over bit positions, of the majority-vote agreement fraction."""
    majority = (bits.mean(axis=0) >= 0.5).astype(np.uint8)
    agreement_per_bit = (bits == majority[None, :]).mean(axis=0)
    return float(agreement_per_bit.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--wm", choices=sorted(CATEGORIES) + ["all"], default="all")
    args = parser.parse_args()

    root = find_dataset_root(args.dataset_dir.resolve()) or find_dataset_root(Path.cwd())
    if root is None:
        raise FileNotFoundError("Could not locate dataset")
    source_root = root / "watermarked_sources"

    wms = list(CATEGORIES) if args.wm == "all" else [args.wm]
    print(f"{'WM':<6} {'method':<10} {'n_bits':>6}  mean_bit_agreement (1.0 = perfect match)")
    print("-" * 60)
    for wm_name in wms:
        source_paths = sorted((source_root / wm_name).glob("*.png"), key=numeric_key)
        for method in METHODS:
            for n_bits in BIT_LENGTHS:
                try:
                    bits = decode_all(source_paths, method, n_bits)
                    agreement = bit_agreement(bits)
                except Exception as exc:
                    agreement = float("nan")
                    print(f"{wm_name:<6} {method:<10} {n_bits:>6}  ERROR: {exc}")
                    continue
                flag = "  <-- LOOKS REAL" if agreement > 0.9 else ""
                print(f"{wm_name:<6} {method:<10} {n_bits:>6}  {agreement:.4f}{flag}")


if __name__ == "__main__":
    main()
