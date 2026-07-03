#!/usr/bin/env python3
"""
Real-Scheme Search, Round 2: rivaGan + properly controlled dwtDct/dwtDctSvd
================================================================================

Round 1 (try_known_schemes.py) found WM_1 = dwtDct via cross-source bit
agreement, confirmed against a clean-image control (leaderboard score
0.466486 -> 0.577703 after re-encoding the recovered message with the real
encoder). This round:
  1. Tests rivaGan (neural watermark, bundled ONNX models in the
     invisible-watermark package) across all 8 groups.
  2. Re-tests dwtDct/dwtDctSvd for the remaining 7 groups with a properly
     resolution-matched clean-image control baseline computed per group (not
     just eyeballed), so weak signals aren't missed or false leads chased.

For every (group, method) pair, reports:
  source cross-agreement, clean-control cross-agreement, and the gap between
  them. A gap far above 0 (WM_1's was ~0.30-0.35) is the signature of a real
  scheme match.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
from imwatermark import WatermarkDecoder
from PIL import Image

CATEGORIES = {
    "WM_1": (1, 25, 256), "WM_2": (26, 50, 256), "WM_3": (51, 75, 256), "WM_4": (76, 100, 256),
    "WM_5": (101, 125, 128), "WM_6": (126, 150, 256), "WM_7": (151, 175, 512), "WM_8": (176, 200, 512),
}


def numeric_key(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def find_dataset_root(base: Path) -> Path | None:
    for root in (base, base / "Dataset"):
        if (root / "clean_targets").is_dir() and (root / "watermarked_sources").is_dir():
            return root
    return None


def collect_same_res_clean(clean_dir: Path, target_res: int, limit: int) -> List[Path]:
    out = []
    for i in range(1, 201):
        p = clean_dir / f"{i}.png"
        with Image.open(p) as img:
            if img.size == (target_res, target_res):
                out.append(p)
        if len(out) >= limit:
            break
    return out


def bit_agreement(paths: List[Path], decoder: WatermarkDecoder, method: str) -> float:
    bits = []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise RuntimeError(f"Failed to read {p}")
        decoded = decoder.decode(bgr, method)
        bits.append(np.asarray(decoded, dtype=np.uint8))
    stacked = np.stack(bits, axis=0)
    majority = (stacked.mean(axis=0) >= 0.5).astype(np.uint8)
    return float((stacked == majority[None, :]).mean())


def main() -> None:
    root = find_dataset_root(Path("Dataset").resolve()) or find_dataset_root(Path.cwd())
    if root is None:
        raise FileNotFoundError("Could not locate dataset")
    clean_dir = root / "clean_targets"
    source_root = root / "watermarked_sources"

    riva_decoder = WatermarkDecoder("bits", 32)
    riva_decoder.loadModel()

    print(f"{'WM':<6} {'method':<10} {'n_bits':>6}  {'src_agree':>9}  {'ctrl_agree':>10}  {'gap':>7}")
    print("-" * 60)
    for wm_name, (start, stop, res) in CATEGORIES.items():
        source_paths = sorted((source_root / wm_name).glob("*.png"), key=numeric_key)
        clean_paths = collect_same_res_clean(clean_dir, res, len(source_paths))

        # rivaGan: fixed 32-bit
        try:
            src_a = bit_agreement(source_paths, riva_decoder, "rivaGan")
            ctrl_a = bit_agreement(clean_paths, riva_decoder, "rivaGan")
            gap = src_a - ctrl_a
            flag = "  <-- LOOKS REAL" if gap > 0.15 else ""
            print(f"{wm_name:<6} {'rivaGan':<10} {32:>6}  {src_a:>9.4f}  {ctrl_a:>10.4f}  {gap:>+7.4f}{flag}")
        except Exception as exc:
            print(f"{wm_name:<6} {'rivaGan':<10} {32:>6}  ERROR: {exc}")

        # dwtDct / dwtDctSvd, properly controlled, at a couple of bit-lengths
        if wm_name == "WM_1":
            continue  # already solved
        for method in ("dwtDct", "dwtDctSvd"):
            for n_bits in (32, 48):
                try:
                    decoder = WatermarkDecoder("bits", n_bits)
                    src_a = bit_agreement(source_paths, decoder, method)
                    ctrl_a = bit_agreement(clean_paths, decoder, method)
                    gap = src_a - ctrl_a
                    flag = "  <-- LOOKS REAL" if gap > 0.15 else ""
                    print(f"{wm_name:<6} {method:<10} {n_bits:>6}  {src_a:>9.4f}  {ctrl_a:>10.4f}  {gap:>+7.4f}{flag}")
                except Exception as exc:
                    print(f"{wm_name:<6} {method:<10} {n_bits:>6}  ERROR: {exc}")


if __name__ == "__main__":
    main()
