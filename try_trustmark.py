#!/usr/bin/env python3
"""
Real-Scheme Search, Round 3 (retry): TrustMark, isolated venv
==================================================================

Same methodology as try_known_schemes.py / try_known_schemes_v2.py /
try_blind_watermark.py (which found WM_1=dwtDct and WM_2=rivaGan, and ruled
out blind_watermark for WM_3/4/5/6/7): decode all same-message sources per
group and check whether they agree with each other far more than a
same-resolution clean-image control does.

Tests TrustMark (Meta / Adobe Content Credentials). Earlier attempt in the
main venv hit an unresolvable numpy<2.0 (trustmark) vs numpy>=2.0 (scipy/
torchmetrics used elsewhere) dependency conflict. This runs in a separate
`venv_trustmark` environment created just for this test, so it can't
destabilize the main LPIPS/pywt/opencv stack.

Run with: venv_trustmark/Scripts/python.exe try_trustmark.py
(NOT the main venv's python -- trustmark is not installed there).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from trustmark import TrustMark

CATEGORIES = {
    "WM_3": (51, 75, 256), "WM_4": (76, 100, 256), "WM_5": (101, 125, 128),
    "WM_6": (126, 150, 256), "WM_7": (151, 175, 512),
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


def raw_decode(tm: TrustMark, img: Image.Image) -> np.ndarray:
    """Bypass BCH error-correction (tm.decode() discards everything if no
    valid codeword is found, which would hide a real-but-imperfect signal
    from our consistency test) and read the neural decoder's raw bit output
    directly -- mirrors TrustMark.subimage_decode()'s first two lines."""
    resized = img.resize((tm.model_resolution_dec, tm.model_resolution_dec), Image.BILINEAR)
    stego = transforms.ToTensor()(resized).unsqueeze(0).to(tm.decoder.device) * 2.0 - 1.0
    with torch.no_grad():
        secret_binaryarray = (tm.decoder.decoder(stego) > 0).cpu().numpy()
    return secret_binaryarray[0].astype(np.uint8)


def decode_bits(tm: TrustMark, paths: List[Path]) -> np.ndarray:
    bits = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        bits.append(raw_decode(tm, img))
    return np.stack(bits, axis=0)


def bit_agreement(bits: np.ndarray) -> float:
    majority = (bits.mean(axis=0) >= 0.5).astype(np.uint8)
    return float((bits == majority[None, :]).mean())


def main() -> None:
    root = find_dataset_root(Path("Dataset").resolve()) or find_dataset_root(Path.cwd())
    if root is None:
        raise FileNotFoundError("Could not locate dataset")
    clean_dir = root / "clean_targets"
    source_root = root / "watermarked_sources"

    for variant in ("Q", "P", "C"):
        print(f"\n=== TrustMark variant: {variant} ===", flush=True)
        try:
            tm = TrustMark(verbose=False, model_type=variant)
        except Exception as exc:
            print(f"  ERROR loading model: {exc}", flush=True)
            continue

        for wm_name, (start, stop, res) in CATEGORIES.items():
            source_paths = sorted((source_root / wm_name).glob("*.png"), key=numeric_key)
            clean_paths = collect_same_res_clean(clean_dir, res, len(source_paths))
            try:
                src_bits = decode_bits(tm, source_paths)
                ctrl_bits = decode_bits(tm, clean_paths)
                src_a = bit_agreement(src_bits)
                ctrl_a = bit_agreement(ctrl_bits)
                gap = src_a - ctrl_a
                flag = "  <-- LOOKS REAL" if gap > 0.15 else ""
                print(f"  {wm_name:<6} src={src_a:.4f}  ctrl={ctrl_a:.4f}  gap={gap:+.4f}{flag}", flush=True)
            except Exception as exc:
                print(f"  {wm_name:<6} ERROR: {exc}", flush=True)


if __name__ == "__main__":
    main()
