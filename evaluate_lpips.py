#!/usr/bin/env python3
"""
Local LPIPS-based Sqlt evaluator
=================================

Computes the REAL LPIPS distance (AlexNet backbone, matching the LPIPS paper
default) between clean targets and a candidate ZIP/dir, then converts to the
assignment's actual quality score: Sqlt = exp(-8*LPIPS).

Usage:
    python evaluate_lpips.py --zip submission_wm5_wavelet_s4.zip --wm WM_5
    python evaluate_lpips.py --dir submission_temp_wm5 --range 101 125
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Optional

import lpips
import numpy as np
import torch
from PIL import Image

WM_RANGES = {
    "WM_1": (1, 25), "WM_2": (26, 50), "WM_3": (51, 75), "WM_4": (76, 100),
    "WM_5": (101, 125), "WM_6": (126, 150), "WM_7": (151, 175), "WM_8": (176, 200),
}


def to_lpips_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def load_from_zip(zf: zipfile.ZipFile, name: str) -> Image.Image:
    with zf.open(name) as f:
        return Image.open(f).convert("RGB").copy()


def evaluate(clean_dir: Path, candidate_zip: Optional[Path], candidate_dir: Optional[Path],
             start: int, stop: int, net: str = "alex") -> None:
    loss_fn = lpips.LPIPS(net=net, verbose=False)
    loss_fn.eval()

    zf = zipfile.ZipFile(candidate_zip, "r") if candidate_zip else None
    try:
        lpips_values = []
        for i in range(start, stop + 1):
            clean_img = Image.open(clean_dir / f"{i}.png").convert("RGB")
            if zf is not None:
                cand_img = load_from_zip(zf, f"{i}.png")
            else:
                cand_img = Image.open(candidate_dir / f"{i}.png").convert("RGB")

            t_clean = to_lpips_tensor(clean_img)
            t_cand = to_lpips_tensor(cand_img)
            with torch.no_grad():
                d = loss_fn(t_clean, t_cand).item()
            lpips_values.append(d)
    finally:
        if zf is not None:
            zf.close()

    lpips_arr = np.array(lpips_values)
    sqlt = np.exp(-8.0 * lpips_arr)
    print(f"Images {start}-{stop} (n={len(lpips_arr)})")
    print(f"  mean LPIPS = {lpips_arr.mean():.5f}  (min={lpips_arr.min():.5f}, max={lpips_arr.max():.5f})")
    print(f"  mean Sqlt  = {sqlt.mean():.5f}  (min={sqlt.min():.5f}, max={sqlt.max():.5f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=Path, default=Path("Dataset/clean_targets"))
    parser.add_argument("--zip", type=Path, default=None)
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--wm", choices=sorted(WM_RANGES), default=None)
    parser.add_argument("--range", type=int, nargs=2, default=None, metavar=("START", "STOP"))
    parser.add_argument("--net", choices=("alex", "vgg", "squeeze"), default="alex")
    args = parser.parse_args()

    if args.zip is None and args.dir is None:
        raise SystemExit("Provide --zip or --dir")
    if args.wm:
        start, stop = WM_RANGES[args.wm]
    elif args.range:
        start, stop = args.range
    else:
        start, stop = 1, 200

    evaluate(args.clean_dir, args.zip, args.dir, start, stop, net=args.net)


if __name__ == "__main__":
    main()
