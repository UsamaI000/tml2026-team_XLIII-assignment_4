# Watermark Forgery — Reproducing the Best Result (0.901033)

## Approach summary

Four of the eight watermark families turned out to be standard, publicly
documented watermarking schemes. Instead of estimating an approximate
perturbation, we **identified the exact scheme** for each (by checking
whether all 25 same-message source images decode to the same bits far more
often than a same-resolution clean/unwatermarked control does), recovered
the true embedded message, and re-encoded it into the assigned clean targets
using that scheme's own real encoder:

| Group | Scheme | Library |
|---|---|---|
| WM_1 | `dwtDct` | `invisible-watermark` |
| WM_2 | `rivaGan` | `invisible-watermark` |
| WM_7 | TrustMark, variant `Q` | `trustmark` |
| WM_8 | TrustMark, variant `P` | `trustmark` |

The remaining four groups (WM_3, WM_4, WM_5, WM_6) did not match any scheme
we tested (`dwtDct`, `dwtDctSvd`, `rivaGan`, `blind_watermark`, LSB, TrustMark
Q/P/C) and use the signal-processing baseline from earlier experimentation
(see `Experiments.xlsx` for that history) — that baseline is included as the
starting checkpoint below, since re-deriving it via code would mean
re-running ~100 historical experiments.

## Prerequisites: two Python environments

`trustmark` requires `numpy<2.0`, while the rest of this project's tooling
requires `numpy>=2.0`. These cannot coexist in one environment, so **two
separate virtual environments are required**:

- **`venv`** — main environment, used for the WM_1/WM_2 fix (`dwtDct`,
  `rivaGan`, via `invisible-watermark`).
- **`venv_trustmark`** — isolated environment, used only for the WM_7/WM_8
  fix (`trustmark`).

### Setup

```bash
# Main environment
python -m venv venv
venv/Scripts/python.exe -m pip install -r requirements.txt      # Windows
# source venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

# Isolated TrustMark environment
python -m venv venv_trustmark
venv_trustmark/Scripts/python.exe -m pip install -r requirements_trustmark.txt      # Windows
# source venv_trustmark/bin/activate && pip install -r requirements_trustmark.txt   # macOS/Linux
```

### Dataset

Extract the assignment dataset so that `Dataset/clean_targets/` and
`Dataset/watermarked_sources/WM_1..WM_8/` exist under the project root
(both scripts below will also auto-extract from `Dataset.zip` if present
and `Dataset/` is missing).

## Reproducing the 0.901033 result

Starting point: `latest_best_result.zip` (the pre-fix baseline containing the
correct WM_3/4/5/6 images already; scores 0.466486 on its own). Apply the
four fixes in sequence, each patching only its own WM group's 25 images and
leaving everything else untouched:

```bash
# 1. WM_1 -> dwtDct   (main venv)
venv/Scripts/python.exe approach_real_scheme_forgery.py \
  --dataset_dir Dataset --base_zip latest_best_result.zip \
  --out_dir out_wm1 --zip_out step1_wm1.zip \
  --wm WM_1 --method dwtDct --n_bits 32

# 2. WM_2 -> rivaGan   (main venv)
venv/Scripts/python.exe approach_real_scheme_forgery.py \
  --dataset_dir Dataset --base_zip step1_wm1.zip \
  --out_dir out_wm2 --zip_out step2_wm2.zip \
  --wm WM_2 --method rivaGan --n_bits 32

# 3. WM_7 -> TrustMark Q   (isolated venv_trustmark)
venv_trustmark/Scripts/python.exe approach_trustmark_forgery.py \
  --dataset_dir Dataset --base_zip step2_wm2.zip \
  --out_dir out_wm7 --zip_out step3_wm7.zip \
  --wm WM_7 --variant Q

# 4. WM_8 -> TrustMark P   (isolated venv_trustmark)
venv_trustmark/Scripts/python.exe approach_trustmark_forgery.py \
  --dataset_dir Dataset --base_zip step3_wm7.zip \
  --out_dir out_wm8 --zip_out final_submission.zip \
  --wm WM_8 --variant P
```

## File reference

| File | Role |
|---|---|
| `approach_real_scheme_forgery.py` | Produces the WM_1 and WM_2 fixes (main venv) |
| `approach_trustmark_forgery.py` | Produces the WM_7 and WM_8 fixes (isolated venv_trustmark) |
| `try_known_schemes.py` | Discovery script: found WM_1=dwtDct (and screened WM_2/8 for dwtDct/dwtDctSvd) |
| `try_known_schemes_v2.py` | Discovery script: found WM_2=rivaGan, with resolution-matched controls |
| `try_trustmark.py` | Discovery script: found WM_7=TrustMark-Q (run with `venv_trustmark`) |
| `evaluate_lpips.py` | Computes real LPIPS/Sqlt for any candidate ZIP (main venv; quality-verification only, not required for reproduction) |
| `latest_best_result.zip` | Pre-fix baseline (0.466486), starting point for the four fixes above |
| `latest_best_results_0901.zip` | The final submitted result (0.901033) |
