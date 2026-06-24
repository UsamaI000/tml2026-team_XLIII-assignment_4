# Watermark Forgery Assignment Plan

## Required submission constraints

- Output must be a `.zip` containing exactly `1.png` through `200.png` at the ZIP root.
- No folders, hidden files, model artifacts, or extra outputs inside the leaderboard ZIP.
- Target mapping is fixed: `WM_1 -> 1..25`, `WM_2 -> 26..50`, ..., `WM_8 -> 176..200`.
- Do not edit `task_template.py` or `submission.py`; use them only as reference and for leaderboard upload.

## Approaches to try

### 1. Residual fingerprint transfer — priority/best first

File: `approach_residual_fingerprint_forgery.py`

Why useful: all 25 source images inside each `WM_i` share the same watermark message but have different visual content. A high-pass residual extracts small invisible patterns; robust aggregation across the 25 sources cancels image-specific content and keeps the common watermark signal. Adding this fingerprint to clean targets should preserve visual quality better than direct image blending.

Recommended first commands:

```bash
python approach_residual_fingerprint_forgery.py --zip_file Dataset.zip --strength 6 --zip_out submission_residual_s6.zip
python validate_forgery_submission_zip.py submission_residual_s6.zip
```

Then sweep strength after each hourly leaderboard submission:

```bash
python approach_residual_fingerprint_forgery.py --zip_file Dataset.zip --strength 3 --zip_out submission_residual_s3.zip
python approach_residual_fingerprint_forgery.py --zip_file Dataset.zip --strength 9 --zip_out submission_residual_s9.zip
python approach_residual_fingerprint_forgery.py --zip_file Dataset.zip --strength 12 --zip_out submission_residual_s12.zip
```

Use the smallest strength that gives a leaderboard gain, because the final score multiplies detection strength by visual quality.

### 2. Frequency band-pass fingerprint transfer

File: `approach_frequency_bandpass_forgery.py`

Why useful: if one or more watermarking methods encode the message in periodic or mid/high-frequency components, Gaussian residual extraction may suppress or distort part of the signal. A Fourier-domain band-pass keeps frequency-localized watermark patterns while removing most low-frequency source content.

Commands:

```bash
python approach_frequency_bandpass_forgery.py --zip_file Dataset.zip --strength 4 --low 0.06 --high 0.40 --zip_out submission_freq_s4.zip
python validate_forgery_submission_zip.py submission_freq_s4.zip
```

Then try `--strength 2`, `6`, and `8`; also try `--low 0.03 --high 0.45` if the first result is weak.

### 3. Alpha-blend baseline / sanity check

File: `approach_alpha_blend_baseline.py`

Why useful: direct blending is the simplest copy attack and confirms that the target mapping and ZIP packaging are correct. It is not expected to be visually strong because source image content leaks into the targets.

Commands:

```bash
python approach_alpha_blend_baseline.py --zip_file Dataset.zip --alpha 0.10 --zip_out submission_alpha_010.zip
python validate_forgery_submission_zip.py submission_alpha_010.zip
```

Try `--alpha 0.05`, `0.15`, `0.25`; avoid `0.50` unless only testing the naive template behavior.

## Submission workflow

1. Generate a candidate ZIP with one approach script.
2. Validate it:

```bash
python validate_forgery_submission_zip.py path/to/candidate.zip
```

3. Edit only your local copy of `submission.py` fields for API key and `FILE_PATH` when submitting to the server. Do not put API keys into the code repository.
4. Because submissions have a 60-minute cooldown, test approaches in this order:
   1. residual strength 6
   2. residual strength 9
   3. residual strength 3
   4. frequency strength 4
   5. alpha blend 0.10 only as a sanity check

## Report notes

Explain that the attack is black-box: no watermarking scheme internals are known. The key idea is a copy attack: estimate watermark-specific perturbations from watermarked source examples and apply them to assigned clean targets. Discuss the quality-strength trade-off: larger perturbations improve bit accuracy but reduce LPIPS-based visual quality.
