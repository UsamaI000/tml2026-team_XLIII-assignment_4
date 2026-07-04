**Matriculation Number(s):** [MATRICULATION NUMBER]
**CMS Team ID:** [CMS TEAM ID]

# Watermark Forgery Attack — Report

## Introduction

This assignment asks us to act as black-box attackers against eight unidentified
image watermarking schemes. For each scheme we are given 25 source images that
all carry the same hidden message, and 25 clean target images that must be made
to carry that same message while remaining visually close to the originals. We
have no access to the watermarking models, their training data, or any
documentation of which scheme was used — only the input/output pairs implied by
the source images themselves. Success is measured by the product of a detection
score $S_{det}=\max(0,(\text{BitAccuracy}-0.5)\cdot 2)$ and a perceptual quality
score $S_{qlt}=e^{-8\cdot\text{LPIPS}}$, so a forgery must simultaneously
recover the embedded message *and* stay visually close to the clean image —
neither alone is sufficient. This is a realistic model of a **copy attack**:
an adversary who can collect a modest number of outputs sharing one embedded
credential (e.g. from one publisher or one API key) tries to stamp that same
credential onto unrelated, potentially malicious content.

## Approach

Our approach went through two qualitatively different phases.

### Phase 1: Blind statistical estimation (copy attack)

We began from the classical watermark copy-attack idea (Kutter &
Voloshynovskiy, 2000, cited in the assignment brief): since the 25 source
images per group share one message but differ in content, a **high-pass
residual** (image minus a Gaussian-blurred copy of itself, isolating
fine-grained content) computed for each source and then combined with a
**per-pixel median across all 25 images** should approximately cancel
image-specific content while reinforcing whatever component is common to all
of them — the watermark. The resulting "fingerprint" was added to the
correspondingly-mapped clean targets.

Key hyperparameters and the reasoning behind each:

- **Gaussian blur $\sigma$** (swept 0.5–10): controls which spatial frequencies
  survive into the residual. Different groups needed markedly different values
  (e.g. WM_1/WM_2/WM_6 preferred $\sigma\approx 6$, WM_3 preferred
  $\sigma\approx 8$–10), consistent with each scheme embedding at a different
  characteristic spatial scale.
- **Perturbation strength** (2–9, robust-normalized before scaling): the
  dominant strength–quality trade-off; tuned per group against real LPIPS to
  spend just enough of the quality budget to move detection without
  needlessly degrading $S_{qlt}$.
- **Aggregation method** (mean / median / trimmed-mean): median was
  consistently best, since it is more robust to any one source image's
  idiosyncratic texture dominating the estimate.
- **Optional frequency band-pass component and texture-adaptive masking**:
  tested to capture periodic/frequency-localized structure and to reallocate
  signal toward busy image regions; in practice, disabling the frequency
  component ("no\_freq" variants) usually helped, indicating the extra band
  was adding noise rather than signal for most groups.

Each of the 8 groups was tuned independently, since no single global
configuration worked well across all of them. This phase used roughly 100
leaderboard submissions and converged on a plateau of **0.4665**, with
diminishing returns on every further hyperparameter variation attempted
(including narrowband FFT peak-locking and wavelet-domain denoising, which
looked promising in local statistical tests but did not transfer to the real
leaderboard score).

### Phase 2: Exact scheme identification

Given the plateau, we tested a different hypothesis: that some of the 8
"unidentified" schemes are simply **standard, publicly available watermarking
implementations** rather than bespoke research methods — plausible for an
academic dataset. For each group, we decoded all 25 same-message sources with
several well-known open-source libraries (`invisible-watermark`'s classical
`dwtDct`/`dwtDctSvd` and neural `rivaGan` methods; `blind_watermark`'s
independently-implemented DWT-DCT-SVD scheme; Meta/Adobe's `TrustMark`) and
checked whether the 25 sources decoded to an unusually **consistent** bit
string relative to a same-resolution clean-image control (which calibrates the
decoder's noise floor on non-watermarked content). A large, stable gap between
source-image and control agreement is a strong, specific signature of a true
scheme match — much stronger evidence than any statistical similarity metric,
since it is an exact repeatable decode, not an approximation.

This identified four matches: **WM\_1 = dwtDct**, **WM\_2 = rivaGan**,
**WM\_7 = TrustMark (variant Q)**, **WM\_8 = TrustMark (variant P)**. For each,
we recovered the exact majority-vote message from the 25 real sources and
re-encoded that *exact* message into the assigned clean targets using the
library's own real encoder network — bypassing TrustMark's optional
error-correction wrapper on both decode and encode so we operate at precisely
the same raw-bit interface used for detection. This is not an approximation of
the scheme; it is the same mechanism the original encoder uses, which is why
round-trip self-decode fidelity for these fixes reached 85–100%, and why each
fix produced a single large score jump rather than the incremental gains
typical of Phase 1. WM\_3, WM\_4, WM\_5 and WM\_6 did not match any of the six
scheme/variant combinations we tried and still rely on the Phase 1 residual
configuration.

## Key Results

| Step | Approach | Public leaderboard score |
|---|---|---|
| Early residual sweeps | Gaussian high-pass, single global config | 0.326 → 0.372 |
| Hybrid residual + frequency | Added band-pass component | 0.385 |
| Per-group adaptive tuning | ~100 experiments, independent config per WM group | **0.4665** (Phase 1 plateau) |
| + WM\_1 identified as `dwtDct` | Exact message recovered & re-encoded | 0.578 (+0.111) |
| + WM\_2 identified as `rivaGan` | Exact message recovered & re-encoded | 0.701 (+0.123) |
| + WM\_7 identified as TrustMark-Q | Exact message recovered & re-encoded | 0.804 (+0.103) |
| + WM\_8 identified as TrustMark-P | Exact message recovered & re-encoded | **0.901** (+0.097) |

The clearest finding is the size of the jump each exact-scheme fix produced:
roughly ten times the gain of an average Phase-1 hyperparameter change, and
achieved in a single submission rather than an iterative search over dozens of
attempts. This is strong empirical evidence that, whenever a copy attack has
even a modest chance of the target being a known public scheme, **checking for
an exact match first dominates statistically estimating a generic
perturbation** — the two approaches are not just different in degree but in
kind: one recovers the true embedding exactly, the other only approximates a
plausible direction for it.

## Conclusion

We successfully forged four of the eight watermarking schemes with
near-perfect fidelity (cross-source decode agreement up to 100%, and
round-trip re-encoding fidelity of 85–100%) using only the 25 example
watermarked images per scheme, off-the-shelf open-source tooling, and no
access to any model internals, training data, or documentation — and achieved
non-trivial forgery success on the remaining four via purely statistical
estimation. This has direct real-world implications. First, it shows that
watermark-based provenance and authenticity systems are only as strong as the
secrecy of the underlying scheme and message: any deployment that reuses a
publicly available watermarking library with a single static, shared message
per model or per release is vulnerable to exactly the attack demonstrated
here, requiring no specialized adversarial ML expertise, only a handful of
example outputs and a short list of common libraries to check against. Second,
even where the exact scheme could not be identified, generic residual
estimation still moved the needle, meaning the *absence* of an identifiable
public scheme is not sufficient protection either — it only raises the bar
from "trivial" to "moderately resourced." Consequently, watermark presence
should not, by itself, be treated as reliable proof of content provenance in
adversarial contexts such as misinformation detection or content authenticity
verification, unless the deployment also uses per-instance or per-user secret
keys (so that no two legitimate outputs share a transferable message) and
ideally cryptographically binds the embedded credential to content-specific
data. Watermarking can still be a useful *soft* signal, but our results show
it should not be relied upon as a hard security guarantee against a motivated
attacker with access to only a small number of example outputs.
