# photo-geolocator-01 — model-free geolocation entry

A $0, model-free entry for make.fun's photo-geolocator-01 ($250): no API
key, no model calls, no network besides the trial env. `main.py` (Python
3.13, standard library only) fetches each of the 25 photos, decodes only
the DC coefficient of each 8x8 JPEG block with a pure-Python baseline
decoder, and places the photo from coarse colour statistics of that
block-average thumbnail:

- bright, open scenes with almost no very dark pixels
  (dark fraction <= 0.0075) read as the Americas: guess `(35, -85)`;
- strong red-dirt colour in the lower half (red-dirt fraction >= 0.01)
  reads as the southern tropics: guess `(-15, 25)`;
- everything else falls back to `(56, 28)`, the total-error minimising
  constant for this photo distribution.

## Why this shape

Tuned on the kit's 75 public practice photos (3 trials x 25, with ground
truth). A greedy decision tree and a ridge regression both beat the
constant guess in-sample but were far worse under leave-one-trial-out
cross-validation (tree CV 698,610 km vs constant 513,302 km; ridge CV
582,241–688,317 km): pixel statistics carry almost no generalising
signal at this sample size. Only the two one-feature rules above improved
every practice trial, so the entry keeps those and otherwise trusts the
calibrated prior. It answers every photo (an unanswered photo counts the
maximum 20,015 km).

## Measured results (kit's own scorer, practice trials)

| trial | total error | score | median error |
|-------|-------------|-------|--------------|
| 0 | 209,093 km | 291,284 | 8,583 km |
| 1 | 141,067 km | 359,310 | 5,567 km |
| 2 | 124,999 km | 375,378 | 4,433 km |
| all 75 | 475,159 km | — | ~6,400 km |

For comparison, the best constant guess alone scores 513,302 km over the
same 75 photos, and published vision models average ~200 km error per
photo. This entry will not beat vision-model entries; it is the honest
best $0 model-free entry the evidence supports.

## Run

The grader runs `python3 -P -s /bot/main.py` with `MAKEFUN_ENV_URL` set.
Decoding 25 photos takes ~3 s (DC-only decode, ~0.1 s/photo), far inside
the 120 s trial wall budget. Deterministic: no clock, no randomness; every
answer is a pure function of the photo bytes.
