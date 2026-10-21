# Data setup

The project uses the [GTZAN music genre dataset on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). Download it yourself under the provider's terms. Do not commit recordings or dataset archives.

Expected local layout:

```text
data/gtzan/
  images_original/
    blues/blues00000.png
    classical/classical00000.png
    ...
    rock/rock00000.png
  features_3_sec.csv
```

Canonical class order: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock. Image filenames such as `blues00000.png` and numeric-feature filenames such as `blues.00000.0.wav` map to the same source identity, `blues/00000`.

`prepare` verifies image decoding, rejects exact duplicate image bytes, and allocates approximately 60% / 20% / 20% of tracks **within each genre** to train / validation / test. Integer rounding is recorded in the actual manifest. At least five images per genre are required. Every slice and numeric segment inherits its source track's partition. Preserve the generated manifest: evaluation requires the exact same file bytes as training.

Each image is split into 32 equal-width vertical pieces and resized to RGB 64 × 64. Images are decoded on demand. The default crop removes near-white outer margins. This heuristic does not reliably remove plot labels or axes: inspect your dataset's images before running experiments, and change preprocessing consistently if needed. Blank images and images narrower than the slice count fail explicitly.

The nominal dataset contains 1,000 tracks; the recorded experiment contains 31,968 slices, consistent with 999 loaded images at 32 slices each. This repository does not guess which source image was absent. Your manifest is the authority for your run's counts.

The feature baseline uses the same manifest and excludes CSV rows whose source tracks are absent from it, reporting that count. It requires numeric, finite features and complete CSV coverage for all manifest tracks. No scaler is needed for the tree baseline.

Source-track splitting is not artist-disjoint evaluation or audio-level deduplication. Exact image hashes cannot detect re-encoded copies, alternate excerpts of a recording, or artist overlap. See the [GTZAN analysis by Sturm](https://arxiv.org/abs/1306.1461) before interpreting results as broad generalization.
