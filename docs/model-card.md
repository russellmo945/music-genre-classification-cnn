# Model card

**Task:** single-label classification of spectrogram images into the ten GTZAN genres. **Intended use:** education, reproducible experimentation, and portfolio discussion. **Not established:** production reliability, broad music taxonomy coverage, calibrated confidence, or generalization to unseen artists.

## Selected architecture

Input: RGB 64 × 64. Six blocks of 3 × 3 convolution (padding 1), ReLU, 2 × 2 max pooling, and dropout 0.3. Channels: 12, 24, 48, 96, 192, 384. The spatial representation becomes 1 × 1. Dense layers: 384 → 512 → 128 → 64 → 10, with ReLU between hidden layers and raw output logits. Loss: cross-entropy. Optimizer: SGD, learning rate 0.01, momentum 0.9. Historical selected settings: batch size 64, 100 epochs.

`configs/model_set13.json` retains that topology and the main optimization settings. It does **not** reproduce the historical experiment exactly: splitting, cropping, seeds, evaluation mode, and checkpoint selection have deliberately changed. Historical results must not be attached to new weights.

## Reproducibility

Run commands from the repository root. Dependencies are bounded in `pyproject.toml`; the tested environment is recorded in `docs/validation.json`. Training seeds Python, NumPy, and PyTorch before model construction. CUDA backend settings reduce nondeterminism but do not guarantee identical results across devices or library versions. Save the run's config, environment, manifest, checkpoint, history, and evaluation JSON together.

Inference requires a checkpoint created by this package. Historical checkpoints were not supplied. Loading uses PyTorch's weights-only mode. Track scores average slice softmax vectors and are not calibrated probability estimates. The image input must use compatible rendering and preprocessing.

## Limitations

GTZAN is small and has documented duplication and labeling issues. A genre label is an imperfect simplification of musical style. Image color maps, borders, labels, and rendering settings can become shortcuts. Local audio conversion uses 128 mel bands, an upper frequency capped at 8 kHz / Nyquist, log-power scaling, and an axis-free magma image; it is not pixel-identical to the original plotted spectrogram export. Genre and recording-domain shift remain unmeasured.

## Publication notes

Keep this repository private until the team and course-sharing constraints are resolved. Add Russell's specific contributions only after confirming them; this package credits all four original team members. Dataset files, private paths in outputs, student IDs, graded annotations, and music recordings are excluded from the portfolio package. The archival notebook retains original source code, including legacy Colab paths, for transparency.
