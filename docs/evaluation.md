# Evaluation methodology and results

This project evaluates a six-convolutional-layer CNN on GTZAN spectrograms and an external music collection, with a Random Forest baseline on numeric audio features. Exact recorded metrics and source-file fingerprints are in [results/experiments/metrics.json](../results/experiments/metrics.json).

## Reported results

| Experiment | Accuracy | Evaluation unit |
| --- | ---: | --- |
| CNN — GTZAN training | 77.52% | Spectrogram slice |
| CNN — GTZAN validation | 65.95% | Spectrogram slice |
| CNN — GTZAN test | 69.35% | Spectrogram slice |
| Random Forest — validation | 79.38% | Numeric audio segment |
| CNN — external music | 65.35% | Reported slice-based evaluation |

The README rounds the test and external scores to **69.4%** and **65.4%**. The recorded CNN dataset contains **31,968 slices**, consistent with 999 images at 32 slices each; “approximately 32,000” describes that scale. The separately recorded test macro-F1 is **0.6914**, not an accuracy.

## How to interpret the experiments

The [experiment notebook](../notebooks/01_experiments.ipynb) partitions GTZAN after generating slices, using a 60/20/20 split. Slices from a single song can therefore appear in more than one partition. The reported scores describe this slice-level experiment and do not establish performance on unseen songs.

The notebook's scoring functions do not consistently enable evaluation mode, so dropout can remain active. Accuracy and macro-F1 were computed in separate passes. Its class indices also depend on directory order. These details matter when interpreting the saved results or rerunning the notebook.

The Random Forest uses 1,000 trees, maximum depth 10, and an 80/10/10 split of numeric segments. Segments from the same source track can cross partitions, and numeric scaling is fitted before splitting. Its validation score is not a controlled comparison with the CNN's score.

The project report describes an external collection of **100 Tidal tracks**, using intervals 0–30 and 60–90 seconds and a Librosa mel-spectrogram pipeline. The recorded external accuracy is 65.35%. External assets, the evaluated sample count, and a portable class mapping are not bundled, so this result has not been independently reproduced.

## Package evaluation workflow

The command-line package uses a source-track manifest shared by the CNN and Random Forest. Its track-disjoint scores have not yet been measured on GTZAN; the table above must not be interpreted as results from this protocol.

1. Assign source tracks within each genre to approximately 60% train, 20% validation, and 20% test. Every slice or numeric segment inherits its track's split.
2. Freeze the manifest, class mapping, and source-image hashes.
3. Seed Python, NumPy, and PyTorch before model construction. Train with SGD and select the checkpoint using validation **track macro-F1**.
4. After model selection, evaluate untouched test tracks with `eval()` and inference mode. Compute all metrics from the same pass.
5. Report slice and track accuracy, macro-F1, class-wise metrics, confusion matrices, and counts. Track predictions average slice softmax vectors.
6. Fit the Random Forest using the same track partitions. Trees do not require feature scaling.

Each run saves its configuration, environment, manifest, training history, and validation metrics. Checkpoints retain the class mapping, model settings, selection metric, and manifest hash. Evaluation rejects changed source images or a different manifest.

Preprocessing is explicit in the package: optional near-white border cropping, vertical slicing, and resizing are applied consistently during training and prediction. The notebook includes a separate cropping experiment, but its training loader reads the `images_original` directory.

## Validation and next steps

Unit and synthetic integration tests check pipeline execution, split integrity, preprocessing, checkpoint loading, and evaluation behavior. They do not measure music-classification performance. The GitHub Actions test run on October 3, 2026 passed; see [validation records](validation.json).

Next experiments should benchmark the track-disjoint protocol, repeat across seeds, inspect per-genre errors, standardize audio rendering, and check artist overlap and duplicate recordings. Track-disjoint splitting alone does not establish artist-disjoint generalization.

References: [PyTorch model loading and evaluation](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html), [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html), and [Sturm's GTZAN analysis](https://arxiv.org/abs/1306.1461).
