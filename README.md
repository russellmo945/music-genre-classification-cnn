# Music Genre Classification System

**Python · PyTorch · Librosa · CNNs · Audio Classification**

A deep learning project that classifies music into **10 genres** using mel-spectrogram images. The project explores spectrogram preprocessing, configurable CNN architectures, hyperparameter tuning, and evaluation on GTZAN and an external music collection.

**Project period: January–April 2024**

## Project highlights

- Prepared approximately **32,000 spectrogram segments** using 32-way slicing and **64 × 64** RGB resizing.
- Explored **15 named hyperparameter configurations**, varying channel widths, dense layers, learning rate, and batch size.
- Selected a CNN with **six convolutional layers**, max pooling, dropout, and a dense classifier.
- Built a **Librosa** pipeline for external audio evaluation using mel-spectrograms.
- Included a **Random Forest baseline**, command-line training and inference, checkpoint metadata, and automated tests.

## Results

| Experiment | Reported accuracy |
| --- | ---: |
| CNN — GTZAN validation | 65.9% |
| CNN — GTZAN test | **69.4%** |
| CNN — external evaluation, reported collection of 100 Tidal tracks | **65.4%** |

**Evaluation note:** GTZAN scores use a slice-level split that can share source tracks across partitions, so they do not measure unseen-track performance. The runnable pipeline uses track-disjoint splits; those scores have not yet been measured. External results are recorded from the project experiment and have not been independently reproduced. See [evaluation details](docs/evaluation.md).

## Architecture and pipeline

```mermaid
flowchart LR
    A[GTZAN spectrograms] --> B[Track split manifest]
    B --> C[64 × 64 image slices]
    C --> D[Six-block CNN]
    D --> E[Genre predictions and metrics]
```

The CNN uses channel widths **12 → 24 → 48 → 96 → 192 → 384**, followed by dense layers **512 → 128 → 64 → 10**. Each convolutional block applies ReLU, max pooling, and dropout. The selected configuration uses SGD with learning rate **0.01**, momentum **0.9**, batch size **64**, and **100 epochs**.

The package supports deterministic track splits, lazy image loading, a fixed class mapping, and both slice-level and track-level evaluation.

## Run locally

Python 3.11 or newer. Run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

The smoke test checks the full workflow using synthetic data. Download GTZAN separately to train a model; pretrained weights are not bundled.

Download the dataset separately and follow [data setup](docs/data.md). Then:

```bash
music-genre prepare --images data/gtzan/images_original --out data/splits.json
music-genre train --images data/gtzan/images_original --manifest data/splits.json --config configs/model_set13.json --out runs/cnn
music-genre evaluate --images data/gtzan/images_original --manifest data/splits.json --checkpoint runs/cnn/best.pt --out runs/cnn/test.json
music-genre baseline --csv data/gtzan/features_3_sec.csv --manifest data/splits.json --out runs/baseline.json
music-genre predict --image data/gtzan/images_original/blues/blues00000.png --checkpoint runs/cnn/best.pt
```

Use `--device cuda` on train, evaluate, and predict for a compatible GPU installation. The selected configuration runs for 100 epochs; `--epochs 1` is useful for checking your setup, not reporting model quality. Evaluate the test split only after fixing model and preprocessing choices.

Optional local-audio conversion:

```bash
python -m pip install -e '.[audio]'
music-genre spectrogram --audio /path/to/your/audio.wav --out data/example.png --offset 0 --duration 30
music-genre predict --image data/example.png --checkpoint runs/cnn/best.pt
```

Audio must contain the full requested interval. New audio rendering can differ from GTZAN image rendering, so this is an inference utility, not a validated deployment demo. Softmax scores are not calibrated confidence estimates.

## Explore the project

| File / directory | Purpose |
| --- | --- |
| [Project walkthrough](notebooks/02_project_walkthrough.ipynb) | Guided introduction to data, architecture, and evaluation |
| [Experiment notebook](notebooks/01_original_experiment.ipynb) | CNN experiments, preprocessing, and external-audio evaluation |
| [Evaluation details](docs/evaluation.md) | Split methodology, metric interpretation, and limitations |
| [Model card](docs/model-card.md) | Intended use, architecture, limitations, and reproducibility |
| [Recorded results](results/historical/metrics.json) | Recorded values and source-file fingerprints |
| `src/music_genre/` | Data, model, training, evaluation, baseline, and CLI |
| `tests/` and `scripts/smoke_test.py` | Correctness checks and executable integration test |
