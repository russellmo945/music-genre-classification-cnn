# Repository guide

**Repository:** `music-genre-classification-cnn`  
**Project:** Music Genre Classification System  
**Default branch:** `main`

PyTorch CNN for ten-class music genre classification, with spectrogram preprocessing, hyperparameter experiments, a Random Forest baseline, and a training and inference CLI.

## Where to start

- [README](../README.md): project overview, reported results, architecture, and commands.
- [Data setup](data.md): GTZAN directory layout, class mapping, and preprocessing.
- [Experiments](experiments.md): the 15 named configurations and selected settings.
- [Project walkthrough](../notebooks/02_project_walkthrough.ipynb): guided model and evaluation examples.
- [Evaluation methodology](evaluation.md): score interpretation and the package's evaluation protocol.
- [Model card](model-card.md): architecture, reproducibility, and limitations.

## Local checks

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

The smoke test uses temporary synthetic images and feature rows. Real-data training requires a separately downloaded dataset. Run outputs belong under `runs/`; data and checkpoints are excluded by `.gitignore`.

The GitHub Actions workflow runs unit and integration checks on pushes and pull requests. The initial hosted run passed on October 3, 2026; [validation.json](validation.json) records that run and the local validation environment.
