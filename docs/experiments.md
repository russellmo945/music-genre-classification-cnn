# CNN experiments

The experiment notebook defines 15 named model configurations. All use six convolutional layers and ten output genres. Channel widths double after each convolutional block. Model set 1 also includes 60-, 100-, and 200-epoch runs.

| Configuration | Initial channels | Dense hidden layers | Learning rate | Batch size | Epochs |
| --- | ---: | --- | ---: | ---: | --- |
| model_set1 | 6 | 1024 → 512 → 128 → 64 | 0.01 | 32 | 60, 100, 200 |
| model_set2 | 12 | 1024 → 512 → 128 → 64 | 0.01 | 32 | 100 |
| model_set3 | 6 | 512 → 128 → 64 | 0.01 | 32 | 100 |
| model_set4 | 6 | 1024 → 512 → 128 → 64 | 0.001 | 32 | 100 |
| model_set5 | 6 | 1024 → 512 → 128 → 64 | 0.01 | 64 | 100 |
| model_set6 | 12 | 512 → 128 → 64 | 0.01 | 32 | 100 |
| model_set7 | 12 | 1024 → 512 → 128 → 64 | 0.001 | 32 | 100 |
| model_set8 | 12 | 1024 → 512 → 128 → 64 | 0.01 | 64 | 100 |
| model_set9 | 6 | 512 → 128 → 64 | 0.001 | 32 | 100 |
| model_set10 | 6 | 512 → 128 → 64 | 0.01 | 64 | 100 |
| model_set11 | 6 | 1024 → 512 → 128 → 64 | 0.001 | 64 | 100 |
| model_set12 | 12 | 512 → 128 → 64 | 0.001 | 32 | 100 |
| model_set13 (selected) | 12 | 512 → 128 → 64 | 0.01 | 64 | 100 |
| model_set14 | 6 | 512 → 128 → 64 | 0.001 | 64 | 100 |
| model_set15 | 12 | 512 → 128 → 64 | 0.001 | 64 | 100 |

## Selected configuration

**Model set 13** uses initial channels 12, dense hidden layers 512 → 128 → 64, learning rate 0.01, batch size 64, and 100 epochs. The package configuration is [configs/model_set13.json](../configs/model_set13.json).

Recorded test accuracy is **69.4%** under the notebook’s slice-level protocol. See [evaluation methodology](evaluation.md) before comparing it with a new run. Full per-configuration score logs are not bundled; this table documents settings, not a ranking of model performance.

The [experiment notebook](../notebooks/01_experiments.ipynb) contains the model definitions and training calls. The command-line pipeline described in the README trains and evaluates using a fixed source-track manifest.
