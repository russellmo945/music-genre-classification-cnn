# Evaluation audit: historical experiment vs. portfolio refactor

The source notebook and original project report were reviewed before packaging. References below use **zero-based cell indices in the supplied notebook**, before the archive's introductory cell was inserted. Source SHA-256 hashes are retained in `results/historical/metrics.json`.

| Finding in the historical code | Consequence | Refactor |
| --- | --- | --- |
| Cell 9 slices images before `random_split` | One track can appear in multiple partitions | Split tracks before any slicing |
| Cell 9 assigns labels from unsorted directory listings | Class indices are not portable across directories | Fixed class order saved with checkpoints |
| Cell 7 crops images, but cell 9 reads `images_original` | Cropped images are not actually used by that dataset | Explicit shared preprocessing configuration |
| Cells 36–40 omit evaluation mode during scoring | Dropout stays active; separate scoring passes disagree | `eval()` plus inference mode; metrics from one pass |
| Cell 40 seeds after constructing the model | Model initialization is not controlled by that seed | Seed before construction |
| Cell 22 scales numeric data before splitting | Preprocessing is fitted using held-out data | No unnecessary scaling for Random Forest |
| Cell 28 implements 80/10/10, despite a 60/20/20 description | Baseline protocol differs from CNN protocol | Shared track manifest for both models |
| Cell 40 saves the final epoch | No best-validation checkpoint selection | Select by validation track macro-F1 |

## Which numbers are supported?

The stored outputs record CNN training accuracy 77.52%, validation accuracy 65.95%, and test accuracy 69.35% (cells 73–74). Cell 73's separately computed macro-F1 is 0.6914. That F1 is **not an accuracy** and comes from a separate pass with dropout potentially active. The notebook's saved slice count is 31,968 (cell 11), rather than the nominal 32,000 described in the report.

The Random Forest validation accuracy is 79.38% (cell 33), with 1,000 trees and maximum depth 10. Its original split operates on numeric segment rows, so segments from the same track can cross partitions there too. It is not a fair comparison against the CNN's historical result.

The external experiment stores 65.35% accuracy (cell 77). The report describes 100 songs, two 30-second intervals per song, and 6,400 slices. The external assets, exact evaluated sample count, and portable class mapping are unavailable; that score cannot be independently verified here. It is labeled historical, not promoted as a reproduced external benchmark.

These limitations affect the strength of the performance claims. They do not erase the original model implementation or experimentation; they motivate the new evaluation protocol.

## New evaluation contract

1. Freeze a class-stratified source-track manifest, including file hashes.
2. Train using only train tracks; select a checkpoint by validation **track macro-F1**.
3. Evaluate once on untouched test tracks after modeling decisions are fixed.
4. Report slice accuracy and macro-F1 alongside track accuracy and macro-F1, class-wise metrics, confusion matrices, and sample counts.
5. Aggregate a track by averaging its slices' softmax vectors before taking the highest-scoring class.
6. Run the Random Forest on the same track partitions; report segment and track metrics separately.

The package stores config, environment, manifest, history, and best validation metrics with each run. A checkpoint carries its class map, config, selection criterion, and original manifest hash. Changed dataset bytes or a changed manifest fail evaluation.

**No new GTZAN or external benchmark has been run for this refactor.** Synthetic integration checks establish that the pipeline executes and enforces its contracts; they do not measure useful music classification.

Further work: identify duplicate recordings and artist overlap; standardize audio-to-image rendering; compare alternative architectures with the fixed split; repeat experiments across seeds; quantify uncertainty; and evaluate a separately sourced, correctly labeled external set.

References: [PyTorch saving/loading guidance](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html), [scikit-learn cross-validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html), and [Sturm's GTZAN analysis](https://arxiv.org/abs/1306.1461).
