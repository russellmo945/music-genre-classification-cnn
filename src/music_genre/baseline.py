"""Random Forest on GTZAN numeric features using the CNN's source-track split."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .data import GENRES, read_manifest, track_id
from .metrics import summarize


def run_baseline(csv_path: Path, manifest_path: Path, trees=1000, seed=42):
    manifest = read_manifest(manifest_path)
    frame = pd.read_csv(csv_path)
    if not {"filename", "label"}.issubset(frame.columns):
        raise ValueError("CSV must contain filename and label columns")
    identities = [track_id(str(f), str(g)) for f, g in zip(frame["filename"], frame["label"])]
    frame = frame.assign(track_id=identities)
    lookup = {r["track_id"]: r["split"] for r in manifest["records"]}
    missing = set(lookup) - set(identities)
    if missing:
        raise ValueError(f"Feature CSV is missing {len(missing)} manifest tracks")
    excluded = sum(t not in lookup for t in identities)
    frame = frame.loc[frame.track_id.isin(lookup)].copy()
    frame["split"] = frame.track_id.map(lookup)
    columns = [c for c in frame if c not in {"filename", "length", "label", "track_id", "split"}]
    if not columns or not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError("Feature columns must be nonempty, numeric and finite")
    train = frame[frame.split == "train"]
    model = RandomForestClassifier(n_estimators=trees, max_depth=10, random_state=seed, n_jobs=-1)
    # Trees do not need scaling: no normalization is fit on held-out data.
    model.fit(train[columns], train.label.map({g: i for i, g in enumerate(GENRES)}))
    output = {"status": "new_run_not_historical", "split_unit": "source_track", "excluded_rows": excluded,
              "feature_columns": columns, "trees": trees, "max_depth": 10, "seed": seed}
    for split in ["validation", "test"]:
        part = frame[frame.split == split]
        probs = model.predict_proba(part[columns])
        labels = part.label.map({g: i for i, g in enumerate(GENRES)}).to_numpy()
        grouped_probs, grouped_labels = [], []
        for identity in sorted(part.track_id.unique()):
            mask = (part.track_id == identity).to_numpy()
            grouped_probs.append(probs[mask].mean(axis=0))
            grouped_labels.append(labels[mask][0])
        output[split] = {"segment": summarize(labels, probs, list(GENRES)),
                         "track": summarize(grouped_labels, grouped_probs, list(GENRES))}
    return output
