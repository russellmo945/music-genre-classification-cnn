"""One evaluation pass for consistent slice and source-track metrics."""

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def summarize(labels, probabilities, classes):
    predictions = np.argmax(probabilities, axis=1)
    indices = list(range(len(classes)))
    return {"count": len(labels), "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, labels=indices, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=indices).tolist(),
            "per_class": classification_report(labels, predictions, labels=indices, target_names=classes,
                                                output_dict=True, zero_division=0)}


@torch.inference_mode()
def evaluate(model, loader, classes, device="cpu"):
    was_training = model.training
    model.eval()
    probabilities, labels, identities = [], [], []
    loss_sum = 0.
    try:
        for images, targets, tracks in loader:
            logits = model(images.to(device))
            loss_sum += torch.nn.functional.cross_entropy(logits, targets.to(device), reduction="sum").item()
            probabilities.extend(logits.softmax(dim=1).cpu().numpy())
            labels.extend(targets.tolist())
            identities.extend(tracks)
    finally:
        model.train(was_training)
    if not labels:
        raise ValueError("Cannot evaluate an empty loader")
    groups = defaultdict(list)
    group_labels = {}
    for track, label, probs in zip(identities, labels, probabilities):
        if track in group_labels and group_labels[track] != label:
            raise ValueError(f"Conflicting labels for track {track}")
        groups[track].append(probs)
        group_labels[track] = label
    track_labels = [group_labels[t] for t in groups]
    track_probs = [np.mean(groups[t], axis=0) for t in groups]
    return {"loss": loss_sum / len(labels), "classes": classes,
            "slice": summarize(labels, probabilities, classes),
            "track": summarize(track_labels, track_probs, classes),
            "aggregation": "mean of slice softmax probabilities per source track"}
