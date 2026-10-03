"""Training, checkpoint metadata, and held-out evaluation."""

import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import GENRES, SpectrogramSlices, read_manifest, sha256, verify_images
from .metrics import evaluate
from .model import GenreCNN


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dataset(root, manifest, split, config):
    records = [r for r in manifest["records"] if r["split"] == split]
    return SpectrogramSlices(root, records, config["slices"], config["model"]["input_size"], config["crop_white_border"])


def train(images, manifest_path, config_path, output, device="cpu", epochs=None):
    config = json.loads(Path(config_path).read_text())
    if epochs is not None:
        config["epochs"] = epochs
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["learning_rate"] <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive")
    if config["model"]["num_classes"] != len(GENRES):
        raise ValueError("GTZAN requires ten classes")
    manifest = read_manifest(manifest_path)
    verify_images(images, manifest)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty to prevent overwriting a run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config["seed"])
    model = GenreCNN(**config["model"]).to(device)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(dataset(images, manifest, "train", config), batch_size=config["batch_size"],
                              shuffle=True, generator=generator)
    val_loader = DataLoader(dataset(images, manifest, "validation", config), batch_size=config["batch_size"])
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"])
    write_json(output / "config.json", config)
    write_json(output / "manifest.json", manifest)
    write_json(output / "environment.json", {"python": platform.python_version(), "torch": str(torch.__version__),
                                            "numpy": np.__version__, "device": device})
    history, best = [], -1.
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = total_correct = count = 0
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            count += targets.numel()
            total_loss += loss.item() * targets.numel()
            total_correct += (logits.argmax(1) == targets).sum().item()
        val = evaluate(model, val_loader, list(GENRES), device)
        row = {"epoch": epoch, "train_loss": total_loss / count, "train_accuracy": total_correct / count,
               "validation_loss": val["loss"], "validation_slice_accuracy": val["slice"]["accuracy"],
               "validation_track_macro_f1": val["track"]["macro_f1"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["track"]["macro_f1"] > best:
            best = val["track"]["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "config": config, "classes": list(GENRES),
                        "manifest_sha256": sha256(Path(manifest_path)), "epoch": epoch,
                        "selection_metric": "validation_track_macro_f1", "selection_value": best}, output / "best.pt")
            write_json(output / "best_validation.json", val)
        write_json(output / "history.json", history)
    return output / "best.pt"


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["classes"] != list(GENRES):
        raise ValueError("Checkpoint class mapping does not match the canonical GTZAN mapping")
    model = GenreCNN(**checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def evaluate_checkpoint(images, manifest_path, checkpoint_path, split="test", device="cpu"):
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if sha256(Path(manifest_path)) != checkpoint["manifest_sha256"]:
        raise ValueError("Use the exact manifest used for training; a changed split invalidates evaluation")
    manifest = read_manifest(manifest_path)
    verify_images(images, manifest)
    loader = DataLoader(dataset(images, manifest, split, checkpoint["config"]),
                        batch_size=checkpoint["config"]["batch_size"])
    result = evaluate(model, loader, checkpoint["classes"], device)
    result.update({"split": split, "checkpoint_epoch": checkpoint["epoch"],
                   "manifest_sha256": checkpoint["manifest_sha256"], "status": "new_run_not_historical"})
    return result
