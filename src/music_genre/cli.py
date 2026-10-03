"""Small CLI for data preparation, training, evaluation and image inference."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .data import build_manifest, image_slices
from .experiment import evaluate_checkpoint, load_checkpoint, train, write_json


def parser():
    root = argparse.ArgumentParser(description="APS360 music genre classification")
    subs = root.add_subparsers(dest="command", required=True)
    prepare = subs.add_parser("prepare", help="Create a deterministic source-track split")
    prepare.add_argument("--images", required=True, type=Path)
    prepare.add_argument("--out", required=True, type=Path)
    prepare.add_argument("--seed", type=int, default=42)
    training = subs.add_parser("train", help="Train CNN; select by validation track macro-F1")
    training.add_argument("--images", required=True, type=Path)
    training.add_argument("--manifest", required=True, type=Path)
    training.add_argument("--config", type=Path, default=Path("configs/model_set13.json"))
    training.add_argument("--out", required=True, type=Path)
    training.add_argument("--epochs", type=int)
    training.add_argument("--device", default="cpu")
    evaluation = subs.add_parser("evaluate", help="Evaluate a saved CNN with the original split manifest")
    evaluation.add_argument("--images", required=True, type=Path)
    evaluation.add_argument("--manifest", required=True, type=Path)
    evaluation.add_argument("--checkpoint", required=True, type=Path)
    evaluation.add_argument("--out", required=True, type=Path)
    evaluation.add_argument("--split", choices=["validation", "test"], default="test")
    evaluation.add_argument("--device", default="cpu")
    baseline = subs.add_parser("baseline", help="Fit Random Forest on numeric features using the same track split")
    baseline.add_argument("--csv", required=True, type=Path)
    baseline.add_argument("--manifest", required=True, type=Path)
    baseline.add_argument("--out", required=True, type=Path)
    baseline.add_argument("--trees", type=int, default=1000)
    prediction = subs.add_parser("predict", help="Predict one spectrogram using a locally trained checkpoint")
    prediction.add_argument("--image", required=True, type=Path)
    prediction.add_argument("--checkpoint", required=True, type=Path)
    prediction.add_argument("--device", default="cpu")
    audio = subs.add_parser("spectrogram", help="Convert a local audio interval to a mel-spectrogram PNG")
    audio.add_argument("--audio", required=True, type=Path)
    audio.add_argument("--out", required=True, type=Path)
    audio.add_argument("--offset", type=float, default=0.)
    audio.add_argument("--duration", type=float, default=30.)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare":
        result = build_manifest(args.images, args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists():
            raise ValueError("Manifest exists; choose a new path to preserve experiment provenance")
        write_json(args.out, result)
        print(f"Saved {len(result['records'])} source tracks to {args.out}")
    elif args.command == "train":
        train(args.images, args.manifest, args.config, args.out, args.device, args.epochs)
    elif args.command == "evaluate":
        result = evaluate_checkpoint(args.images, args.manifest, args.checkpoint, args.split, args.device)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, result)
        print(json.dumps({k: result[k] for k in ["split", "loss", "track"]}, indent=2))
    elif args.command == "baseline":
        from .baseline import run_baseline
        result = run_baseline(args.csv, args.manifest, args.trees)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, result)
        print(f"Saved baseline metrics to {args.out}")
    elif args.command == "predict":
        model, checkpoint = load_checkpoint(args.checkpoint, args.device)
        config = checkpoint["config"]
        with Image.open(args.image) as image:
            inputs = image_slices(image, config["slices"], config["model"]["input_size"], config["crop_white_border"])
        with torch.inference_mode():
            probs = model(inputs.to(args.device)).softmax(1).mean(0).cpu().tolist()
        ranked = sorted(zip(checkpoint["classes"], probs), key=lambda x: x[1], reverse=True)
        print(json.dumps({"predicted_genre": ranked[0][0], "scores": dict(ranked),
                          "note": "Mean softmax scores, not calibrated confidence probabilities"}, indent=2))
    elif args.command == "spectrogram":
        from .audio import spectrogram
        print(spectrogram(args.audio, args.out, args.offset, args.duration))


if __name__ == "__main__":
    main()
