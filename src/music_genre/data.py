"""Deterministic, source-track-disjoint image manifests and lazy slicing."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops
from torch.utils.data import Dataset

GENRES = ("blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock")
SPLITS = ("train", "validation", "test")


def track_id(filename: str, genre: str) -> str:
    """Match GTZAN images (blues00000.png) and CSV rows (blues.00000.0.wav)."""
    match = re.match(rf"^{re.escape(genre)}[._-]?(\d{{5}})(?:[._-]|$)", Path(filename).stem)
    if not match:
        raise ValueError(f"Expected a GTZAN track filename, got {filename!r} for {genre!r}")
    return f"{genre}/{match.group(1)}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(root: Path, seed: int = 42) -> dict:
    root = Path(root)
    records = []
    seen_hashes = set()
    rng = random.Random(seed)
    for genre in GENRES:
        paths = sorted(p for p in (root / genre).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if len(paths) < 5:
            raise ValueError(f"{genre}: need at least 5 source-track images in {root / genre}")
        group = []
        for path in paths:
            with Image.open(path) as img:
                img.verify()
            digest = sha256(path)
            if digest in seen_hashes:
                raise ValueError(f"Duplicate image bytes: {path}; resolve duplicates before splitting")
            seen_hashes.add(digest)
            group.append({"path": path.relative_to(root).as_posix(), "genre": genre,
                          "track_id": track_id(path.name, genre), "sha256": digest})
        rng.shuffle(group)
        train_end, val_end = int(.6 * len(group)), int(.8 * len(group))
        for i, record in enumerate(group):
            record["split"] = "train" if i < train_end else "validation" if i < val_end else "test"
            records.append(record)
    manifest = {"schema_version": 1, "seed": seed, "classes": list(GENRES),
                "split_unit": "source_track", "records": sorted(records, key=lambda r: r["track_id"])}
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> None:
    if manifest.get("classes") != list(GENRES) or manifest.get("split_unit") != "source_track":
        raise ValueError("Manifest must use canonical classes and source-track splitting")
    seen = set()
    counts = {(g, s): 0 for g in GENRES for s in SPLITS}
    for record in manifest["records"]:
        identity = record["track_id"]
        if identity in seen:
            raise ValueError(f"Duplicate source track in manifest: {identity}")
        seen.add(identity)
        if (record["genre"], record["split"]) not in counts:
            raise ValueError("Unknown class or split in manifest")
        if track_id(record["path"].split("/")[-1], record["genre"]) != identity:
            raise ValueError(f"Track identity does not match path: {identity}")
        path = Path(record["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Manifest image paths must stay inside the supplied image root")
        counts[(record["genre"], record["split"])] += 1
    if not all(counts.values()):
        raise ValueError("Every class must have at least one track in every split")


def read_manifest(path: Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    validate_manifest(manifest)
    return manifest


def verify_images(root: Path, manifest: dict) -> None:
    for record in manifest["records"]:
        path = Path(root) / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"Missing or changed source image: {record['path']}")


def crop_white_border(image: Image.Image) -> Image.Image:
    """Remove near-white margins, without assuming one image size or fixed offsets."""
    image = image.convert("RGB")
    difference = ImageChops.difference(image, Image.new("RGB", image.size, "white"))
    mask = difference.convert("L").point(lambda value: 255 if value > 10 else 0)
    box = mask.getbbox()
    if box is None:
        raise ValueError("Cannot classify an all-white image")
    return image.crop(box)


def image_slices(image: Image.Image, slices: int = 32, size: int = 64, crop: bool = True) -> torch.Tensor:
    image = crop_white_border(image) if crop else image.convert("RGB")
    if slices < 1 or size < 1 or image.width < slices:
        raise ValueError("Image width must be at least the number of slices; sizes must be positive")
    tensors = []
    for i in range(slices):
        piece = image.crop((i * image.width // slices, 0, (i + 1) * image.width // slices, image.height))
        piece = piece.resize((size, size), Image.Resampling.BILINEAR)
        tensors.append(torch.from_numpy(np.asarray(piece, dtype=np.float32).copy()).permute(2, 0, 1) / 255.)
    return torch.stack(tensors)


class SpectrogramSlices(Dataset):
    """Decode only the requested slice, keeping the full dataset out of memory."""

    def __init__(self, root: Path, records: list[dict], slices: int = 32, size: int = 64, crop: bool = True):
        if not records or slices < 1 or size < 1:
            raise ValueError("A nonempty dataset and positive dimensions are required")
        self.root, self.records, self.slices, self.size, self.crop = Path(root), records, slices, size, crop

    def __len__(self):
        return len(self.records) * self.slices

    def __getitem__(self, index):
        track_index, slice_index = divmod(index, self.slices)
        record = self.records[track_index]
        with Image.open(self.root / record["path"]) as original:
            image = crop_white_border(original) if self.crop else original.convert("RGB")
            if image.width < self.slices:
                raise ValueError(f"Image too narrow for {self.slices} slices: {record['path']}")
            piece = image.crop((slice_index * image.width // self.slices, 0,
                                (slice_index + 1) * image.width // self.slices, image.height))
            piece = piece.resize((self.size, self.size), Image.Resampling.BILINEAR)
            tensor = torch.from_numpy(np.asarray(piece, dtype=np.float32).copy()).permute(2, 0, 1) / 255.
        return tensor, GENRES.index(record["genre"]), record["track_id"]
