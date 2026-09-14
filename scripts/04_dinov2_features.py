"""Extract DINOv2 features for each cell crop.

The results deck's Phase 2 replaced seven brightness statistics with ~1,500
DINOv2 features and kept a linear head. This reproduces that feature step so
the same model can be run against the previous and the rebuilt labels.

Uses ViT-S/14 (384-dim) by default rather than the deck's ~1,500 dims. The
comparison being made is label-vs-label with the feature extractor held fixed,
so the backbone size does not affect the conclusion -- but absolute R^2 is not
directly comparable to the deck's figures. Pass ``--model dinov2_vitb14`` (768)
or ``dinov2_vitl14`` (1024) to get closer.

Crops are 96x96; DINOv2 needs a multiple of its patch size (14), so they are
resized to 98x98.

Usage::

    uv run python scripts/03_build_dataset.py --save-crops
    uv run python scripts/04_dinov2_features.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=Path,
                    default=REPO / "data" / "processed" / "crops.npz")
    ap.add_argument("--out", type=Path,
                    default=REPO / "data" / "processed" / "dinov2.npz")
    ap.add_argument("--model", default="dinov2_vits14")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--size", type=int, default=98, help="multiple of 14")
    args = ap.parse_args()
    # Resolve so a relative --out still prints cleanly below.
    args.crops = args.crops.resolve()
    args.out = args.out.resolve()

    if not args.crops.exists():
        sys.exit(f"{args.crops} not found -- run "
                 "scripts/03_build_dataset.py --save-crops first")

    import torch
    import torch.nn.functional as F

    d = np.load(args.crops, allow_pickle=False)
    rgb = d["rgb"]
    names = d["names"]
    print(f"crops: {rgb.shape}  ({rgb.dtype})")

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    model = torch.hub.load("facebookresearch/dinov2", args.model, verbose=False)
    model.eval().to(device)
    print(f"model: {args.model}, embed dim {model.embed_dim}")

    feats: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(rgb), args.batch):
            chunk = rgb[i : i + args.batch].astype(np.float32) / 255.0
            chunk = (chunk - IMAGENET_MEAN) / IMAGENET_STD
            t = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(device)
            if t.shape[-1] != args.size:
                t = F.interpolate(
                    t, size=(args.size, args.size),
                    mode="bilinear", align_corners=False,
                )
            out = model(t)                       # CLS token
            feats.append(out.float().cpu().numpy())
            if (i // args.batch) % 10 == 0:
                print(f"  {min(i + args.batch, len(rgb))}/{len(rgb)}")

    X = np.concatenate(feats).astype(np.float32)
    print(f"features: {X.shape}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, X=X, names=names,
        label_new=d["label_new"], label_old=d["label_old"],
    )
    print(f"wrote {args.out.relative_to(REPO)} "
          f"({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
