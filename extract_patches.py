#!/usr/bin/env python3
"""
extract_patches.py

Extract multiple patches from a TIFF/WSI and save them to disk.

Examples
--------
# Grid patches
python extract_patches.py \
  --input camelyon17/images/patient_000_node_0.tif \
  --outdir patches \
  --mode grid \
  --patch-size 256 \
  --stride 256 \
  --level 2 \
  --format png

# Random patches (skip mostly-white/background)
python extract_patches.py \
  --input camelyon17/images/patient_000_node_0.tif \
  --outdir patches_random \
  --mode random \
  --patch-size 256 \
  --num-patches 2000 \
  --level 0 \
  --min-tissue 0.15 \
  --seed 123 \
  --format jpg
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def try_import_openslide():
    try:
        import openslide  # type: ignore
        return openslide
    except Exception:
        return None


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def save_patch(arr_rgb: np.ndarray, out_path: Path, fmt: str, jpg_quality: int) -> None:
    img = Image.fromarray(arr_rgb)
    if fmt.lower() in {"jpg", "jpeg"}:
        img.save(out_path, quality=jpg_quality, optimize=True)
    else:
        img.save(out_path)


def tissue_fraction_simple(rgb: np.ndarray, white_thresh: int = 220) -> float:
    """
    Very simple tissue detector:
    Count non-white pixels (all channels < white_thresh).
    Returns fraction of pixels considered tissue.
    """
    if rgb.size == 0:
        return 0.0
    nonwhite = np.any(rgb < white_thresh, axis=2)
    return float(nonwhite.mean())


def read_region_openslide(slide, x: int, y: int, level: int, patch_size: int) -> np.ndarray:
    """
    OpenSlide returns RGBA; convert to RGB uint8.
    """
    region = slide.read_region((x, y), level, (patch_size, patch_size))
    region = region.convert("RGB")
    return np.asarray(region, dtype=np.uint8)


def extract_grid(
    slide,
    outdir: Path,
    patch_size: int,
    stride: int,
    level: int,
    fmt: str,
    min_tissue: float,
    jpg_quality: int,
    max_patches: Optional[int],
) -> int:
    level_w, level_h = slide.level_dimensions[level]
    # Coordinates are in level-0 reference frame for OpenSlide read_region.
    downsample = float(slide.level_downsamples[level])

    count = 0
    y = 0
    while y + patch_size <= level_h:
        x = 0
        while x + patch_size <= level_w:
            x0 = int(x * downsample)
            y0 = int(y * downsample)

            rgb = read_region_openslide(slide, x0, y0, level, patch_size)

            if min_tissue > 0:
                tf = tissue_fraction_simple(rgb)
                if tf < min_tissue:
                    x += stride
                    continue

            out_path = outdir / f"patch_L{level}_x{x}_y{y}.{fmt}"
            save_patch(rgb, out_path, fmt, jpg_quality)
            count += 1

            if max_patches is not None and count >= max_patches:
                return count

            x += stride
        y += stride
    return count


def extract_random(
    slide,
    outdir: Path,
    patch_size: int,
    num_patches: int,
    level: int,
    fmt: str,
    min_tissue: float,
    jpg_quality: int,
    seed: int,
    max_tries_factor: int = 20,
) -> int:
    rng = random.Random(seed)

    level_w, level_h = slide.level_dimensions[level]
    downsample = float(slide.level_downsamples[level])

    # Sample top-left positions in *level* coordinates, then convert to level-0.
    max_x = max(0, level_w - patch_size)
    max_y = max(0, level_h - patch_size)

    count = 0
    tries = 0
    max_tries = max(num_patches * max_tries_factor, num_patches)

    while count < num_patches and tries < max_tries:
        tries += 1
        x = rng.randint(0, max_x) if max_x > 0 else 0
        y = rng.randint(0, max_y) if max_y > 0 else 0

        x0 = int(x * downsample)
        y0 = int(y * downsample)

        rgb = read_region_openslide(slide, x0, y0, level, patch_size)

        if min_tissue > 0:
            tf = tissue_fraction_simple(rgb)
            if tf < min_tissue:
                continue

        out_path = outdir / f"patch_L{level}_x{x}_y{y}_{count:06d}.{fmt}"
        save_patch(rgb, out_path, fmt, jpg_quality)
        count += 1

    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to .tif/.tiff (WSI preferred)")
    ap.add_argument("--outdir", required=True, help="Output directory for patches")
    ap.add_argument("--mode", choices=["grid", "random"], default="grid")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256, help="Grid stride (only for grid mode)")
    ap.add_argument("--num-patches", type=int, default=1000, help="Random patches (only for random mode)")
    ap.add_argument("--level", type=int, default=0, help="OpenSlide pyramid level (0 is highest resolution)")
    ap.add_argument("--min-tissue", type=float, default=0.0, help="Skip patches with tissue fraction below this (0..1)")
    ap.add_argument("--format", choices=["png", "jpg", "jpeg"], default="png")
    ap.add_argument("--jpg-quality", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-patches", type=int, default=0, help="Stop after N patches in grid mode (0 = no limit)")
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    fmt = "jpg" if args.format == "jpeg" else args.format
    max_patches = None if args.max_patches <= 0 else args.max_patches

    openslide = try_import_openslide()
    if openslide is None:
        raise SystemExit(
            "OpenSlide is not installed. Install it for WSI TIFFs:\n"
            "  pip install openslide-python\n"
            "macOS (brew):\n"
            "  brew install openslide\n"
        )

    slide = openslide.OpenSlide(str(in_path))

    print("Opened:", in_path)
    print("Levels:", slide.level_count)
    print("Level dimensions:", slide.level_dimensions)
    print("Level downsamples:", slide.level_downsamples)

    if args.level < 0 or args.level >= slide.level_count:
        raise SystemExit(f"Invalid --level {args.level}. Slide has {slide.level_count} levels.")

    if args.mode == "grid":
        n = extract_grid(
            slide=slide,
            outdir=outdir,
            patch_size=args.patch_size,
            stride=args.stride,
            level=args.level,
            fmt=fmt,
            min_tissue=args.min_tissue,
            jpg_quality=args.jpg_quality,
            max_patches=max_patches,
        )
    else:
        n = extract_random(
            slide=slide,
            outdir=outdir,
            patch_size=args.patch_size,
            num_patches=args.num_patches,
            level=args.level,
            fmt=fmt,
            min_tissue=args.min_tissue,
            jpg_quality=args.jpg_quality,
            seed=args.seed,
        )

    slide.close()
    print(f"Saved {n} patches to {outdir}")


if __name__ == "__main__":
    main()