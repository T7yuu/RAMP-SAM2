"""Lightweight functional checks for conservative retrieved-prior transfer.

Run from this directory:
python test_prior_transfer.py
"""
from pathlib import Path
import json

import numpy as np
from PIL import Image

from prior_transfer import (
    bbox_from_mask,
    is_prior_geometry_compatible,
    load_resized_prior,
    prepare_aligned_prior,
)


KB_PATH = Path(__file__).with_name("unified_medical_kb.json")


def first_existing_prior():
    for item in json.loads(KB_PATH.read_text(encoding="utf-8")):
        mask_path = Path(item.get("mask_path", ""))
        if mask_path.is_file():
            return mask_path
    raise RuntimeError("No readable prior mask path was found in the knowledge base.")


def main():
    prior_path = first_existing_prior()
    prior_image = Image.open(prior_path).convert("L")
    width, height = prior_image.size
    prior_mask, prior_box = load_resized_prior(str(prior_path), (width, height))
    assert prior_mask is not None and prior_box is not None, "Prior mask should load with a valid box."
    assert bbox_from_mask(prior_mask) is not None, "Prior must contain foreground."

    # Self-geometry is compatible. A deliberately displaced query box is rejected.
    assert is_prior_geometry_compatible(str(prior_path), prior_box, (width, height))
    shifted_box = np.array([0, 0, max(1, width // 8), max(1, height // 8)], dtype=np.float32)
    assert not is_prior_geometry_compatible(str(prior_path), shifted_box, (width, height))

    # This central box intentionally leaves the top-left output region outside the
    # transfer support, allowing a direct check of the neutral background contract.
    query_box = np.array([width // 4, height // 4, 3 * width // 4, 3 * height // 4], dtype=np.float32)
    logits, aligned_mask = prepare_aligned_prior(
        str(prior_path), query_box, (width, height), device="cpu"
    )
    assert logits is not None and tuple(logits.shape) == (1, 1, 256, 256)
    assert aligned_mask is not None and aligned_mask.shape == (height, width)
    # Local soft logits are bounded and do not force global background.
    assert float(logits.abs().max()) <= 2.0001
    assert abs(float(logits[0, 0, 0, 0])) < 1e-6

    print("prior_transfer functional checks passed")
    print(f"prior={prior_path}")
    print(f"source_size={width}x{height}, prior_area_ratio={prior_mask.mean():.4f}")
    print(f"aligned_area_ratio={aligned_mask.mean():.4f}, logit_range=({float(logits.min()):.3f}, {float(logits.max()):.3f})")


if __name__ == "__main__":
    main()
