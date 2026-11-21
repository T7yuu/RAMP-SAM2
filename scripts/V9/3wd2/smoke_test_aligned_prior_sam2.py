"""One-image end-to-end smoke test for aligned retrieved-prior SAM2 prompting.

The test reads an existing prompt file, runs the revised aligned-soft-prior path on
one ISIC2018 image, and writes a disposable prediction to the local smoke-test folder.
It does not alter the original prompt files or full evaluation outputs.
"""
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

SAM2_REPO = r"E:\PythonD\ERA\sam2_code"
if SAM2_REPO not in sys.path:
    sys.path.insert(0, SAM2_REPO)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from prior_transfer import prepare_aligned_prior

IMAGE_DIR = Path(r"E:\Datasets\ISIC2018\test\images_compressed")
PROMPT_DIR = Path(r"E:\Datasets\ISIC2018\test\prompts_rag_revised_safe")
OUTPUT_DIR = Path(__file__).with_name("smoke_test_output")
SAM2_CONFIG = r"E:\PythonD\ERA\sam2_weights\config.yaml"
SAM2_CHECKPOINT = r"E:\PythonD\ERA\sam2_weights\sam2.1_hiera_base_plus.pt"


def clamp_box(box, width, height):
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    x1, x2 = sorted((float(np.clip(x1, 0, width - 1)), float(np.clip(x2, 0, width - 1))))
    y1, y2 = sorted((float(np.clip(y1, 0, height - 1)), float(np.clip(y2, 0, height - 1))))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def load_first_accept_prompt():
    for image_path in sorted(IMAGE_DIR.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        prompt_path = PROMPT_DIR / f"{image_path.stem}_segmentation.txt"
        if not prompt_path.is_file():
            continue
        lines = prompt_path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 1 or (len(lines) > 1 and "uncertain" in lines[1].lower()):
            continue
        parts = lines[0].split()
        if len(parts) >= 7 and Path(parts[6]).is_file():
            return image_path, np.asarray([float(v) for v in parts[:4]], dtype=np.float32), np.asarray([float(v) for v in parts[4:6]], dtype=np.float32), parts[6]
    raise RuntimeError("No existing Accept prompt with a readable prior mask was found.")


def main():
    image_path, box, point, prior_path = load_first_accept_prompt()
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    box = clamp_box(box, width, height)
    point = np.clip(point, [0, 0], [width - 1, height - 1])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prior_logits, aligned_mask = prepare_aligned_prior(prior_path, box, (width, height), device=device)
    if prior_logits is None or aligned_mask is None:
        raise RuntimeError("Aligned prior construction unexpectedly failed.")
    assert tuple(prior_logits.shape) == (1, 1, 256, 256)
    assert float(prior_logits.abs().max()) <= 2.0001

    predictor = SAM2ImagePredictor(build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT).to(device))
    predictor.set_image(image)
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        masks, scores, _ = predictor.predict(
            point_coords=np.asarray([point]), point_labels=np.asarray([1]),
            box=box, mask_input=prior_logits, multimask_output=True
        )
    best_idx = int(np.argmax(scores))
    output_mask = (np.asarray(masks[best_idx]) > 0).astype(np.uint8) * 255
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"{image_path.stem}_aligned_prior.png"
    Image.fromarray(output_mask).save(output_path)

    print("aligned-prior SAM2 smoke test passed")
    print(f"device={device}")
    print(f"image={image_path.name}, image_size={width}x{height}")
    print(f"prior={prior_path}")
    print(f"prior_logit_range=({float(prior_logits.min()):.3f}, {float(prior_logits.max()):.3f})")
    print(f"aligned_prior_area_ratio={aligned_mask.mean():.4f}")
    print(f"sam2_scores={np.asarray(scores).round(4).tolist()}, selected={best_idx}")
    print(f"prediction_area_ratio={(output_mask > 0).mean():.4f}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()

