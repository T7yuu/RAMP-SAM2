"""Create revision-safe prompt folders from existing frozen retrieval outputs.

This utility does not recompute BiomedCLIP retrieval. It preserves the precomputed
query box, point, and retrieved mask path, then applies the revised geometry gate.
It is intended for a controlled rerun when the cached BiomedCLIP model cannot be
loaded. Original prompt folders are never modified.
"""
from pathlib import Path

from PIL import Image

from prior_transfer import is_prior_geometry_compatible


JOBS = (
    (
        Path(r"E:\Datasets\ISIC2018\test\images_compressed"),
        Path(r"E:\Datasets\ISIC2018\test\prompts_rag"),
        Path(r"E:\Datasets\ISIC2018\test\prompts_rag_revised_safe"),
    ),
    (
        Path(r"E:\Datasets\PH2\trainx"),
        Path(r"E:\Datasets\PH2\trainx\prompts_rag_cross_ph2"),
        Path(r"E:\Datasets\PH2\trainx\prompts_rag_cross_ph2_revised_safe"),
    ),
)


def image_lookup(image_dir: Path):
    result = {}
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            result[path.stem.replace("_segmentation", "")] = path
    return result


def parse_prompt(prompt_path: Path):
    lines = prompt_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None
    parts = lines[0].split()
    if len(parts) < 4:
        return None
    try:
        box = [float(v) for v in parts[:4]]
    except ValueError:
        return None
    # Legacy PH2 prompts contain only a box and prior path. Reconstruct a neutral
    # center point so downstream parsing follows the revised six-coordinate format.
    if len(parts) >= 6:
        values = box + [float(parts[4]), float(parts[5])]
        prior = parts[6] if len(parts) > 6 else None
    else:
        values = box + [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]
        prior = parts[4] if len(parts) > 4 else None
    is_defer = len(lines) > 1 and "uncertain" in lines[1].lower()
    return values, prior, is_defer


def process_job(image_dir: Path, old_dir: Path, new_dir: Path):
    images = image_lookup(image_dir)
    new_dir.mkdir(parents=True, exist_ok=True)
    counts = {"accept_retained": 0, "defer_existing": 0, "defer_geometry": 0, "skipped": 0}
    for prompt_path in sorted(old_dir.glob("*_segmentation.txt")):
        sample_id = prompt_path.stem.replace("_segmentation", "")
        parsed = parse_prompt(prompt_path)
        image_path = images.get(sample_id)
        if parsed is None or image_path is None:
            counts["skipped"] += 1
            continue
        values, prior_path, was_defer = parsed
        box = values[:4]
        output_path = new_dir / prompt_path.name
        if was_defer:
            output_path.write_text(" ".join(map(str, values)) + "\nuncertain", encoding="utf-8")
            counts["defer_existing"] += 1
            continue
        with Image.open(image_path) as image:
            geometry_ok = is_prior_geometry_compatible(prior_path, box, image.size)
        if geometry_ok:
            output_path.write_text(" ".join(map(str, values)) + f" {prior_path}", encoding="utf-8")
            counts["accept_retained"] += 1
        else:
            output_path.write_text(" ".join(map(str, values)) + "\nuncertain", encoding="utf-8")
            counts["defer_geometry"] += 1
    print(f"{image_dir.name}: {counts}; output={new_dir}")


def main():
    for job in JOBS:
        process_job(*job)


if __name__ == "__main__":
    main()
