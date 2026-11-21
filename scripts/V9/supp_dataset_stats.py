"""Compute cohort statistics for the supplementary material.

Reads ISIC2018 (train/validate/test) and PH2 image + mask pairs from the local
dataset roots and reports sizes, resolutions and lesion-area fractions.  Only
aggregate statistics are emitted; no per-case data leaves this script.
"""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ISIC = Path("E:/Datasets/ISIC2018")
PH2 = Path("E:/Datasets/PH2")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def isic_split(split: str):
    imgs = list_images(ISIC / split / "images_compressed")
    gts = list_images(ISIC / split / "groundtruth")
    return imgs, gts


def ph2_pairs():
    imgs = list_images(PH2 / "trainx")
    gts = list_images(PH2 / "trainy")
    return imgs, gts


def gt_lookup(gts):
    # match by filename stem; PH2 masks carry a *_lesion suffix
    table = {}
    for p in gts:
        stem = p.stem
        for suffix in ("_lesion", "_segmentation"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        table[stem] = p
    return table


def collect(imgs, gts, name):
    lut = gt_lookup(gts)
    ws, hs, fracs, missing = [], [], [], 0
    for ip in imgs:
        mp = lut.get(ip.stem)
        if mp is None:
            missing += 1
            continue
        with Image.open(ip) as im:
            w, h = im.size
        ws.append(w)
        hs.append(h)
        m = np.asarray(Image.open(mp).convert("L"))
        fracs.append(float((m > 127).mean()))
    fracs = np.asarray(fracs)
    return {
        "cohort": name,
        "n_images": len(imgs),
        "n_masks": len(gts),
        "n_matched": len(fracs),
        "n_unmatched": missing,
        "width": {"min": int(min(ws)), "max": int(max(ws)), "median": int(np.median(ws))},
        "height": {"min": int(min(hs)), "max": int(max(hs)), "median": int(np.median(hs))},
        "lesion_area_fraction": {
            "mean": float(fracs.mean()),
            "std": float(fracs.std(ddof=1)),
            "p10": float(np.percentile(fracs, 10)),
            "p25": float(np.percentile(fracs, 25)),
            "p50": float(np.percentile(fracs, 50)),
            "p75": float(np.percentile(fracs, 75)),
            "p90": float(np.percentile(fracs, 90)),
            "min": float(fracs.min()),
            "max": float(fracs.max()),
            "frac_below_1pct": float((fracs < 0.01).mean()),
            "frac_above_50pct": float((fracs > 0.50).mean()),
        },
        "_raw_fracs": fracs,
        "_raw_wh": list(zip(ws, hs)),
    }


def main():
    out = {}
    raws = {}
    for split in ("train", "validate", "test"):
        imgs, gts = isic_split(split)
        r = collect(imgs, gts, f"ISIC2018-{split}")
        raws[f"ISIC2018-{split}"] = (r.pop("_raw_fracs"), r.pop("_raw_wh"))
        out[f"ISIC2018-{split}"] = r
    imgs, gts = ph2_pairs()
    r = collect(imgs, gts, "PH2")
    raws["PH2"] = (r.pop("_raw_fracs"), r.pop("_raw_wh"))
    out["PH2"] = r

    os.makedirs("scripts/V9/supp_stats", exist_ok=True)
    with open("scripts/V9/supp_stats/cohort_stats.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    np.savez(
        "scripts/V9/supp_stats/cohort_raw.npz",
        **{f"{k}_frac": v[0] for k, v in raws.items()},
        **{f"{k}_wh": np.asarray(v[1]) for k, v in raws.items()},
    )

    for k, v in out.items():
        print(k, "n=", v["n_images"], "matched=", v["n_matched"])
        print("   W", v["width"], "H", v["height"])
        print("   area", {kk: round(vv, 4) for kk, vv in v["lesion_area_fraction"].items()})


if __name__ == "__main__":
    main()
