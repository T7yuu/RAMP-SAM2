"""Re-measure RAMP-SAM2 inference latency on the SAME pipeline that produces the
reported Dice (0.7842) -- the validation-cohort production path driven by
``run_systematic_ablation.py``:

    build_pool()            shared 6/9-scale ensemble, ONE batched SAM2 call
                            (branch-independent)
    + prior-injected decode ONLY when route == "Accept"
    refine_and_select()     1.30x box + logit re-decode, then re-rank
    apply_postprocess()     LCC + adaptive elliptical closing

The historical ``measure_efficiency`` timed a DIFFERENT pipeline
(``run_feature_space_ablation.predict_route``, ISIC2018/test, n=1000,
single-decode Accept), which is why its 304.3/854.7 ms did not match the paper's
Dice configuration.  This script times the actual experiments so latency and Dice
come from one configuration.

This file reuses the real functions from ``run_systematic_ablation.py`` verbatim
(imported, not re-implemented), so the measured call sequence is identical to the
ablated run; it only adds CUDA-synchronised ``perf_counter`` brackets.

Usage:
  python scripts/V9/measure_efficiency_v2.py --timing-images 100 --warmup 3 \
      --output scripts/V9/efficiency_results_v2/summary.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(r"E:/PythonD/ERA3")
RSA_PATH = ROOT / "scripts/V9/run_systematic_ablation.py"
DATA = Path(r"E:/Datasets/ISIC2018/validate")
MAX_IMAGE_DIM = 2048
DEFAULT_VARIANT = "prior_soft_affine_neutral"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sync(dev):
    if str(dev) == "cuda":
        torch.cuda.synchronize()


def timing_summary(xs):
    if not xs:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "n": 0}
    a = np.asarray(xs, dtype=float)
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "n": int(a.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=DATA)
    ap.add_argument("--variant", type=str, default=DEFAULT_VARIANT)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--prompt-dir", type=Path, default=None)
    ap.add_argument("--timing-images", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "scripts/V9/efficiency_results_v2/summary.json")
    args = ap.parse_args()

    # The production script is a module whose globals we reuse, so import it and
    # reuse its own loaded sub-modules (ua / sa / prompt_mod) rather than
    # re-implementing any of the call sequence.
    rsa = load_module(RSA_PATH, "ramp_run_systematic_ablation")

    ua = rsa.load_module(rsa.UA_PATH, "ramp_updated_ablation")
    sa = rsa.load_module(rsa.SA_PATH, "ramp_systematic_ablation")
    # prompt.py imports sibling modules (prior_transfer, ...), so its directory
    # must be importable before we exec it -- same order as the production script.
    sys.path.insert(0, str(rsa.PROMPT_SCRIPT.parent))
    prompt_mod = rsa.load_module(rsa.PROMPT_SCRIPT, "ramp_prompt_for_scores")

    data = args.dataset_root
    ua.IMAGE_DIR, ua.GT_DIR = data / "images_compressed", data / "groundtruth"
    ua.PROMPT_DIR = args.prompt_dir if args.prompt_dir else data / "prompts_rag_revised_safe"

    device = ua.load_current_module().DEVICE
    mod = ua.load_current_module()
    print(f"device={device}; data={data}; prompts={ua.PROMPT_DIR}; variant={args.variant}")

    predictor = mod.SAM2ImagePredictor(
        mod.build_sam2(str(mod.SAM2_MODEL_CFG), str(mod.SAM2_CHECKPOINT)).to(device))

    adapter_path = str(args.adapter) if args.adapter else (
        mod.ADAPTER_PATH if hasattr(mod, "ADAPTER_PATH")
        else r"E:/PythonD/ERA3/scripts/V9/scripts/2wd_adapter.pth")
    model, preprocess, tokenizer, adapter = prompt_mod.load_biomedclip_and_adapter(device, adapter_path)
    print(f"adapter={adapter_path}")
    kb_dict, kb_raw, _ = prompt_mod.load_knowledge_base(
        r"E:/PythonD/ERA3/scripts/V9/3wd2/unified_medical_kb.json",
        r"E:/PythonD/ERA3/scripts/V9/scripts/outputs_medical/medical_features.npz")
    kb_list = list(kb_dict.values()); kb_raw_np = kb_raw.numpy()
    rng = np.random.default_rng(42)
    with torch.no_grad():
        kb_proj = adapter(kb_raw.to(device)).cpu().numpy()

    suite = [c for c in sa.build_ablation_suite() if c.name == args.variant]
    if not suite:
        raise SystemExit(f"variant {args.variant} not found")
    cfg = suite[0]

    files = sorted([p for p in ua.IMAGE_DIR.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if args.timing_images:
        files = files[:args.timing_images]
    print(f"timing {len(files)} images (warmup={args.warmup})")

    times = defaultdict(list)
    route_counts = defaultdict(int)
    base_ensemble_sizes = []
    accept_extra = []

    for idx, image_path in enumerate(tqdm(files, desc="latency")):
        try:
            image_id = ua.clean_id(image_path)
            gt_file = ua.find_gt(image_id)
            if gt_file is None:
                continue
            image_full = Image.open(image_path).convert("RGB")
            full_w, full_h = image_full.size
            scale = min(1.0, float(MAX_IMAGE_DIM) / max(full_w, full_h))
            image = image_full.resize(
                (max(1, int(round(full_w * scale))), max(1, int(round(full_h * scale)))),
                Image.Resampling.LANCZOS) if scale < 1.0 else image_full
            if scale < 1.0:
                image_full.close()
            gt = ua.mask_from_file(gt_file, (image.height, image.width)); w, h = image.size
            warm = idx < args.warmup

            sync(device); t_img0 = time.perf_counter()
            predictor.set_image(image)
            sync(device); t_img = (time.perf_counter() - t_img0) * 1e3

            sync(device); t_b0 = time.perf_counter()

            # --- routing / prior retrieval -------------------------------------
            pool = ua.build_pool(mod, predictor, image, image_id, fast=True,
                                 prompt_scale=(scale, scale))
            prompt_file = ua.PROMPT_DIR / f"{image_id}_segmentation.txt"
            _, _, cached_decision, prior_path = mod.load_prompts_2wd(prompt_file)
            cached_prior = None
            if prior_path and prior_path != "none":
                cached_prior = (ua.mask_from_file(Path(prior_path), (h, w))
                                if Path(prior_path).exists() else None)
            _, _, qraw, text_feat = prompt_mod.get_biomedclip_attention_bbox_and_point(
                model, preprocess, tokenizer, image,
                prompt_mod.CLINICAL_PROMPT_TEMPLATES.get("ISIC2018", "medical image"), device)
            with torch.no_grad():
                qproj = adapter(qraw).cpu().numpy().squeeze()
                text_np = text_feat.cpu().numpy().squeeze()
            sim_img = kb_proj @ qproj; sim_txt = kb_raw_np @ text_np

            def _setup(mask):
                if mask is None:
                    return None, 0., 0., 999.
                b, r, c = rsa.prior_geometry_stats(mask, pool["box"], w, h)
                return mask, b, r, c

            prior_mask, biou, ratio, cdist, best_sim = (
                _setup(cached_prior) + (float(np.max(.95 * sim_img + .05 * sim_txt)),))
            sync(device); t1 = time.perf_counter()

            # --- shared ensemble is built inside build_pool above; the prior
            #     decode below is the ONLY branch-dependent decode ------------
            cands = [c for c in pool["candidates"] if c.get("origin") != "prior"]
            n_base = len(cands)
            # candidates == prompts x 3 (SAM2 multi-mask); count distinct prompts too
            n_prompts = len({str(c.get("origin", "")) for c in cands})

            route = cfg.force_route or ("Accept" if sa.route_accept(best_sim, biou, ratio, cdist,
                                                                    cfg.gates) else "Defer")
            n_prior_added = 0
            t_prior = 0.0
            t_build = 0.0
            t_feat = 0.0
            if route == "Accept" and prior_mask is not None:
                if cfg.prior_source == "ellipse":
                    inject_mask = rsa.synth_query_ellipse(pool["box"], (w, h))
                elif cfg.prior_source == "random":
                    inject_mask = rsa.random_retrieval_prior(kb_list, (w, h), rng)
                else:
                    inject_mask = prior_mask
                if inject_mask is not None:
                    sync(device); tb0 = time.perf_counter()
                    logits, aligned = sa.build_prior_variant(inject_mask, pool["box"], cfg, (w, h))
                    sync(device); t_build = (time.perf_counter() - tb0) * 1e3
                    sync(device); tp0 = time.perf_counter()
                    rsa.add_predict_candidates(predictor, image, cands, pool["box"], pool["point"],
                                               mask_input=logits.numpy()[0], origin="prior")
                    sync(device); t_prior = (time.perf_counter() - tp0) * 1e3
                    n_prior_added = 1
            sync(device); t2 = time.perf_counter()

            sync(device); tf0 = time.perf_counter()
            feats = ua.candidate_features(image, cands, mod)
            sync(device); t_feat = (time.perf_counter() - tf0) * 1e3
            scoring_prior = prior_mask if (route == "Accept" and prior_mask is not None) else None

            # --- selection + 1.30x logit refinement ---------------------------
            sync(device); t3a = time.perf_counter()
            chosen = rsa.select(pool, cands, feats, cfg, ua, mod, prior_mask=scoring_prior)
            mask = rsa.refine_and_select(predictor, pool, chosen, cands, feats, cfg, ua, mod,
                                         prior_mask=scoring_prior)
            sync(device); t3 = time.perf_counter()

            # --- post-process -------------------------------------------------
            sync(device); t4a = time.perf_counter()
            mask = sa.apply_postprocess(mask, (w, h), cfg)
            sync(device); t4 = time.perf_counter()

            if warm:
                continue
            route_counts[route] += 1
            base_ensemble_sizes.append(n_prompts)
            accept_extra.append(n_prior_added)
            times["image_encoding"].append(t_img)
            times["routing_retrieval"].append((t1 - t_b0) * 1e3)
            times["prior_build"].append(t_build)
            times["prior_decode"].append(t_prior)
            times["candidate_features"].append(t_feat)
            times["select_and_refine"].append((t3 - t3a) * 1e3)
            times["postprocess"].append((t4 - t4a) * 1e3)
            times["branch_total"].append((t4 - t_b0) * 1e3)
            times["end_to_end"].append(t_img + (t4 - t_b0) * 1e3)
            times["_route_" + route].append((t4 - t_b0) * 1e3)
            times["_priorbuild_" + route].append(t_build)
            times["_feat_" + route].append(t_feat)
        except Exception as exc:
            tqdm.write(f"fail {image_path.name}: {exc}")

    out = {
        "dataset": str(data),
        "variant": args.variant,
        "notes": "branch latency = everything after SAM2 image encoding; the shared "
                 "ensemble decode is inside build_pool and counted in routing_retrieval.",
        "warmup": args.warmup,
        "route_counts": dict(route_counts),
        "mean_base_ensemble_prompts": float(np.mean(base_ensemble_sizes)) if base_ensemble_sizes else 0.0,
        "mean_prior_proposals_added": float(np.mean(accept_extra)) if accept_extra else 0.0,
        "timings": {k: timing_summary(v) for k, v in times.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
