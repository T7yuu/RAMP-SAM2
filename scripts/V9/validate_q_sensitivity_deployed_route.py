"""Q_i coefficient sensitivity on the deployed route (the Table 4 path).

Why this script exists
----------------------
`validate_q_sensitivity_deployed.py` rebuilds the pool with ``fast=False`` and
then scores a separately filtered candidate list.  Its nominal macro Dice is
0.7826, which does not equal the 0.7842 reported for the deployed system in
Table 4, so the sweep and the ablation table are not on the same footing.

This script instead reuses the exact pipeline of ``run_systematic_ablation.py``
for the variant ``prior_soft_affine_neutral``:

  * ``build_pool(..., fast=True, prompt_scale=(scale, scale))``
  * cached Top-1 prior from the prompt file, four-factor gate on its geometry
  * ``candidate_features`` + the RANK_PRODUCTION scoring terms
  * secondary logit refinement and ``apply_postprocess``

The only quantity that varies is the 11-dimensional Q_i weight vector, applied
wherever Q_i is evaluated (initial ranking and the re-ranking of the three
refined masks).  With the nominal vector the pipeline is identical to the one
that produced Table 4, so the baseline must reproduce 0.7842 by construction.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(r"E:/PythonD/ERA3")
V9 = ROOT / "scripts" / "V9"
UA_PATH = V9 / "run_updated_ablation.py"
SA_PATH = V9 / "systematic_ablation.py"
RSA_PATH = V9 / "run_systematic_ablation.py"
PROMPT_SCRIPT = V9 / "3wd2" / "prompt.py"

DATA = Path(r"E:/Datasets/ISIC2018/validate")
PROMPT_DIR = DATA / "prompts_rag_revised_safe"
DEFAULT_OUT = V9 / "q_sensitivity_deployed_route"

BASE_WEIGHTS = {
    "confidence": 3.0, "colour": 1.0, "edge": 0.05, "compactness": 0.1,
    "point_inside": 0.5, "point_outside": -3.0, "box_fill_penalty": -5.0,
    "area_penalty": -8.0, "box_area_penalty": -0.6, "prior_iou": 0.15,
    "empty_penalty": -3.0,
}
VARIANT = "prior_soft_affine_neutral"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ua = load_module(UA_PATH, "ramp_updated_ablation")
sa = load_module(SA_PATH, "ramp_systematic_ablation")
rsa = load_module(RSA_PATH, "ramp_systematic_ablation_runner")
sys.path.insert(0, str(PROMPT_SCRIPT.parent))
prompt_mod = load_module(PROMPT_SCRIPT, "ramp_prompt_for_scores")


def score_with_weights(image, candidates, features, weights, prior_mask, mod):
    """Mirror of ua.score_candidates(terms=RANK_PRODUCTION) with a weight vector."""
    if not candidates:
        return None
    contrasts = ua.robust_unit_scale([f["contrast"] for f in features])
    edges = ua.robust_unit_scale([f["edge"] for f in features])
    h, w = (np.asarray(image).shape[:2] if not isinstance(image, Image.Image)
            else np.asarray(image).shape[:2])
    usable = prior_mask is not None and 0.001 < float(np.mean(prior_mask)) < 0.50
    q_all = []
    for i, (c, f) in enumerate(zip(candidates, features)):
        q = weights["confidence"] * float(c["score"])
        q += weights["colour"] * float(contrasts[i])
        q += weights["edge"] * float(edges[i])
        q += weights["compactness"] * f["compactness"]
        if c.get("point") is not None:
            q += weights["point_inside"] if f["contains_point"] else weights["point_outside"]
        q += weights["box_fill_penalty"] * max(0.0, f["box_fill"] - 0.90)
        q += weights["area_penalty"] * max(0.0, f["area_ratio"] - 0.90)
        if c.get("box") is not None:
            q += weights["box_area_penalty"] * mod.box_area_ratio(c["box"], w, h)
        if f["area_ratio"] < mod.MIN_MASK_AREA_RATIO:
            q += weights["empty_penalty"]
        if usable:
            cand = np.asarray(c["mask"]) > 0
            inter = float(np.logical_and(cand, prior_mask).sum())
            union = float(np.logical_or(cand, prior_mask).sum())
            q += weights["prior_iou"] * inter / max(1.0, union)
        q_all.append(q)
    return int(np.argmax(q_all))


def refine(predictor, image, chosen, cfg, mod):
    """Secondary logit refinement, as in run_systematic_ablation.refine_and_select."""
    if chosen is None or not cfg.use_logit_refinement:
        return []
    if chosen.get("box") is None or chosen.get("point") is None:
        return []
    w, h = image.size
    rb = mod.expand_box(chosen["box"], w, h, scale=cfg.box_refinement_scale)
    with torch.inference_mode(), torch.autocast(mod.DEVICE, dtype=torch.bfloat16):
        masks, scores, logits = predictor.predict(
            point_coords=np.array([chosen["point"]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            box=rb, mask_input=chosen["logit"][None, ...], multimask_output=True)
    return [{"mask": np.asarray(m) > 0, "score": float(s), "box": rb,
             "point": chosen["point"], "origin": "refined",
             "logit": np.asarray(l), "prior_ref": chosen.get("prior_ref")}
            for m, s, l in zip(np.asarray(masks), np.asarray(scores), np.asarray(logits))]


def per_image_metrics(pred, gt):
    p, g = np.asarray(pred) > 0, np.asarray(gt) > 0
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    tn = float(np.logical_and(~p, ~g).sum())
    return tp, fp, fn, tn, 2 * tp / max(1e-9, 2 * tp + fp + fn)


def aggregate(rows):
    tp = sum(r[0] for r in rows); fp = sum(r[1] for r in rows)
    fn = sum(r[2] for r in rows); tn = sum(r[3] for r in rows)
    return {"images": len(rows),
            "dice_micro": 2 * tp / max(1e-9, 2 * tp + fp + fn),
            "dice_macro": float(np.mean([r[4] for r in rows])),
            "dice_macro_std": float(np.std([r[4] for r in rows], ddof=1))}


def build_configs(n_joint, seed):
    cfgs = [("baseline", dict(BASE_WEIGHTS))]
    for name, value in BASE_WEIGHTS.items():
        for direction, mult in (("down20", 0.8), ("up20", 1.2)):
            w = dict(BASE_WEIGHTS); w[name] = value * mult
            cfgs.append((f"{name}_{direction}", w))
    rng = np.random.default_rng(seed)
    for i in range(n_joint):
        cfgs.append((f"joint_{i+1:03d}",
                     {k: v * rng.uniform(0.8, 1.2) for k, v in BASE_WEIGHTS.items()}))
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    ua.IMAGE_DIR, ua.GT_DIR = DATA / "images_compressed", DATA / "groundtruth"
    ua.PROMPT_DIR = PROMPT_DIR
    mod = ua.load_current_module()
    predictor = mod.SAM2ImagePredictor(
        mod.build_sam2(str(mod.SAM2_MODEL_CFG), str(mod.SAM2_CHECKPOINT)).to(mod.DEVICE))

    adapter_path = str(getattr(mod, "ADAPTER_PATH", None) or
                       r"E:/PythonD/ERA3/scripts/V9/scripts/2wd_adapter.pth")
    model, preprocess, tokenizer, adapter = prompt_mod.load_biomedclip_and_adapter(
        mod.DEVICE, adapter_path)
    kb_dict, kb_raw, _ = prompt_mod.load_knowledge_base(
        r"E:/PythonD/ERA3/scripts/V9/3wd2/unified_medical_kb.json",
        r"E:/PythonD/ERA3/scripts/V9/scripts/outputs_medical/medical_features.npz")
    kb_list = list(kb_dict.values()); kb_raw_np = kb_raw.numpy()
    with torch.no_grad():
        kb_proj = adapter(kb_raw.to(mod.DEVICE)).cpu().numpy()

    cfg = next(c for c in sa.build_ablation_suite() if c.name == VARIANT)
    cfgs = build_configs(args.joint_samples, args.seed)

    files = sorted(p for p in ua.IMAGE_DIR.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        files = files[:args.limit]

    rows = {n: [] for n, _ in cfgs}
    n_accept = n_defer = n_failed = 0
    per_image_rows = []
    print(f"device={mod.DEVICE} images={len(files)} configs={len(cfgs)}", flush=True)

    for image_path in tqdm(files, desc="Q_i sensitivity (deployed route)"):
        image_id = ua.clean_id(image_path)
        gt_file = ua.find_gt(image_id)
        if gt_file is None:
            n_failed += 1
            continue
        image_full = Image.open(image_path).convert("RGB")
        fw, fh = image_full.size
        scale = min(1.0, float(rsa.MAX_IMAGE_DIM) / max(fw, fh))
        if scale < 1.0:
            image = image_full.resize((max(1, int(round(fw * scale))),
                                       max(1, int(round(fh * scale)))),
                                      Image.Resampling.LANCZOS)
            image_full.close()
        else:
            image = image_full
        gt = ua.mask_from_file(gt_file, (image.height, image.width))
        w, h = image.size
        try:
            predictor.set_image(image)
            pool = ua.build_pool(mod, predictor, image, image_id, fast=True,
                                 prompt_scale=(scale, scale))
            prompt_file = PROMPT_DIR / f"{image_id}_segmentation.txt"
            _, _, cached_decision, prior_path = mod.load_prompts_2wd(prompt_file)
            cached_prior = None
            if prior_path and prior_path != "none":
                pp = Path(prior_path)
                if pp.exists():
                    cached_prior = ua.mask_from_file(pp, (h, w))

            # ---- cached Top-1 prior + four-factor gate (production semantics) ----
            if cached_prior is None:
                biou, ratio, cdist = 0.0, 0.0, 999.0
            else:
                biou, ratio, cdist = rsa.prior_geometry_stats(cached_prior, pool["box"], w, h)
            image_tensor = preprocess(image).unsqueeze(0).to(mod.DEVICE)
            _, _, qraw, text_feat = prompt_mod.get_biomedclip_attention_bbox_and_point(
                model, preprocess, tokenizer, image,
                prompt_mod.CLINICAL_PROMPT_TEMPLATES.get("ISIC2018", "medical image"),
                mod.DEVICE)
            with torch.no_grad():
                qproj = adapter(qraw).cpu().numpy().squeeze()
                text_np = text_feat.cpu().numpy().squeeze()
            best_sim = float(np.max(.95 * (kb_proj @ qproj) + .05 * (kb_raw_np @ text_np)))
            accept = sa.route_accept(best_sim, biou, ratio, cdist, cfg.gates)
            route = "Accept" if accept else "Defer"
            n_accept += route == "Accept"; n_defer += route == "Defer"

            cands = [c for c in pool["candidates"] if c.get("origin") != "prior"]
            scoring_prior = None
            if route == "Accept" and cached_prior is not None:
                logits, aligned = sa.build_prior_variant(cached_prior, pool["box"], cfg, (w, h))
                rsa.add_predict_candidates(predictor, image, cands, pool["box"], pool["point"],
                                           mask_input=logits.numpy()[0], origin="prior",
                                           prior_ref=aligned)
                scoring_prior = cached_prior
            feats = ua.candidate_features(image, cands, mod)

            # ---- per-config selection, refinement cached by chosen identity ----
            chosen_by_cfg = {}
            for name, wts in cfgs:
                idx = score_with_weights(image, cands, feats, wts, scoring_prior, mod)
                chosen_by_cfg[name] = None if idx is None else cands[idx]

            refined_cache, mask_cache = {}, {}
            for name, wts in cfgs:
                c = chosen_by_cfg[name]
                if c is None:
                    rows[name].append((0.0, 0.0, 0.0, float(gt.size), 0.0))
                    continue
                key = id(c)
                if key not in refined_cache:
                    refined_cache[key] = refine(predictor, image, c, cfg, mod)
                refined = refined_cache[key]
                if refined:
                    rf = ua.candidate_features(image, refined, mod)
                    ridx = score_with_weights(image, refined, rf, wts, scoring_prior, mod)
                    if ridx is not None:
                        c = refined[ridx]
                mkey = id(c)
                if mkey not in mask_cache:
                    mask_cache[mkey] = sa.apply_postprocess(
                        np.asarray(c["mask"]) > 0, (w, h), cfg)
                tp, fp, fn, tn, dice = per_image_metrics(mask_cache[mkey], gt)
                rows[name].append((tp, fp, fn, tn, dice))
                per_image_rows.append({"image_id": image_id, "config": name,
                                       "route": route, "dice": dice})
        except Exception as exc:
            print(f"failed {image_id}: {type(exc).__name__}: {exc}")
            n_failed += 1

    summary = {n: aggregate(r) for n, r in rows.items()}
    base = summary["baseline"]["dice_macro"]
    one_at = []
    for name, _ in cfgs:
        if name == "baseline" or name.startswith("joint_"):
            continue
        s = summary[name]
        one_at.append({"config": name, "dice_macro": s["dice_macro"],
                       "delta_macro": s["dice_macro"] - base,
                       "rel_macro_pct": 100 * (s["dice_macro"] - base) / base})
    joints = [summary[n] for n, _ in cfgs if n.startswith("joint_")]
    report = {
        "protocol": {"route": "run_systematic_ablation.prior_soft_affine_neutral",
                     "prompt_scale": "fixed", "fast": True,
                     "refinement": True, "postprocess": True,
                     "joint_samples": args.joint_samples, "seed": args.seed,
                     "images": len(rows["baseline"]), "failed": n_failed,
                     "accept": n_accept, "defer": n_defer,
                     "note": "Q_i weights applied at both selection stages"},
        "baseline": summary["baseline"],
        "one_at_a_time": one_at,
        "one_at_a_time_range_macro": [min(r["dice_macro"] for r in one_at),
                                      max(r["dice_macro"] for r in one_at)],
        "joint": ({"samples": len(joints),
                   "mean_dice_macro": float(np.mean([j["dice_macro"] for j in joints])),
                   "std_dice_macro": float(np.std([j["dice_macro"] for j in joints], ddof=1)),
                   "min_dice_macro": float(np.min([j["dice_macro"] for j in joints])),
                   "max_dice_macro": float(np.max([j["dice_macro"] for j in joints]))}
                  if joints else None),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output / "one_at_a_time.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(one_at[0].keys())); wr.writeheader(); wr.writerows(one_at)
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        wr.writeheader(); wr.writerows(per_image_rows)
    print(json.dumps({k: report[k] for k in ("baseline", "one_at_a_time_range_macro", "joint")},
                     indent=2))


if __name__ == "__main__":
    main()
