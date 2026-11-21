import sys
import os
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import csv

from prompt_PH2 import load_biomedclip_and_adapter, get_biomedclip_attention_bbox, load_knowledge_base, \
    CLINICAL_PROMPT_TEMPLATES
import v5_multi_box_PH2
from prior_transfer import is_prior_geometry_compatible, prepare_aligned_prior

DATASET_NAME = "PH2"
VAL_DIR = Path(r"E:\Datasets\PH2")
IMAGE_DIR = VAL_DIR / "trainx"
GT_DIR = VAL_DIR / "trainy"

SAM2_MODEL_CFG = Path(r"E:\PythonD\ERA\sam2_weights\config.yaml")
SAM2_CHECKPOINT = Path(r"E:\PythonD\ERA\sam2_weights\sam2.1_hiera_base_plus.pt")
# 🌟 指向原有的知识库
KB_JSON = r"E:\PythonD\ERA3\scripts\V9\3wd2\unified_medical_kb.json"
KB_NPZ = r"E:\PythonD\ERA3\scripts\V9\scripts\outputs_medical\medical_features.npz"
ADAPTER_PATH = r"E:\PythonD\ERA3\scripts\V9\scripts\3wd_adapter.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cache_validation_predictions():
    print("🚀 启动缓存阶段: 仿真 Accept (跨域先验注入) 与 Defer (多框集成)...")
    model, preprocess_val, tokenizer, adapter = load_biomedclip_and_adapter(DEVICE, ADAPTER_PATH)
    kb_dict, kb_features_raw, _ = load_knowledge_base(KB_JSON, KB_NPZ)
    kb_list = list(kb_dict.values())
    kb_features_raw_np = kb_features_raw.numpy()

    with torch.no_grad():
        kb_features_3wd = adapter(kb_features_raw.to(DEVICE)).cpu().numpy()

    sam2_predictor = v5_multi_box_PH2.SAM2ImagePredictor(
        v5_multi_box_PH2.build_sam2(str(SAM2_MODEL_CFG), str(SAM2_CHECKPOINT)).to(DEVICE))

    # 移除 "Dermoscopic_Image" 限制，直接读取所有支持后缀的图片
    image_files = sorted([f for f in IMAGE_DIR.rglob('*') if f.suffix.lower() in ('.jpg', '.png', '.jpeg', '.bmp')])
    text_label = CLINICAL_PROMPT_TEMPLATES.get(DATASET_NAME, "medical image of lesion")
    cached_data = []

    for img_path in tqdm(image_files, desc="缓存跨域验证集"):
        clean_id = img_path.stem.replace("_segmentation", "")

        possible_gts = list(GT_DIR.rglob(f"{clean_id}_lesion.bmp")) + list(GT_DIR.rglob(f"{clean_id}.png")) + list(
            GT_DIR.rglob(f"{clean_id}_segmentation.png"))
        if not possible_gts: continue
        gt_path = possible_gts[0]

        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        gt_mask = np.array(Image.open(gt_path).convert('L')) > 127

        candidate_bbox, best_point, query_feat_raw, text_feat = get_biomedclip_attention_bbox(model, preprocess_val, tokenizer,
                                                                                  image, text_label, DEVICE)
        best_sim = -1.0
        mask_accept, mask_defer = np.zeros((H, W), dtype=bool), np.zeros((H, W), dtype=bool)
        prior_is_safe = False

        if candidate_bbox is not None:
            with torch.no_grad():
                query_feat_3wd = adapter(query_feat_raw).cpu().numpy().squeeze()
                text_feat_np = text_feat.cpu().numpy().squeeze()

            sim_scores_img = np.dot(kb_features_3wd, query_feat_3wd.T)
            sim_scores_text = np.dot(kb_features_raw_np, text_feat_np.T)
            sim_scores = 0.95 * sim_scores_img + 0.05 * sim_scores_text

            best_idx = np.argmax(sim_scores)
            best_sim = float(sim_scores[best_idx])
            prior_mask_path = kb_list[best_idx].get('mask_path', 'none')

            # Match deployed PH2 routing: geometry checks precede injection and
            # the prior is normalized into the query box.
            prior_is_safe = is_prior_geometry_compatible(prior_mask_path, candidate_bbox, (W, H))
            if prior_is_safe:
                mask_input, aligned_prior = prepare_aligned_prior(
                    prior_mask_path, candidate_bbox, (W, H), DEVICE
                )
                prior_is_safe = mask_input is not None and aligned_prior is not None
            else:
                mask_input = None

            sam2_predictor.set_image(image)
            box_prompt = np.array(candidate_bbox)
            in_point, in_label = np.array([best_point]), np.array([1])

            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                m_acc, s_acc, _ = sam2_predictor.predict(
                    point_coords=in_point, point_labels=in_label,
                    box=box_prompt,
                    mask_input=mask_input if prior_is_safe else None,
                    multimask_output=True,
                )
                mask_accept = v5_multi_box_PH2.postprocess_medical_mask(m_acc[np.argmax(s_acc)], W, H)

                all_masks, all_scores = [], []
                for box in v5_multi_box_PH2.jitter_bbox(box_prompt, W, H):
                    m_def, s_def, _ = sam2_predictor.predict(point_coords=in_point, point_labels=in_label, box=box, multimask_output=True)
                    all_masks.append(m_def)
                    all_scores.append(s_def)
                c_masks, c_scores = np.concatenate(all_masks, axis=0), np.concatenate(all_scores, axis=0)
                valid_idx = np.where(c_scores > 0.75)[0]
                mask_defer = v5_multi_box_PH2.ensemble_voting(c_masks[valid_idx], c_scores[valid_idx]) if len(
                    valid_idx) > 0 else c_masks[np.argmax(c_scores)]
                mask_defer = v5_multi_box_PH2.postprocess_medical_mask(mask_defer, W, H)

        if mask_accept.shape != gt_mask.shape:
            gt_mask = cv2.resize(gt_mask.astype(np.uint8), (mask_accept.shape[1], mask_accept.shape[0]),
                                 interpolation=cv2.INTER_NEAREST) > 0

        cached_data.append({
            'id': clean_id, 'sim': best_sim, 'gt': gt_mask,
            'mask_accept': mask_accept, 'mask_defer': mask_defer, 'conflict': not prior_is_safe
        })

    return cached_data


def grid_search_thresholds(cached_data):
    print("\nStarting PH2 two-way Accept/Defer threshold search...")
    best_dice = 0.0
    best_th = 0.0
    ths = np.arange(0.00, 0.90, 0.02)
    results_log = []

    for th in tqdm(ths, desc="Threshold search"):
        tp, fp, fn, tn = 0, 0, 0, 0
        counts = {"Accept": 0, "Defer": 0}

        for item in cached_data:
            sim = item['sim']
            if sim >= th and not item.get('conflict', True):
                pred = item['mask_accept']
                counts["Accept"] += 1
            else:
                pred = item['mask_defer']
                counts["Defer"] += 1

            gt = item['gt']
            tp += np.logical_and(pred, gt).sum()
            fp += np.logical_and(pred, ~gt).sum()
            fn += np.logical_and(~pred, gt).sum()
            tn += np.logical_and(~pred, ~gt).sum()

        metrics = v5_multi_box_PH2.calculate_metrics_from_totals(tp, fp, fn, tn)
        current_dice = metrics.get('dice', 0)
        precision = tp / (tp + fp + 1e-8)

        results_log.append({
            "Routing_TH": round(th, 2),
            "Dice": round(current_dice, 4),
            "SP": round(metrics.get('specificity', 0), 4),
            "SE": round(metrics.get('sensitivity', 0), 4),
            "Precision": round(precision, 4),
            "Num_Accept": counts["Accept"],
            "Num_Defer": counts["Defer"],
        })

        if current_dice > best_dice:
            best_dice = current_dice
            best_th = th
            tqdm.write(f"New best PH2 two-way threshold = {th:.2f} | Dice: {best_dice:.4f} | counts: {counts}")

    print(f"\nBest PH2 two-way routing threshold = {best_th:.2f}")
    csv_filename = "grid_search_sensitivity_log_cross_domain_2wd.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results_log[0].keys())
        writer.writeheader()
        writer.writerows(results_log)
    print(f"\nPH2 two-way sensitivity log saved to '{csv_filename}'")
if __name__ == "__main__":
    cache = cache_validation_predictions()
    if cache:
        grid_search_thresholds(cache)

