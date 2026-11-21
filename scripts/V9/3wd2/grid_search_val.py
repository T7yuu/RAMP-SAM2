import sys
import os
import csv
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from prompt import load_biomedclip_and_adapter, get_biomedclip_attention_bbox_and_point, load_knowledge_base, \
    CLINICAL_PROMPT_TEMPLATES
import v5_multi_box
from prior_transfer import is_prior_geometry_compatible, prepare_aligned_prior

DATASET_NAME = "ISIC2018"
VAL_DIR = Path(r"E:\Datasets\ISIC2018\validate")
IMAGE_DIR = VAL_DIR / "images_compressed"
GT_DIR = VAL_DIR / "groundtruth"

SAM2_MODEL_CFG = Path(r"E:\PythonD\ERA\sam2_weights\config.yaml")
SAM2_CHECKPOINT = Path(r"E:\PythonD\ERA\sam2_weights\sam2.1_hiera_base_plus.pt")
KB_JSON = r"E:\PythonD\ERA3\scripts\V9\3wd2\unified_medical_kb.json"
KB_NPZ = r"E:\PythonD\ERA3\scripts\V9\scripts\outputs_medical\medical_features.npz"
ADAPTER_PATH = r"E:\PythonD\ERA3\scripts\V9\scripts\2wd_adapter.pth"  # 指向新模型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_bbox_from_mask(mask_2d):
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    if not np.any(rows): return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [xmin, ymin, xmax, ymax]


def calculate_iou(box1, box2):
    if box1 is None or box2 is None: return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = b1_area + b2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def cache_validation_predictions():
    print("🚀 启动缓存阶段: 仿真 Accept (先验注入) 与 Defer (多框集成)...")
    model, preprocess_val, tokenizer, adapter = load_biomedclip_and_adapter(DEVICE, ADAPTER_PATH)
    kb_dict, kb_features_raw, _ = load_knowledge_base(KB_JSON, KB_NPZ)
    kb_list = list(kb_dict.values())
    kb_features_raw_np = kb_features_raw.numpy()

    with torch.no_grad():
        kb_features_2wd = adapter(kb_features_raw.to(DEVICE)).cpu().numpy()

    sam2_predictor = v5_multi_box.SAM2ImagePredictor(
        v5_multi_box.build_sam2(str(SAM2_MODEL_CFG), str(SAM2_CHECKPOINT)).to(DEVICE))
    image_files = sorted([f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in ('.jpg', '.png')])
    text_label = CLINICAL_PROMPT_TEMPLATES.get(DATASET_NAME, "medical image of lesion")
    cached_data = []

    for img_path in tqdm(image_files, desc="缓存验证集"):
        clean_id = img_path.stem.replace("_segmentation", "")
        gt_path = next(
            (GT_DIR / n for n in [f"{clean_id}.png", f"{clean_id}_segmentation.png"] if (GT_DIR / n).exists()), None)
        if not gt_path: continue

        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        gt_mask = np.array(Image.open(gt_path).convert('L')) > 127

        candidate_bbox, best_point, query_feat_raw, text_feat = get_biomedclip_attention_bbox_and_point(model,
                                                                                                        preprocess_val,
                                                                                                        tokenizer,
                                                                                                        image,
                                                                                                        text_label,
                                                                                                        DEVICE)

        best_sim = -1.0
        mask_accept, mask_defer = np.zeros((H, W), dtype=bool), np.zeros((H, W), dtype=bool)
        spatial_conflict = False

        if candidate_bbox is not None:
            with torch.no_grad():
                query_feat_2wd = adapter(query_feat_raw).cpu().numpy().squeeze()
                text_feat_np = text_feat.cpu().numpy().squeeze()

            sim_scores_img = np.dot(kb_features_2wd, query_feat_2wd.T)
            sim_scores_text = np.dot(kb_features_raw_np, text_feat_np.T)
            sim_scores = 0.95 * sim_scores_img + 0.05 * sim_scores_text

            best_idx = np.argmax(sim_scores)
            best_sim = float(sim_scores[best_idx])
            prior_mask_path = kb_list[best_idx].get('mask_path', 'none')

            # Match deployed geometry-safe routing; never resize a source mask over the whole query image.
            prior_is_safe = is_prior_geometry_compatible(prior_mask_path, candidate_bbox, (W, H))
            if prior_is_safe:
                prior_mask_tensor, aligned_prior = prepare_aligned_prior(
                    prior_mask_path, candidate_bbox, (W, H), DEVICE
                )
                prior_is_safe = prior_mask_tensor is not None and aligned_prior is not None
            else:
                prior_mask_tensor = None
            spatial_conflict = not prior_is_safe

            sam2_predictor.set_image(image)
            in_point, in_label, box_prompt = np.array([best_point]), np.array([1]), np.array(candidate_bbox)

            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                m_acc, s_acc, _ = sam2_predictor.predict(point_coords=in_point, point_labels=in_label, box=box_prompt,
                                                         mask_input=prior_mask_tensor, multimask_output=True)
                mask_accept = v5_multi_box.postprocess_medical_mask(m_acc[np.argmax(s_acc)], W, H)

                all_masks, all_scores = [], []
                for box in v5_multi_box.jitter_bbox(box_prompt, W, H):
                    m_def, s_def, _ = sam2_predictor.predict(point_coords=in_point, point_labels=in_label, box=box,
                                                             multimask_output=True)
                    all_masks.append(m_def)
                    all_scores.append(s_def)
                c_masks, c_scores = np.concatenate(all_masks, axis=0), np.concatenate(all_scores, axis=0)
                valid_idx = np.where(c_scores > 0.75)[0]
                mask_defer = v5_multi_box.ensemble_voting(c_masks[valid_idx], c_scores[valid_idx]) if len(
                    valid_idx) > 0 else c_masks[np.argmax(c_scores)]
                mask_defer = v5_multi_box.postprocess_medical_mask(mask_defer, W, H)

        if mask_accept.shape != gt_mask.shape:
            gt_mask = cv2.resize(gt_mask.astype(np.uint8), (mask_accept.shape[1], mask_accept.shape[0]),
                                 interpolation=cv2.INTER_NEAREST) > 0

        cached_data.append({
            'id': clean_id, 'sim': best_sim, 'gt': gt_mask,
            'mask_accept': mask_accept, 'mask_defer': mask_defer,
            'conflict': spatial_conflict
        })

    return cached_data


def grid_search_thresholds(cached_data):
    print("\n🔍 开始高速网格搜索 (单阈值 2WD)...")
    best_dice = 0.0
    best_th = 0

    ths = np.arange(0.00, 0.90, 0.02)
    results_log = []

    for th in tqdm(ths, desc="阈值搜索"):
        tp, fp, fn, tn = 0, 0, 0, 0
        counts = {"Accept": 0, "Defer": 0}

        for item in cached_data:
            sim = item['sim']

            # 单阈值判定 + 冲突拦截
            if sim >= th and not item['conflict']:
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

        metrics = v5_multi_box.calculate_metrics_from_totals(tp, fp, fn, tn)
        current_dice = metrics.get('dice', 0)

        results_log.append({
            "Routing_TH": round(th, 2),
            "Dice": round(current_dice, 4),
            "SP(特异度)": round(metrics.get('specificity', 0), 4),
            "SE(召回率)": round(metrics.get('sensitivity', 0), 4),
            "Num_Accept": counts["Accept"],
            "Num_Defer": counts["Defer"],
        })

        if current_dice > best_dice:
            best_dice = current_dice
            best_th = th
            tqdm.write(f"🌟 新纪录! TH = {th:.2f} | Dice: {best_dice:.4f} | 分布: {counts}")

    print(f"\n🏆 最优路由参数: Threshold = {best_th:.2f}")

    csv_filename = "grid_search_sensitivity_log.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results_log[0].keys())
        writer.writeheader()
        writer.writerows(results_log)

    print(f"\n✅ 完整敏感性分析日志已保存至 '{csv_filename}'")


if __name__ == "__main__":
    cache = cache_validation_predictions()
    if cache:
        grid_search_thresholds(cache)


