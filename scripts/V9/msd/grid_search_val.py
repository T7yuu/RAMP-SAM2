import sys
import os
import numpy as np
import torch
import cv2
import csv
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from prompt import load_biomedclip_and_adapter, get_biomedclip_attention_bbox, load_knowledge_base, \
    CLINICAL_PROMPT_TEMPLATES
import v5_multi_box

DATASET_NAME = "Task09_Spleen_2D"
VAL_DIR = Path(r"E:\Datasets\Task09_Spleen_2D\val")
IMAGE_DIR = VAL_DIR / "images"
GT_DIR = VAL_DIR / "groundtruth"

SAM2_MODEL_CFG = Path(r"E:\PythonD\ERA\sam2_weights\config.yaml")
SAM2_CHECKPOINT = Path(r"E:\PythonD\ERA\sam2_weights\sam2.1_hiera_base_plus.pt")
KB_JSON = "./unified_medical_kb.json"
KB_NPZ = "./outputs_medical/medical_features.npz"
ADAPTER_PATH = "./3wd_adapter.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cache_validation_predictions():
    print("🚀 启动缓存阶段: 纯Box仿真 Accept 与 Defer ...")
    model, preprocess_val, tokenizer, adapter = load_biomedclip_and_adapter(DEVICE, ADAPTER_PATH)
    kb_dict, kb_features_raw, _ = load_knowledge_base(KB_JSON, KB_NPZ)
    kb_list = list(kb_dict.values())
    kb_features_raw_np = kb_features_raw.numpy()

    with torch.no_grad():
        kb_features_3wd = adapter(kb_features_raw.to(DEVICE)).cpu().numpy()

    sam2_predictor = v5_multi_box.SAM2ImagePredictor(
        v5_multi_box.build_sam2(str(SAM2_MODEL_CFG), str(SAM2_CHECKPOINT)).to(DEVICE))
    image_files = sorted([f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in ('.jpg', '.png')])
    text_label = CLINICAL_PROMPT_TEMPLATES.get(DATASET_NAME, "medical image")
    cached_data = []

    for img_path in tqdm(image_files, desc="缓存验证集"):
        clean_id = img_path.stem.replace("_segmentation", "")
        gt_path = next(
            (GT_DIR / n for n in [f"{clean_id}.png", f"{clean_id}_segmentation.png"] if (GT_DIR / n).exists()), None)
        if not gt_path: continue

        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        gt_mask = np.array(Image.open(gt_path).convert('L')) > 127

        candidate_bbox, query_feat_raw, text_feat = get_biomedclip_attention_bbox(model, preprocess_val, tokenizer,
                                                                                  image, text_label, DEVICE)
        best_sim = -1.0
        mask_accept, mask_defer = np.zeros((H, W), dtype=bool), np.zeros((H, W), dtype=bool)

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

            sam2_predictor.set_image(image)
            box_prompt = np.array(candidate_bbox)

            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                # 🚀 纯 Box 注入先验
                mask_input = v5_multi_box.process_prior_mask(prior_mask_path, DEVICE)
                m_acc, s_acc, _ = sam2_predictor.predict(box=box_prompt, mask_input=mask_input, multimask_output=True)
                mask_accept = v5_multi_box.postprocess_medical_mask(m_acc[np.argmax(s_acc)], W, H)

                # 🚀 纯 Box 多框集成
                all_masks, all_scores = [], []
                for box in v5_multi_box.jitter_bbox(box_prompt, W, H):
                    m_def, s_def, _ = sam2_predictor.predict(box=box, multimask_output=True)
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

        cached_data.append(
            {'id': clean_id, 'sim': best_sim, 'gt': gt_mask, 'mask_accept': mask_accept, 'mask_defer': mask_defer,
             'mask_reject': np.zeros_like(gt_mask)})
    return cached_data


def grid_search_thresholds(cached_data):
    # 网格搜索逻辑与此前完全一致，直接沿用
    print("\n🔍 开始高速网格搜索及阈值敏感性分析...")
    best_dice = 0.0
    best_params = {"accept_th": 0, "defer_th": 0}
    acc_ths, def_ths = np.arange(0.85, 0.99, 0.01), np.arange(0.50, 0.85, 0.02)
    results_log = []

    with tqdm(total=len(acc_ths) * len(def_ths), desc="参数网格搜索") as pbar:
        for acc_th in acc_ths:
            for def_th in def_ths:
                if def_th >= acc_th: pbar.update(1); continue
                tp, fp, fn, tn = 0, 0, 0, 0
                counts = {"Accept": 0, "Defer": 0, "Reject": 0}

                for item in cached_data:
                    sim = item['sim']
                    if sim >= acc_th:
                        pred, counts["Accept"] = item['mask_accept'], counts["Accept"] + 1
                    elif sim >= def_th:
                        pred, counts["Defer"] = item['mask_defer'], counts["Defer"] + 1
                    else:
                        pred, counts["Reject"] = item['mask_reject'], counts["Reject"] + 1

                    gt = item['gt']
                    tp += np.logical_and(pred, gt).sum()
                    fp += np.logical_and(pred, ~gt).sum()
                    fn += np.logical_and(~pred, gt).sum()
                    tn += np.logical_and(~pred, ~gt).sum()

                metrics = v5_multi_box.calculate_metrics_from_totals(tp, fp, fn, tn)
                current_dice = metrics.get('dice', 0)

                results_log.append({
                    "Accept_TH": round(acc_th, 2), "Defer_TH": round(def_th, 2),
                    "Dice": round(current_dice, 4), "SP": round(metrics.get('specificity', 0), 4),
                    "SE": round(metrics.get('sensitivity', 0), 4), "Precision": round(tp / (tp + fp + 1e-8), 4),
                    "Num_Accept": counts["Accept"], "Num_Defer": counts["Defer"], "Num_Reject": counts["Reject"]
                })

                if current_dice > best_dice:
                    best_dice, best_params = current_dice, {"accept_th": acc_th, "defer_th": def_th}
                    tqdm.write(
                        f"🌟 新纪录! Accept ≥ {acc_th:.2f}, Defer ≥ {def_th:.2f} | Dice: {best_dice:.4f} | 分布: {counts}")
                pbar.update(1)

    print(f"\n🏆 最优整体参数: Accept = {best_params['accept_th']:.2f}, Defer = {best_params['defer_th']:.2f}")
    with open("grid_search_sensitivity_log.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results_log[0].keys())
        writer.writeheader();
        writer.writerows(results_log)


if __name__ == "__main__":
    cache = cache_validation_predictions()
    if cache: grid_search_thresholds(cache)