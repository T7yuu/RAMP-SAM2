import sys
import os
from pathlib import Path
from typing import Dict
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sam2_repo_path = r"E:\PythonD\ERA\sam2_code"
if sam2_repo_path not in sys.path: sys.path.insert(0, sam2_repo_path)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

USE_ENSEMBLE = True

DATASET_NAME = "Task09_Spleen_2D"
BASE_DIR = Path(r"E:\Datasets\Task09_Spleen_2D\test")
IMAGE_DIR = BASE_DIR / "images"
PROMPT_DIR = BASE_DIR / "prompts_rag"
GT_DIR = BASE_DIR / "groundtruth"
OUTPUT_DIR = Path(rf"E:/PythonD/ERA3/scripts/V9/outputs_v5_{DATASET_NAME}_{'ensemble' if USE_ENSEMBLE else 'single'}")

SAM2_MODEL_CFG = Path(r"E:\PythonD\ERA\sam2_weights\config.yaml")
SAM2_CHECKPOINT = Path(r"E:\PythonD\ERA\sam2_weights\sam2.1_hiera_base_plus.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ensemble_voting(masks, scores, threshold=0.5):
    if len(masks) == 0: return None
    weights = scores[:, np.newaxis, np.newaxis]
    prob_map = np.sum(masks.astype(np.float32) * weights, axis=0) / (np.sum(weights) + 1e-8)
    return prob_map > threshold

def calculate_metrics_from_totals(tp, fp, fn, tn) -> Dict[str, float]:
    eps = 1e-8
    return {
        "dice": (2. * tp) / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "sensitivity": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps)
    }

def evaluate_folder_macro(masks_dir: Path, gt_dir: Path):
    tp, fp, fn, tn = 0.0, 0.0, 0.0, 0.0
    for mask_file in tqdm([f for f in masks_dir.iterdir() if f.suffix in ('.jpg', '.png')], desc="评测中"):
        clean_id = mask_file.stem.replace("_segmentation", "")
        gt_file = next((gt_dir / n for n in [f"{clean_id}.png", f"{clean_id}_segmentation.png"] if (gt_dir / n).exists()), None)
        if not gt_file: continue

        pm = np.array(Image.open(mask_file).convert('L')) > 127
        gm = np.array(Image.open(gt_file).convert('L')) > 127
        if pm.shape != gm.shape: pm = cv2.resize(pm.astype(np.uint8), (gm.shape[1], gm.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

        tp += np.logical_and(pm, gm).sum()
        fp += np.logical_and(pm, ~gm).sum()
        fn += np.logical_and(~pm, gm).sum()
        tn += np.logical_and(~pm, ~gm).sum()
    return calculate_metrics_from_totals(tp, fp, fn, tn)

def load_prompts_3wd(prompt_file: Path):
    if not prompt_file.exists(): return None, "Reject", None
    with open(prompt_file, 'r', encoding='utf-8') as f: lines = f.readlines()
    if not lines or lines[0].strip().lower() == "null": return None, "Reject", None

    decision = "Defer" if (len(lines) > 1 and "uncertain" in lines[1].lower()) else "Accept"
    parts = lines[0].strip().split()
    try:
        # 🚀 仅读取 4 个 Box 坐标
        coords = [float(p) for p in parts[:4]]
        prior_mask = parts[4] if len(parts) > 4 else None
        return np.array(coords), decision, prior_mask
    except:
        return None, "Reject", None

def process_prior_mask(mask_path, device):
    if not mask_path or mask_path == 'none' or not os.path.exists(mask_path): return None
    try:
        mask_img = Image.open(mask_path).convert('L').resize((256, 256), resample=Image.NEAREST)
        logits = np.where(np.array(mask_img) > 127, 10.0, -10.0).astype(np.float32)
        return torch.from_numpy(logits).unsqueeze(0).unsqueeze(0).to(device)
    except: return None

def jitter_bbox(bbox, img_w, img_h, scales=[0.95, 1.0, 1.05]):
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cl = lambda b: [max(0, int(b[0])), max(0, int(b[1])), min(img_w - 1, int(b[2])), min(img_h - 1, int(b[3]))]
    return np.array([cl([cx - w * s / 2, cy - h * s / 2, cx + w * s / 2, cy + h * s / 2]) for s in scales])


def postprocess_medical_mask(mask, W, H, box_prompt=None):
    if not np.any(mask): return mask
    mask_u8 = mask.astype(np.uint8)

    # 【新增限制 1】 物理隔离：强行抹除边界框外部的所有预测结果
    if box_prompt is not None:
        x1, y1, x2, y2 = map(int, box_prompt)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)

        # 创建全黑掩码，只把框内的预测结果复制过来
        restricted_mask = np.zeros_like(mask_u8)
        restricted_mask[y1:y2, x1:x2] = mask_u8[y1:y2, x1:x2]
        mask_u8 = restricted_mask

    # 【新增限制 2】 面积过滤：脾脏不可能占据单张切片 30% 以上的面积
    # 原来设定的 0.90 是防图像反转的，这里针对脾脏单独加一个上限
    if np.sum(mask_u8) / (W * H) > 0.30:
        return np.zeros((H, W), dtype=bool)

    # 保留最大连通域（去除框内可能的游离散点）
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels > 1:
        mask_u8 = (labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)

    return cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0

def main():
    mode_str = "有融合 (Ensemble)" if USE_ENSEMBLE else "无融合 (Single Box)"
    print(f"🚀 启动纯Box终极集成任务 ({mode_str}): {DATASET_NAME}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sam2_predictor = SAM2ImagePredictor(build_sam2(str(SAM2_MODEL_CFG), str(SAM2_CHECKPOINT)).to(DEVICE))
    image_files = sorted([f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    stats = {"Accept": 0, "Defer": 0, "Reject": 0}

    for image_path in tqdm(image_files, desc=f"执行纯Box分割 ({mode_str})"):
        try:
            image = Image.open(image_path).convert("RGB")
            W, H = image.size
            sam2_predictor.set_image(image)
            clean_id = image_path.stem.replace("_segmentation", "")

            box_prompt, decision, prior_mask_path = load_prompts_3wd(PROMPT_DIR / f"{clean_id}_segmentation.txt")
            stats[decision] += 1
            best_mask = np.zeros((H, W), dtype=bool)

            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                if decision == "Accept":
                    mask_input = process_prior_mask(prior_mask_path, DEVICE)
                    # 🚀 纯 Box 调用
                    masks, scores, _ = sam2_predictor.predict(box=box_prompt, mask_input=mask_input, multimask_output=True)
                    best_mask = masks[np.argmax(scores)]

                elif decision == "Defer":
                    if USE_ENSEMBLE:
                        all_masks, all_scores = [], []
                        for box in jitter_bbox(box_prompt, W, H):
                            # 🚀 纯 Box Jittering
                            m, s, _ = sam2_predictor.predict(box=box, multimask_output=True)
                            all_masks.append(m)
                            all_scores.append(s)
                        c_scores = np.concatenate(all_scores, axis=0)
                        valid_idx = np.where(c_scores > 0.75)[0]
                        best_mask = ensemble_voting(np.concatenate(all_masks, axis=0)[valid_idx], c_scores[valid_idx]) if len(valid_idx) > 0 else np.concatenate(all_masks, axis=0)[np.argmax(c_scores)]
                    else:
                        masks, scores, _ = sam2_predictor.predict(box=box_prompt, multimask_output=True)
                        best_mask = masks[np.argmax(scores)]

            best_mask = postprocess_medical_mask(best_mask, W, H, box_prompt=box_prompt)
            Image.fromarray((best_mask.squeeze() > 0).astype(np.uint8) * 255).save(OUTPUT_DIR / f"{clean_id}.jpg", "JPEG")
        except Exception as e:
            tqdm.write(f"⚠️ 失败 {image_path.name}: {e}")

    print(f"\n📊 统计: {stats}")
    metrics = evaluate_folder_macro(OUTPUT_DIR, GT_DIR)
    print(f"\n--- Spleen 纯 Box 突破评测报告 ({mode_str}) ---")
    print(f"Dice: {metrics.get('dice', 0):.4f}")
    print(f"IoU: {metrics.get('iou', 0):.4f}")
    print(f"SE: {metrics.get('sensitivity', 0):.4f}")
    print(f"SP: {metrics.get('specificity', 0):.4f}")

if __name__ == "__main__":
    main()