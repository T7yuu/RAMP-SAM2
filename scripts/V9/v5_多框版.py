import sys
import os
from pathlib import Path
import time
from typing import Dict
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# --- SAM2 环境配置 ---
sam2_repo_path = r"E:\PythonD\ERA\sam2_code"
if sam2_repo_path not in sys.path:
    sys.path.insert(0, sam2_repo_path)
if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- 路径与配置 ---
DATASET_NAME = "ISIC2018"
BASE_DIR = Path(r"E:/Datasets/ISIC2018/test")
IMAGE_DIR = BASE_DIR / "images_compressed"
PROMPT_DIR = BASE_DIR / "prompts_rag"
GT_DIR = BASE_DIR / "groundtruth"
OUTPUT_DIR = Path(rf"E:/PythonD/ERA3/scripts/V9/outputs_v5_{DATASET_NAME}")

SAM2_MODEL_CFG = Path(r"E:\PythonD\ERA\sam2_weights\config.yaml")
SAM2_CHECKPOINT = Path(r"E:\PythonD\ERA\sam2_weights\sam2.1_hiera_base_plus.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================= 🚀 核心改进：多掩码加权投票 =================
def ensemble_voting(masks, scores, threshold=0.5):
    if len(masks) == 0: return None
    weights = scores[:, np.newaxis, np.newaxis]
    weighted_votes = np.sum(masks.astype(np.float32) * weights, axis=0)
    total_weight = np.sum(weights) + 1e-8
    prob_map = weighted_votes / total_weight
    return prob_map > threshold


# ==========================================================

def calculate_metrics_from_totals(tp_total, fp_total, fn_total, tn_total) -> Dict[str, float]:
    tp, fp, fn, tn = float(tp_total), float(fp_total), float(fn_total), float(tn_total)
    epsilon = 1e-8
    return {
        "dice": (2. * tp) / (2 * tp + fp + fn + epsilon),
        "iou": tp / (tp + fp + fn + epsilon),
        "sensitivity": tp / (tp + fn + epsilon),
        "specificity": tn / (tn + fp + epsilon),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + epsilon)
    }


def evaluate_folder_macro(masks_dir: Path, gt_dir: Path) -> Dict[str, float]:
    total_tp, total_fp, total_fn, total_tn = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    processed_count = 0
    mask_files = sorted([f for f in masks_dir.iterdir() if f.suffix.lower() in ('.jpg', '.png')])

    for mask_file in tqdm(mask_files, desc="评测掩码 (宏观)"):
        clean_id = mask_file.stem.replace("_segmentation", "")
        possible_gt_names = [f"{clean_id}.png", f"{clean_id}.jpg", f"{clean_id}_segmentation.png"]
        gt_file = next((gt_dir / n for n in possible_gt_names if (gt_dir / n).exists()), None)
        if gt_file is None: continue

        try:
            pred_mask = np.array(Image.open(mask_file).convert('L')) > 127
            gt_mask = np.array(Image.open(gt_file).convert('L')) > 127
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]),
                                       interpolation=cv2.INTER_NEAREST) > 0

            total_tp += np.logical_and(pred_mask, gt_mask).sum()
            total_fp += np.logical_and(pred_mask, ~gt_mask).sum()
            total_fn += np.logical_and(~pred_mask, gt_mask).sum()
            total_tn += np.logical_and(~pred_mask, ~gt_mask).sum()
            processed_count += 1
        except Exception:
            continue
    return calculate_metrics_from_totals(total_tp, total_fp, total_fn, total_tn) if processed_count > 0 else {}


def load_prompts_3wd(prompt_file: Path):
    if not prompt_file.exists(): return None, None, "Reject"
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if not content or content.lower() == "null": return None, None, "Reject"

    decision = "Defer" if "uncertain" in content.lower() else "Accept"
    for line in content.split('\n'):
        if "uncertain" in line.lower(): continue
        try:
            coords = [float(c) for c in line.strip().split()]
            if len(coords) == 6:
                return np.array(coords[:4]), np.array([coords[4], coords[5]]), decision
        except:
            continue
    return None, None, "Reject"


def jitter_bbox(bbox, img_w, img_h, scales=[0.95, 1.0, 1.05]):
    x_min, y_min, x_max, y_max = bbox
    w, h = x_max - x_min, y_max - y_min
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

    def clamp(box):
        return [max(0, int(box[0])), max(0, int(box[1])), min(img_w - 1, int(box[2])), min(img_h - 1, int(box[3]))]

    return np.array([clamp([cx - w * s / 2, cy - h * s / 2, cx + w * s / 2, cy + h * s / 2]) for s in scales])


def postprocess_medical_mask(mask, W, H):
    if not np.any(mask): return mask
    mask_uint8 = mask.astype(np.uint8)
    if np.sum(mask_uint8) / (W * H) > 0.90: return np.zeros((H, W), dtype=bool)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels > 1:
        mask_uint8 = (labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
    return mask_uint8 > 0


def main():
    print(f"🚀 启动终极集成任务 (多掩码加权投票版): {DATASET_NAME}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sam2_model = build_sam2(str(SAM2_MODEL_CFG), str(SAM2_CHECKPOINT))
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    sam2_predictor.model.to(DEVICE)

    image_files = sorted([f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    stats = {"Accept": 0, "Defer": 0, "Reject": 0}
    start_time = time.perf_counter()

    for image_path in tqdm(image_files, desc=f"执行终极分割"):
        try:
            image = Image.open(image_path).convert("RGB")
            W, H = image.size
            sam2_predictor.set_image(image)

            clean_id = image_path.stem.replace("_segmentation", "")
            prompt_file = PROMPT_DIR / f"{clean_id}_segmentation.txt"

            box_prompt, point_prompt, decision = load_prompts_3wd(prompt_file)
            stats[decision] += 1
            best_mask = None

            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                if decision == "Reject":
                    best_mask = np.zeros((H, W), dtype=bool)

                elif decision == "Accept":
                    in_point = np.array([point_prompt]) if point_prompt is not None else None
                    in_label = np.array([1]) if point_prompt is not None else None

                    masks, scores, _ = sam2_predictor.predict(
                        point_coords=in_point, point_labels=in_label,
                        box=box_prompt, multimask_output=True
                    )

                    # 过滤低分掩码后投票
                    valid_idx = np.where(scores > 0.60)[0]
                    if len(valid_idx) > 0:
                        best_mask = ensemble_voting(masks[valid_idx], scores[valid_idx])
                    else:
                        best_mask = masks[np.argmax(scores)]

                elif decision == "Defer":
                    jittered_boxes = jitter_bbox(box_prompt, W, H)
                    in_point = np.array([point_prompt]) if point_prompt is not None else None
                    in_label = np.array([1]) if point_prompt is not None else None

                    all_masks = []
                    all_scores = []

                    for box in jittered_boxes:
                        masks, scores, _ = sam2_predictor.predict(
                            point_coords=in_point, point_labels=in_label,
                            box=box, multimask_output=True
                        )
                        all_masks.append(masks)
                        all_scores.append(scores)

                    combined_masks = np.concatenate(all_masks, axis=0)
                    combined_scores = np.concatenate(all_scores, axis=0)

                    # Defer 域提高阈值过滤
                    valid_idx = np.where(combined_scores > 0.75)[0]
                    if len(valid_idx) > 0:
                        best_mask = ensemble_voting(combined_masks[valid_idx], combined_scores[valid_idx])
                    else:
                        best_mask = combined_masks[np.argmax(combined_scores)]

                if best_mask is not None:
                    best_mask = postprocess_medical_mask(best_mask, W, H)
                    Image.fromarray((best_mask.squeeze() > 0).astype(np.uint8) * 255).save(
                        OUTPUT_DIR / f"{clean_id}.jpg", "JPEG")

        except Exception as e:
            tqdm.write(f"⚠️ 失败 {image_path.name}: {e}")

    print(f"\n📊 统计: {stats} | 耗时: {time.perf_counter() - start_time:.2f}s")
    final_metrics = evaluate_folder_macro(OUTPUT_DIR, GT_DIR)
    print(
        f"\n--- 终极 V5 突破评测报告 ---\nDice: {final_metrics.get('dice', 0):.4f}\nIoU: {final_metrics.get('iou', 0):.4f}\nSE: {final_metrics.get('sensitivity', 0):.4f}\nSP: {final_metrics.get('specificity', 0):.4f}\nAC: {final_metrics.get('accuracy', 0):.4f}")


if __name__ == "__main__":
    main()