import sys
import os
from pathlib import Path
import time
from typing import Dict, Optional

import logging
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 1. 强制添加 sam2 源码路径
sam2_repo_path = r"E:\PythonD\ERA\sam2_code"
if sam2_repo_path not in sys.path:
    sys.path.insert(0, sam2_repo_path)

if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"导入SAM2库时出错。")
    raise e

# --- 🚀 核心修改：数据集切换配置 ---
DATASET_NAME = "ISIC2018"  # 可选: "ISIC2018" 或 "Kvasir-SEG"

if DATASET_NAME == "ISIC2018":
    BASE_DIR = Path(r"E:/Datasets/ISIC2018/test")
    IMAGE_DIR = BASE_DIR / "images_compressed"
    #PROMPT_DIR = BASE_DIR / "prompts"
    PROMPT_DIR = BASE_DIR / "prompts_rag"
    GT_DIR = BASE_DIR / "groundtruth"
else:
    BASE_DIR = Path(r"E:/Datasets/Kvasir-SEG")
    IMAGE_DIR = BASE_DIR / "images"
    PROMPT_DIR = BASE_DIR / "prompts"
    GT_DIR = BASE_DIR / "masks"

OUTPUT_DIR = Path(rf"E:/PythonD/ERA3/scripts/V9/outputs_{DATASET_NAME}")

SAM2_MODEL_CFG = Path(r"E:\PythonD\ERA\sam2_weights\config.yaml")
SAM2_CHECKPOINT = Path(r"E:\PythonD\ERA\sam2_weights\sam2.1_hiera_base_plus.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================= 工具函数 =================

def calculate_metrics_from_totals(tp_total, fp_total, fn_total, tn_total) -> Dict[str, float]:
    tp, fp, fn, tn = float(tp_total), float(fp_total), float(fn_total), float(tn_total)
    epsilon = 1e-8
    macro_dice = (2. * tp) / (2 * tp + fp + fn + epsilon)
    macro_iou = tp / (tp + fp + fn + epsilon)
    macro_sensitivity = tp / (tp + fn + epsilon)
    macro_specificity = tn / (tn + fp + epsilon)
    macro_accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    return {"dice": macro_dice, "iou": macro_iou, "sensitivity": macro_sensitivity, "specificity": macro_specificity,
            "accuracy": macro_accuracy}


def evaluate_folder_macro(masks_dir: Path, gt_dir: Path) -> Dict[str, float]:
    total_tp, total_fp, total_fn, total_tn = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    processed_count = 0
    # 我们保存的是 .jpg 格式的预测结果
    mask_files = sorted([f for f in masks_dir.iterdir() if f.suffix.lower() == '.jpg'])

    for mask_file in tqdm(mask_files, desc="评测掩码 (宏观)"):
        clean_id = mask_file.stem.replace("_segmentation", "")

        # --- 🚀 修复：多后缀兼容查找 ---
        possible_gt_names = [
            f"{clean_id}.png", f"{clean_id}.jpg", f"{clean_id}.jpeg",
            f"{clean_id}_segmentation.png", f"{clean_id}_segmentation.jpg"
        ]

        gt_file = None
        for name in possible_gt_names:
            temp_path = gt_dir / name
            if temp_path.exists():
                gt_file = temp_path
                break

        if gt_file is None:
            if processed_count < 1: # 只打印第一个找不到的，避免刷屏
                print(f"\n[错误] 找不到对应的 GT 文件。尝试过: {possible_gt_names}，路径: {gt_dir}")
            continue

        try:
            # 统一转为单通道二值图
            pred_mask = np.array(Image.open(mask_file).convert('L')) > 127
            gt_mask = np.array(Image.open(gt_file).convert('L')) > 127

            total_tp += np.logical_and(pred_mask, gt_mask).sum()
            total_fp += np.logical_and(pred_mask, ~gt_mask).sum()
            total_fn += np.logical_and(~pred_mask, gt_mask).sum()
            total_tn += np.logical_and(~pred_mask, ~gt_mask).sum()
            processed_count += 1
        except Exception as e:
            continue

    print(f"\n成功匹配并计算了 {processed_count} 张图像")
    return calculate_metrics_from_totals(total_tp, total_fp, total_fn, total_tn) if processed_count > 0 else {}

def load_prompts_3wd(prompt_file: Path):
    if not prompt_file.exists(): return None, "Reject"
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if content.lower() == "null": return None, "Reject"
    decision = "Defer" if "uncertain" in content.lower() else "Accept"
    boxes = []
    for line in content.split('\n'):
        if "uncertain" in line.lower(): continue
        try:
            coords = [float(c) for c in line.strip().split()]
            if len(coords) == 4:
                # 兼容处理：知识库脚本生成的是 [x, y, w, h]，三支决策脚本生成的是 [x1, y1, x2, y2]
                # SAM2 predict 接收的是 [x1, y1, x2, y2]
                # 如果你的 .txt 是之前 1_生成提示标签.py 产生的，它已经是 [x1, y1, x2, y2]，直接用即可
                boxes.append(coords)
        except:
            continue
    return (np.array(boxes[0]), decision) if boxes else (None, "Reject")


def jitter_bbox(bbox, img_w, img_h, scale=0.05):
    x_min, y_min, x_max, y_max = bbox
    w, h = x_max - x_min, y_max - y_min
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

    def clamp(box):
        return [max(0, int(box[0])), max(0, int(box[1])), min(img_w - 1, int(box[2])), min(img_h - 1, int(box[3]))]

    box_shrunk = clamp(
        [cx - w * (1 - scale) / 2, cy - h * (1 - scale) / 2, cx + w * (1 - scale) / 2, cy + h * (1 - scale) / 2])
    box_expanded = clamp(
        [cx - w * (1 + scale) / 2, cy - h * (1 + scale) / 2, cx + w * (1 + scale) / 2, cy + h * (1 + scale) / 2])

    return np.array([bbox, box_shrunk, box_expanded])


# ================= 主程序 =================

def main():
    print(f"当前处理数据集: {DATASET_NAME}")
    print(f"使用设备: {DEVICE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sam2_model = build_sam2(str(SAM2_MODEL_CFG), str(SAM2_CHECKPOINT))
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    sam2_predictor.model.to(DEVICE)

    image_files = sorted([f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    stats = {"Accept": 0, "Defer": 0, "Reject": 0}
    start_time = time.perf_counter()

    for image_path in tqdm(image_files, desc=f"正在执行 {DATASET_NAME} 分割"):
        try:
            image = Image.open(image_path).convert("RGB")
            W, H = image.size
            sam2_predictor.set_image(image)

            # 🚀 核心修复：文件名解耦
            clean_id = image_path.stem.replace("_segmentation", "")
            prompt_file = PROMPT_DIR / f"{clean_id}_segmentation.txt"

            box_prompt, decision = load_prompts_3wd(prompt_file)

            stats[decision] += 1
            best_mask = None

            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                if decision == "Reject":
                    best_mask = np.zeros((H, W), dtype=bool)
                elif decision == "Accept":
                    masks, scores, _ = sam2_predictor.predict(box=box_prompt, multimask_output=False)
                    best_mask = masks[np.argmax(scores)]
                elif decision == "Defer":
                    jittered_boxes = jitter_bbox(box_prompt, W, H, scale=0.05)
                    all_masks = []
                    for box in jittered_boxes:
                        masks, _, _ = sam2_predictor.predict(box=box, multimask_output=True)
                        all_masks.append(masks)

                    ensemble_masks = np.concatenate(all_masks, axis=0)
                    best_mask = np.sum(ensemble_masks.astype(np.uint8), axis=0) >= 5

                if best_mask is not None:
                    # 统一保存为原图 ID，不带冗余后缀
                    mask_img = Image.fromarray((np.squeeze(best_mask) > 0).astype(np.uint8) * 255)
                    mask_img.save(OUTPUT_DIR / f"{clean_id}.jpg", "JPEG")

        except Exception as e:
            tqdm.write(f"处理失败 {image_path.name}: {e}")

    total_time = time.perf_counter() - start_time
    print(f"\n分布统计: {stats}")

    final_metrics = evaluate_folder_macro(OUTPUT_DIR, GT_DIR)
    report = (
        f"--- {DATASET_NAME} 评测报告 ---\n"
        f"宏观Dice系数:      {final_metrics.get('dice', 0):.4f}\n"
        f"宏观IoU:           {final_metrics.get('iou', 0):.4f}\n"
        f"宏观敏感度 (SE):   {final_metrics.get('sensitivity', 0):.4f}\n"
        f"宏观特异度 (SP):   {final_metrics.get('specificity', 0):.4f}\n"
        f"宏观准确率 (AC):   {final_metrics.get('accuracy', 0):.4f}\n"
    )
    print("\n" + report)


if __name__ == "__main__":
    main()

'''import sys
import os
from pathlib import Path
import time
from typing import Dict, Optional

import logging
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 1. 强制添加 sam2 源码路径
sam2_repo_path = r"E:\PythonD\ERA\sam2_code"
if sam2_repo_path not in sys.path:
    sys.path.insert(0, sam2_repo_path)

if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"导入SAM2库时出错。")
    raise e

# --- 路径配置 (保持原样) ---
IMAGE_DIR = Path(r"E:/Datasets/ISIC2018/test/images_compressed")
PROMPT_DIR = Path(r"E:/Datasets/ISIC2018/test/prompts")
GT_DIR = Path(r"E:/Datasets/ISIC2018/test/groundtruth")
OUTPUT_DIR = Path(r"E:/PythonD/ERA3/scripts/V9/outputs")  # 改回原目录

SAM2_MODEL_CFG = Path(r"E:\PythonD\ERA\sam2_weights\config.yaml")
SAM2_CHECKPOINT = Path(r"E:\PythonD\ERA\sam2_weights\sam2.1_hiera_base_plus.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================= 工具函数 =================

def calculate_metrics_from_totals(tp_total, fp_total, fn_total, tn_total) -> Dict[str, float]:
    tp, fp, fn, tn = float(tp_total), float(fp_total), float(fn_total), float(tn_total)
    epsilon = 1e-8
    macro_dice = (2. * tp) / (2 * tp + fp + fn + epsilon)
    macro_iou = tp / (tp + fp + fn + epsilon)
    macro_sensitivity = tp / (tp + fn + epsilon)
    macro_specificity = tn / (tn + fp + epsilon)
    macro_accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    return {"dice": macro_dice, "iou": macro_iou, "sensitivity": macro_sensitivity, "specificity": macro_specificity,
            "accuracy": macro_accuracy}


def evaluate_folder_macro(masks_dir: Path, gt_dir: Path) -> Dict[str, float]:
    total_tp, total_fp, total_fn, total_tn = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    processed_count = 0
    mask_files = sorted([f for f in masks_dir.iterdir() if f.is_file() and f.suffix.lower() == '.jpg'])

    for mask_file in tqdm(mask_files, desc="评测掩码 (宏观)"):
        gt_file = gt_dir / f"{mask_file.stem}_segmentation.png"
        if not gt_file.exists(): continue
        try:
            pred_mask = np.array(Image.open(mask_file).convert('L')) > 127
            gt_mask = np.array(Image.open(gt_file).convert('L')) > 127
            total_tp += np.logical_and(pred_mask, gt_mask).sum()
            total_fp += np.logical_and(pred_mask, ~gt_mask).sum()
            total_fn += np.logical_and(~pred_mask, gt_mask).sum()
            total_tn += np.logical_and(~pred_mask, ~gt_mask).sum()
            processed_count += 1
        except:
            continue
    return calculate_metrics_from_totals(total_tp, total_fp, total_fn, total_tn) if processed_count > 0 else {}


def load_prompts_3wd(prompt_file: Path):
    if not prompt_file.exists(): return None, "Reject"
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if content.lower() == "null": return None, "Reject"
    decision = "Defer" if "uncertain" in content.lower() else "Accept"
    boxes = []
    for line in content.split('\n'):
        if "uncertain" in line.lower(): continue
        try:
            coords = [float(c) for c in line.strip().split()]
            if len(coords) == 4: boxes.append(coords)
        except:
            continue
    return (np.array(boxes[0]), decision) if boxes else (None, "Reject")


def jitter_bbox(bbox, img_w, img_h, scale=0.05):
    """
    生成 3 个不同的框：原始、缩小 5%、放大 5%
    """
    x_min, y_min, x_max, y_max = bbox
    w, h = x_max - x_min, y_max - y_min
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

    def clamp(box):
        return [max(0, int(box[0])), max(0, int(box[1])), min(img_w - 1, int(box[2])), min(img_h - 1, int(box[3]))]

    box_shrunk = clamp(
        [cx - w * (1 - scale) / 2, cy - h * (1 - scale) / 2, cx + w * (1 - scale) / 2, cy + h * (1 - scale) / 2])
    box_expanded = clamp(
        [cx - w * (1 + scale) / 2, cy - h * (1 + scale) / 2, cx + w * (1 + scale) / 2, cy + h * (1 + scale) / 2])

    return np.array([bbox, box_shrunk, box_expanded])


# ================= 主程序 =================

def main():
    print(f"使用设备: {DEVICE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sam2_model = build_sam2(str(SAM2_MODEL_CFG), str(SAM2_CHECKPOINT))
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    sam2_predictor.model.to(DEVICE)

    image_files = sorted([f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    stats = {"Accept": 0, "Defer": 0, "Reject": 0}
    start_time = time.perf_counter()

    for image_path in tqdm(image_files, desc="正在执行增强分割"):
        try:
            image = Image.open(image_path).convert("RGB")
            W, H = image.size
            sam2_predictor.set_image(image)

            prompt_file = PROMPT_DIR / f"{image_path.stem}_segmentation.txt"
            box_prompt, decision = load_prompts_3wd(prompt_file)

            stats[decision] += 1
            best_mask = None

            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                if decision == "Reject":
                    best_mask = np.zeros((H, W), dtype=bool)
                elif decision == "Accept":
                    masks, scores, _ = sam2_predictor.predict(box=box_prompt, multimask_output=False)
                    best_mask = masks[np.argmax(scores)]
                elif decision == "Defer":
                    # --- 集成投票逻辑 ---
                    jittered_boxes = jitter_bbox(box_prompt, W, H, scale=0.05)
                    all_masks = []
                    for box in jittered_boxes:
                        masks, _, _ = sam2_predictor.predict(box=box, multimask_output=True)
                        all_masks.append(masks)  # 每个框 3 个候选

                    ensemble_masks = np.concatenate(all_masks, axis=0)  # (9, H, W)
                    best_mask = np.sum(ensemble_masks.astype(np.uint8), axis=0) >= 5  # 九票中过半数

                    # 调试对比
                    orig_masks, orig_scores, _ = sam2_predictor.predict(box=box_prompt, multimask_output=False)
                    diff = np.sum(best_mask) - np.sum(orig_masks[np.argmax(orig_scores)])
                    if diff != 0:
                        tqdm.write(f"  >> [Enhanced] {image_path.name}: 修正像素 {diff}")

                if best_mask is not None:
                    mask_img = Image.fromarray((np.squeeze(best_mask) > 0).astype(np.uint8) * 255)
                    mask_img.save(OUTPUT_DIR / f"{image_path.stem}.jpg", "JPEG")

        except Exception as e:
            tqdm.write(f"处理失败 {image_path.name}: {e}")

    total_time = time.perf_counter() - start_time
    print(f"\n分布统计: {stats}")

    final_metrics = evaluate_folder_macro(OUTPUT_DIR, GT_DIR)
    report = (
        f"宏观Dice系数:      {final_metrics.get('dice', 0):.4f}\n"
        f"宏观IoU:           {final_metrics.get('iou', 0):.4f}\n"
        f"宏观敏感度 (SE):   {final_metrics.get('sensitivity', 0):.4f}\n"
        f"宏观特异度 (SP):   {final_metrics.get('specificity', 0):.4f}\n"
        f"宏观准确率 (AC):   {final_metrics.get('accuracy', 0):.4f}\n"
    )
    print("\n" + report)


if __name__ == "__main__":
    main()
    '''