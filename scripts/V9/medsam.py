import os
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys

# 尝试导入原版 SAM 库
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("❌ 找不到 segment_anything 库。")
    sys.exit(1)

# ================= 🚀 路径配置 =================
PH2_DIR = Path(r"E:\Datasets\PH2")
IMAGE_DIR = PH2_DIR / "trainx"
GT_DIR = PH2_DIR / "trainy"

MEDSAM_CHECKPOINT = r"E:\PythonD\ERA3\MdeSam\weights\medsam_vit_b.pth"
MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 🛠️ 核心函数 =================
def calculate_metrics_from_totals(tp: float, fp: float, fn: float, tn: float) -> dict:
    eps = 1e-8
    return {
        "dice": (2. * tp) / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "sensitivity": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps)
    }

# ================= 🚀 主程序 =================
def main():
    print(f"🔥 正在初始化 MedSAM Predictor (全图框盲测模式, 设备: {DEVICE})...")
    if not os.path.exists(MEDSAM_CHECKPOINT):
        print(f"❌ 找不到 MedSAM 权重文件: {MEDSAM_CHECKPOINT}")
        return

    # 初始化 MedSAM 模型
    medsam_model = sam_model_registry[MODEL_TYPE](checkpoint=MEDSAM_CHECKPOINT)
    medsam_model = medsam_model.to(DEVICE)
    medsam_predictor = SamPredictor(medsam_model)

    if not IMAGE_DIR.exists() or not GT_DIR.exists():
        print(f"❌ 数据集路径不存在，请检查: {IMAGE_DIR}")
        return

    image_files = sorted(
        [f for f in IMAGE_DIR.iterdir() if f.is_file() and f.suffix.lower() in ['.bmp', '.png', '.jpg', '.jpeg']])
    print(f"📁 共找到 {len(image_files)} 个测试样本。")

    tp_total, fp_total, fn_total, tn_total = 0.0, 0.0, 0.0, 0.0

    for img_path in tqdm(image_files, desc="纯 Zero-shot 评测 (MedSAM 全图框)"):
        clean_id = img_path.stem

        # 寻找对应的 GT
        gt_path = None
        for ext in ['.bmp', '.png', '.jpg', '.jpeg']:
            temp_path = GT_DIR / f"{clean_id}{ext}"
            if temp_path.exists(): gt_path = temp_path; break
        if not gt_path:
            for ext in ['.bmp', '.png', '.jpg', '.jpeg']:
                temp_path = GT_DIR / f"{clean_id}_lesion{ext}"
                if temp_path.exists(): gt_path = temp_path; break

        if not gt_path: continue

        # 1. 加载图像与标签
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        gt_mask = np.array(Image.open(gt_path).convert("L")) > 127
        H, W = gt_mask.shape

        # 2. 生成全图边界框 (Whole Image Box Prompt)
        # 不依赖任何 GT，直接用图像的宽和高
        whole_image_box = np.array([0, 0, W - 1, H - 1])

        # 3. MedSAM 推理
        medsam_predictor.set_image(image_np)

        with torch.no_grad():
            masks, scores, _ = medsam_predictor.predict(
                box=whole_image_box,
                multimask_output=False # MedSAM 推荐单层输出
            )

        best_mask = masks[0]
        best_mask = best_mask > 0 # 确保布尔类型

        if best_mask.shape != gt_mask.shape:
            best_mask = cv2.resize(best_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

        # 4. 统计混淆矩阵
        tp_total += np.logical_and(best_mask, gt_mask).sum()
        fp_total += np.logical_and(best_mask, ~gt_mask).sum()
        fn_total += np.logical_and(~best_mask, gt_mask).sum()
        tn_total += np.logical_and(~best_mask, ~gt_mask).sum()

    # 5. 输出结果
    metrics = calculate_metrics_from_totals(tp_total, fp_total, fn_total, tn_total)

    print("\n" + "=" * 45)
    print(" 🌟 纯 MedSAM (全图边界框盲测) 测试报告 🌟 ")
    print("=" * 45)
    print(f"Dice 系数 : {metrics['dice']:.4f}")
    print(f"IoU (交并比): {metrics['iou']:.4f}")
    print(f"SE (敏感性) : {metrics['sensitivity']:.4f}")
    print(f"SP (特异性) : {metrics['specificity']:.4f}")
    print(f"AC (准确率) : {metrics['accuracy']:.4f}")
    print("=" * 45)

if __name__ == "__main__":
    main()