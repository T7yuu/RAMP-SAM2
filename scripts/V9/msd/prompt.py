import os
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ================= 🚀 配置区域 =================
CONFIG_PATH = r"E:\model\grounding dino\groundingdino\config\GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = r"E:\model\grounding dino\weights\groundingdino_swint_ogc.pth"

# 加载模型时需确保 GroundingDINO 目录在 sys.path 中
import sys
gd_repo_path = r"E:\model\grounding dino"
if gd_repo_path not in sys.path:
    sys.path.insert(0, gd_repo_path)

from groundingdino.util.inference import load_model, predict
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_PROMPT = "spleen"  # 探测的目标
BOX_THRESHOLD = 0.35  # 探测信心值阈值
TEXT_THRESHOLD = 0.25  # 文本关联度阈值
# ===============================================

# 1. 加载模型 (单例模式)
gd_model = load_model(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)


def get_grounding_dino_bbox(image_path):
    """
    使用 Grounding DINO 获取高精度 Bounding Box
    返回格式: [x1, y1, x2, y2]
    """
    # Grounding DINO 内部有特定的预处理
    image_source, image_transformed = load_image(image_path)
    H, W, _ = image_source.shape

    boxes, logits, phrases = predict(
        model=gd_model,
        image=image_transformed,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )

    if len(boxes) == 0:
        return None, 0.0

    # Grounding DINO 返回的是归一化的 [cx, cy, w, h]
    # 我们需要取分数最高的一个，并转为 [x1, y1, x2, y2] 像素坐标
    best_idx = torch.argmax(logits)
    box = boxes[best_idx] * torch.Tensor([W, H, W, H])
    cx, cy, w, h = box.numpy()

    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)

    # 边界检查与极小 Padding
    pw, ph = 2, 2
    res_box = [max(0, x1 - pw), max(0, y1 - ph), min(W - 1, x2 + pw), min(H - 1, y2 + ph)]

    return res_box, float(logits[best_idx])


def main():
    # 路径配置与之前相同
    IMAGE_DIR = Path(r"E:\Datasets\Task09_Spleen_2D\test\images")
    PROMPT_DIR = Path(r"E:\Datasets\Task09_Spleen_2D\test\prompts_rag")
    VIS_DIR = Path(r"E:\Datasets\Task09_Spleen_2D\test\box_vis")
    os.makedirs(PROMPT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg'))]

    for filename in tqdm(image_files, desc="DINO 探测"):
        img_path = IMAGE_DIR / filename
        clean_id = img_path.stem.replace("_segmentation", "")

        # 获取框和置信度
        bbox, confidence = get_grounding_dino_bbox(str(img_path))

        if bbox is None:
            with open(PROMPT_DIR / f"{clean_id}_segmentation.txt", 'w') as f:
                f.write("null")
            continue

        # 保存为统一格式 [x1, y1, x2, y2]
        prompt_str = f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"

        # 这里的决策逻辑可以根据 DINO 的 confidence 来定
        # 例如：confidence > 0.5 设为 Accept，否则 Defer
        if confidence > 0.5:
            with open(PROMPT_DIR / f"{clean_id}_segmentation.txt", 'w') as f:
                f.write(prompt_str)  # 这里可以根据 RAG 逻辑补充先验 Mask 路径
        else:
            with open(PROMPT_DIR / f"{clean_id}_segmentation.txt", 'w') as f:
                f.write(prompt_str + "\nuncertain")

        # 可视化 (用于调试)
        img_cv = cv2.imread(str(img_path))
        cv2.rectangle(img_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(img_cv, f"DINO:{confidence:.2f}", (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(str(VIS_DIR / f"{clean_id}_vis.jpg"), img_cv)


if __name__ == "__main__":
    main()