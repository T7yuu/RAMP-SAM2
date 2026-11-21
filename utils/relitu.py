import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ================= 🚀 配置路径 =================
DATASET_NAME = "ISIC2018"
BASE_DIR = Path(r"E:/Datasets/ISIC2018/test")

IMAGE_DIR = BASE_DIR / "images_compressed"
PROMPT_DIR = BASE_DIR / "prompts_rag"
MASK_DIR = Path(rf"E:/PythonD/ERA3/scripts/V9/outputs_v5_{DATASET_NAME}")
HEATMAP_DIR = BASE_DIR / "heatmaps"  # 如果你保存了热力图，填这个路径

OUTPUT_VIS_DIR = Path(rf"E:/PythonD/ERA3/scripts/V9/visualizations_{DATASET_NAME}")
MAX_SAMPLES = 50  # 只可视化前 50 张


# ===============================================

def load_prompt_box(prompt_file):
    """解析 .txt 文件，提取 x1, y1, x2, y2 格式的边界框"""
    if not prompt_file.exists():
        return None, "Reject"

    with open(prompt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines or lines[0].strip().lower() == "null":
        return None, "Reject"

    decision = "Defer" if (len(lines) > 1 and "uncertain" in lines[1].lower()) else "Accept"
    parts = lines[0].strip().split()

    try:
        # 严格按照 [x1, y1, x2, y2] 格式提取
        box = [int(float(p)) for p in parts[:4]]
        return box, decision
    except:
        return None, "Reject"


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """在原图上叠加半透明的掩码"""
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def main():
    OUTPUT_VIS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🚀 开始生成可视化对比图 (限制 {MAX_SAMPLES} 张)...")

    image_files = sorted([f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in ('.jpg', '.png', '.jpeg')])

    count = 0
    for img_path in tqdm(image_files, desc="Processing"):
        if count >= MAX_SAMPLES:
            break

        clean_id = img_path.stem.replace("_segmentation", "")

        # 1. 读取原图
        img = cv2.imread(str(img_path))
        if img is None: continue
        H, W = img.shape[:2]

        # 2. 读取并绘制边界框 (Box)
        prompt_file = PROMPT_DIR / f"{clean_id}_segmentation.txt"
        box, decision = load_prompt_box(prompt_file)

        img_with_box = img.copy()
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (255, 0, 0), 3)  # 蓝色框
            cv2.putText(img_with_box, f"Box ({decision})", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        else:
            cv2.putText(img_with_box, "Reject (No Box)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # 3. 读取掩码图 (Mask) 并叠加
        mask_path = MASK_DIR / f"{clean_id}.jpg"  # 注意你的掩码是存为 jpg 还是 png
        if not mask_path.exists():
            mask_path = MASK_DIR / f"{clean_id}.png"

        img_with_mask = img.copy()
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape != (H, W):
                    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                img_with_mask = overlay_mask(img, mask, color=(0, 255, 0))  # 绿色掩码

        # 4. 可选：读取热力图 (Heatmap)
        heatmap_path = HEATMAP_DIR / f"{clean_id}.jpg"
        if heatmap_path.exists():
            heatmap = cv2.imread(str(heatmap_path))
            heatmap = cv2.resize(heatmap, (W, H))
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            img_with_heatmap = cv2.addWeighted(heatmap_colored, 0.6, img, 0.4, 0)

            # 拼接: 原图+框 | 热力图 | 掩码图
            final_vis = np.hstack((img_with_box, img_with_heatmap, img_with_mask))
        else:
            # 拼接: 原图+框 | 掩码图
            final_vis = np.hstack((img_with_box, img_with_mask))

        # 保存结果
        save_path = OUTPUT_VIS_DIR / f"vis_{clean_id}.jpg"
        cv2.imwrite(str(save_path), final_vis)
        count += 1

    print(f"\n✅ 可视化完成！请前往 {OUTPUT_VIS_DIR} 查看结果。")


if __name__ == "__main__":
    main()