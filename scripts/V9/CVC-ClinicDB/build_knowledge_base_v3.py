import os
import sys
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

ROOT_DATA_DIR = r"E:\Datasets"

DATASET_CONFIGS = [
    {
        "type": "CVC",  # 由于 CVC 划分后的结构与 ISIC 一致，可直接复用 isic 的 handler
        "name": "CVC-ClinicDB",
        "path": r"CVC-ClinicDB_Split",
        "text_label": "endoscopy polyp, colon cancer"
    }
]

OUTPUT_JSON_PATH = "./unified_medical_kb.json"


def calculate_bbox_from_mask(mask_path):
    try:
        with Image.open(mask_path) as mask_image:
            img_w, img_h = mask_image.size
            grayscale_mask = mask_image.convert('L')
            np_mask = np.array(grayscale_mask)

        rows, cols = np.where(np_mask > 0)
        if rows.size == 0:
            return None, img_w, img_h

        x_min, y_min = int(np.min(cols)), int(np.min(rows))
        x_max, y_max = int(np.max(cols)), int(np.max(rows))

        # 🚀 修复核心：直接返回统一的 [x1, y1, x2, y2] 格式，废弃宽高计算
        return [x_min, y_min, x_max, y_max], img_w, img_h

    except Exception as e:
        tqdm.write(f"\n[警告] 处理掩码文件 {mask_path} 时发生错误: {e}")
        return None, None, None


def process_isic_dataset(config):
    dataset_path = os.path.join(ROOT_DATA_DIR, config['path'])
    text_label = config['text_label']
    records = []

    for subset in ['train']:
        images_dir = os.path.join(dataset_path, subset, 'images_compressed')
        groundtruth_dir = os.path.join(dataset_path, subset, 'groundtruth')

        if not os.path.isdir(images_dir): continue
        file_list = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_filename in tqdm(file_list, desc=f"处理 {config['name']}/{subset}", leave=False):
            image_path = os.path.join(images_dir, image_filename)
            base_name = Path(image_filename).stem
            mask_filename = f"{base_name}_segmentation.png"
            mask_path = os.path.join(groundtruth_dir, mask_filename)

            if os.path.exists(mask_path):
                bbox, img_w, img_h = calculate_bbox_from_mask(mask_path)
                if bbox is not None:
                    records.append({
                        "image_path": str(Path(image_path).as_posix()),
                        "mask_path": str(Path(mask_path).as_posix()),  # ⭐ 新增
                        "text": text_label,
                        "box": bbox,
                        "image_width": img_w,
                        "image_height": img_h
                    })
    return records


def main():
    print("======================================================")
    print("      多源医学影像知识库统一构建脚本 v3.1 (含Mask)")
    print("======================================================")

    knowledge_base = {}
    handler_map = { 'CVC': process_isic_dataset}

    for config in DATASET_CONFIGS:
        dataset_type = config.get('type')
        if dataset_type in handler_map:
            print(f"\n>>>>> 开始处理 '{config['name']}' <<<<<")
            new_records = handler_map[dataset_type](config)
            for record in new_records:
                knowledge_base[str(Path(record["image_path"]).as_posix())] = record
            print(f">>>>> 新增 {len(new_records)} 条记录 <<<<<")

    output_file = Path(OUTPUT_JSON_PATH)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(list(knowledge_base.values()), f, indent=2, ensure_ascii=False)
    print(f"\n✅ 知识库保存成功: {output_file}")


if __name__ == "__main__":
    main()