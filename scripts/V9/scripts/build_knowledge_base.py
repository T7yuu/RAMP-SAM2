import os
import sys
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

ROOT_DATA_DIR = r"E:\Datasets"

'''DATASET_CONFIGS = [

    {
        "type": "msd",
        "name": "MSD_Task02_Heart",
        "path": r"MSD\Task02_Heart",
        "text_label": "心脏影像的切片"
    },
    {
        "type": "msd",
        "name": "MSD_Task04_Hippocampus",
        "path": r"MSD\Task04_Hippocampus",
        "text_label": "海马体影像的切片"
    },
    {
        "type": "msd",
        "name": "MSD_Task05_Prostate",
        "path": r"MSD\Task05_Prostate",
        "text_label": "前列腺影像的切片"
    },
    {
        "type": "msd",
        "name": "MSD_Task09_Spleen",
        "path": r"MSD\Task09_Spleen",
        "text_label": "脾脏影像的切片"
    },

    {
        "type": "isic",
        "name": "ISIC2018",
        "path": r"ISIC2018",
        "text_label": "皮肤上的黑色斑块"
    },
    {
        "type": "brats",
        "name": "BraTS2021",
        "path": r"BraTS\BraTS2021",
        "text_label": "脑肿瘤影像"
    }
]'''

DATASET_CONFIGS = [
    {
        "type": "isic",
        "name": "ISIC2018",
        "path": r"ISIC2018",
        "text_label": "皮肤上的黑色斑块"
    },
    {
        "type": "kvasir",
        "name": "Kvasir-SEG",
        "path": r"Kvasir-SEG",
        "text_label": "胃肠道息肉"
    }
]

OUTPUT_JSON_PATH = "./unified_medical_kb.json"

def calculate_bbox_from_mask(mask_path):
    try:
        with Image.open(mask_path) as mask_image:
            grayscale_mask = mask_image.convert('L')
            np_mask = np.array(grayscale_mask)

        rows, cols = np.where(np_mask > 0)

        if rows.size == 0:
            return None

        x_min, y_min = int(np.min(cols)), int(np.min(rows))
        x_max, y_max = int(np.max(cols)), int(np.max(rows))

        width = x_max - x_min + 1
        height = y_max - y_min + 1

        return [x_min, y_min, width, height]
    except Exception as e:
        tqdm.write(f"\n[警告] 处理掩码文件 {mask_path} 时发生错误: {e}")
        return None


def process_msd_task(config):
    task_path = os.path.join(ROOT_DATA_DIR, config['path'])
    text_label = config['text_label']
    records = []

    images_dir = os.path.join(task_path, 'imagesTr')
    labels_dir = os.path.join(task_path, 'labelsTr')

    if not os.path.isdir(images_dir): return []

    image_paths = []
    for root, _, files in os.walk(images_dir):
        for name in files:
            if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, name))

    for image_path in tqdm(image_paths, desc=f"处理 {config['name']}", leave=False):
        label_path_str = image_path.replace('imagesTr', 'labelsTr')
        if os.path.exists(label_path_str):
            bbox = calculate_bbox_from_mask(label_path_str)
            if bbox:
                records.append({
                    "image_path": str(Path(image_path).as_posix()),
                    "text": text_label,
                    "box": bbox
                })
    return records


def process_kvasir_dataset(config):
    dataset_path = os.path.join(ROOT_DATA_DIR, config['path'])
    text_label = config['text_label']
    records = []

    # Kvasir-SEG 标准结构通常是 images 和 masks 文件夹
    images_dir = os.path.join(dataset_path, 'images')
    masks_dir = os.path.join(dataset_path, 'masks')

    if not os.path.isdir(images_dir):
        tqdm.write(f"错误: 找不到路径 {images_dir}")
        return []

    file_list = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_filename in tqdm(file_list, desc=f"处理 {config['name']}", leave=False):
        image_path = os.path.join(images_dir, image_filename)

        # Kvasir 的掩码文件名通常与原图完全一致，但后缀可能是 .png 或 .jpg
        # 这里做一个简单的兼容处理
        mask_filename = image_filename
        if not os.path.exists(os.path.join(masks_dir, mask_filename)):
            # 如果找不到同名文件，尝试换成 .png (Kvasir 常用)
            mask_filename = Path(image_filename).stem + ".png"

        mask_path = os.path.join(masks_dir, mask_filename)

        if os.path.exists(mask_path):
            bbox = calculate_bbox_from_mask(mask_path)
            if bbox:
                records.append({
                    "image_path": str(Path(image_path).as_posix()),
                    "text": text_label,
                    "box": bbox
                })
    return records

def process_isic_dataset(config):
    dataset_path = os.path.join(ROOT_DATA_DIR, config['path'])
    text_label = config['text_label']
    records = []

    # 修改点：将原先的 ['train', 'test', 'validate'] 改为只包含 'train'
    for subset in ['train']:
        images_dir = os.path.join(dataset_path, subset, 'images_compressed')
        groundtruth_dir = os.path.join(dataset_path, subset, 'groundtruth')

        if not os.path.isdir(images_dir):
            tqdm.write(f"[提示] 跳过不存在的目录: {images_dir}")
            continue

        file_list = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_filename in tqdm(file_list, desc=f"处理 {config['name']}/{subset}", leave=False):
            image_path = os.path.join(images_dir, image_filename)

            base_name = Path(image_filename).stem
            # ISIC2018 的标注文件名通常是 原文件名 + _segmentation.png
            mask_filename = f"{base_name}_segmentation.png"
            mask_path = os.path.join(groundtruth_dir, mask_filename)

            if os.path.exists(mask_path):
                bbox = calculate_bbox_from_mask(mask_path)
                if bbox:
                    records.append({
                        "image_path": str(Path(image_path).as_posix()),
                        "text": text_label,
                        "box": bbox
                    })
    return records


def process_brats_dataset(config):
    dataset_path = os.path.join(ROOT_DATA_DIR, config['path'])
    text_label = config['text_label']
    records = []

    images_dir = os.path.join(dataset_path, 'images')
    masks_dir = os.path.join(dataset_path, 'masks')

    if not os.path.isdir(images_dir): return []

    for image_filename in tqdm(os.listdir(images_dir), desc=f"处理 {config['name']}", leave=False):
        if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue

        image_path = os.path.join(images_dir, image_filename)
        # 假设BraTS的mask和image文件名完全相同
        mask_path = os.path.join(masks_dir, image_filename)

        if os.path.exists(mask_path):
            bbox = calculate_bbox_from_mask(mask_path)
            if bbox:
                records.append({
                    "image_path": str(Path(image_path).as_posix()),
                    "text": text_label,
                    "box": bbox
                })
    return records


def main():
    print("======================================================")
    print("      多源医学影像知识库统一构建脚本 v2.0")
    print("======================================================")

    if not os.path.isdir(ROOT_DATA_DIR):
        print(f"错误: 根目录 '{ROOT_DATA_DIR}' 不存在。请检查路径。")
        sys.exit(1)

    knowledge_base = {}

    handler_map = {
        'msd': process_msd_task,
        'isic': process_isic_dataset,
        'brats': process_brats_dataset,
        'kvasir': process_kvasir_dataset  # 注册新处理器
    }

    for config in DATASET_CONFIGS:
        dataset_type = config.get('type')
        if dataset_type in handler_map:
            print(f"\n>>>>> 开始处理数据集类型: '{dataset_type.upper()}', 名称: '{config['name']}' <<<<<")
            new_records = handler_map[dataset_type](config)

            for record in new_records:
                record_id = str(Path(record["image_path"]).as_posix())
                knowledge_base[record_id] = record

            print(f">>>>> '{config['name']}' 处理完成，更新/新增 {len(new_records)} 条记录。 <<<<<")
        else:
            print(f"\n警告: 未知的处理器类型 '{dataset_type}'，跳过 '{config['name']}'。")

    print("\n======================================================")
    print("所有数据集处理完毕！")
    print(f"  - 成功生成知识库总记录数: {len(knowledge_base)}")

    output_file = Path(OUTPUT_JSON_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n正在将统一知识库保存到: {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(list(knowledge_base.values()), f, indent=2, ensure_ascii=False)
        print(f"'{output_file}' 保存成功！")
    except Exception as e:
        print(f"!!!!!! 保存JSON文件时发生严重错误: {e}")

    print("======================================================")


if __name__ == "__main__":
    main()