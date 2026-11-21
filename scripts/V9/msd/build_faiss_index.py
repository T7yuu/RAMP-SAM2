import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import open_clip

KNOWLEDGE_BASE_PATH = "./unified_medical_kb.json"
OUTPUT_DIR = "./outputs_medical/"


def build_feature_matrix():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("加载 BiomedCLIP (医疗专精视觉语言模型)...")
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    model, _, preprocess_val = open_clip.create_model_and_transforms(model_name)
    model = model.to(device).eval()

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    features_output_path = output_path / "medical_features.npz"

    with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)

    all_features, valid_kb_ids = [], []
    with torch.no_grad():
        for entry in tqdm(knowledge_base, desc="提取特征"):
            try:
                image_path = entry.get('image_path')
                if not image_path or not os.path.exists(image_path): continue

                image = Image.open(image_path).convert('RGB')
                img_tensor = preprocess_val(image).unsqueeze(0).to(device)
                image_feature = torch.nn.functional.normalize(model.encode_image(img_tensor), dim=-1)

                all_features.append(image_feature.cpu().numpy().squeeze())
                valid_kb_ids.append(str(Path(image_path).as_posix()))
            except Exception as e:
                continue

    features_matrix = np.array(all_features, dtype=np.float32)
    np.savez(features_output_path, features=features_matrix, ids=np.array(valid_kb_ids))
    print(f"✅ BiomedCLIP 特征矩阵保存成功: {features_output_path}")


if __name__ == '__main__':
    build_feature_matrix()