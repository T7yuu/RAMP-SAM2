import os
import json
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import open_clip

# 🚀 环境变量配置：解决 HuggingFace 下载卡顿和警告
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

def load_biomedclip(device):
    print("加载 BiomedCLIP (医疗专精大模型)...")
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    model, _, preprocess_val = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)

    model = model.to(device)
    model.eval()
    return model, preprocess_val, tokenizer

def get_biomedclip_attention_bbox(model, preprocess_val, tokenizer, image, text_prompt, device):
    W, H = image.size
    img_tensor = preprocess_val(image).unsqueeze(0).to(device)
    text_tokens = tokenizer([text_prompt]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

        visual_model = model.visual
        x = visual_model.trunk.patch_embed(img_tensor)
        cls_token = visual_model.trunk.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = visual_model.trunk.pos_drop(x + visual_model.trunk.pos_embed)

        for blk in visual_model.trunk.blocks:
            x = blk(x)
        x = visual_model.trunk.norm(x)

        patch_feats = x[:, 1:, :]
        cls_feat_raw = x[:, 0, :]

        # 正确调用 Linear 神经网络层
        patch_feats_proj = visual_model.head(patch_feats)
        cls_feat_proj = visual_model.head(cls_feat_raw)

        patch_feats_proj = torch.nn.functional.normalize(patch_feats_proj, dim=-1)
        cls_feat_proj = torch.nn.functional.normalize(cls_feat_proj, dim=-1)

        sim_scores = torch.einsum('bnd,bd->bn', patch_feats_proj, text_features)
        attn_map = sim_scores.view(14, 14).cpu().numpy()

    # 🚀 致命数学陷阱修复：标准的 Min-Max 归一化，彻底消除全图高亮底噪！
    attn_min = attn_map.min()
    attn_max = attn_map.max()
    attn_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-8)
    attn_map = (attn_map * 255).astype(np.float32)

    attn_map_resized = cv2.resize(attn_map, (W, H), interpolation=cv2.INTER_CUBIC)

    # 高斯中心先验：压制四周的皮肤镜暗角阴影
    Y, X = np.ogrid[:H, :W]
    cy, cx = H / 2.0, W / 2.0
    sigma = min(H, W) / 2.5
    gaussian_weight = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

    attn_map_resized = attn_map_resized * gaussian_weight
    attn_map_resized = (attn_map_resized / attn_map_resized.max() * 255).astype(np.uint8)

    # 🚀 放弃固定阈值，改用极其强大的 Otsu (大津法) 自适应阈值
    # 🚀 致命修复：抛弃 Otsu！
    # 直接取 BiomedCLIP 注意力热力图“最热的 20% 区域”（山顶）
    # 这会强制生成一个非常小、非常紧凑、绝对处于病灶核心的 Bounding Box
    thresh_val = attn_map_resized.max() * 0.80
    _, binary_map = cv2.threshold(attn_map_resized, thresh_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, cls_feat_proj.cpu().numpy()

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 10% 黄金呼吸空间
    pad_w = int(w * 0.10)
    pad_h = int(h * 0.10)

    candidate_bbox = [
        max(0, x - pad_w),
        max(0, y - pad_h),
        min(W - 1, x + w + pad_w),
        min(H - 1, y + h + pad_h)
    ]

    return candidate_bbox, cls_feat_proj.cpu().numpy()

def load_knowledge_base(json_path, npz_path):
    print("加载统一知识库与特征矩阵...")
    with open(json_path, 'r', encoding='utf-8') as f:
        kb_data = json.load(f)

    kb_dict = {item['image_path']: item for item in kb_data}
    features_data = np.load(npz_path)
    kb_features = features_data['features']
    kb_ids = features_data['ids']

    norms = np.linalg.norm(kb_features, axis=1, keepdims=True)
    kb_features = kb_features / (norms + 1e-8)
    return kb_dict, kb_features, kb_ids

def main():
    DATASET_NAME = "ISIC2018"
    IMAGE_DIR = Path(r"E:/Datasets/ISIC2018/test/images_compressed")
    PROMPT_DIR = Path(r"E:/Datasets/ISIC2018/test/prompts_rag")
    TEXT_LABEL = "melanoma skin lesion"

    KB_JSON = r"E:\PythonD\ERA3\scripts\V9\scripts\unified_medical_kb.json"
    KB_NPZ = r"E:/PythonD/ERA3/scripts/V9/scripts/outputs_medical/medical_features.npz"

    # 🚀 降维打击：匹配真实的 BiomedCLIP 特征相似度分布
    SIM_ACCEPT_TH = 0.58
    SIM_DEFER_TH = 0.45

    os.makedirs(PROMPT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess_val, tokenizer = load_biomedclip(device)
    kb_dict, kb_features, kb_ids = load_knowledge_base(KB_JSON, KB_NPZ)

    counts = {"Accept": 0, "Defer": 0, "Reject": 0}
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg'))]

    for filename in tqdm(image_files, desc="基于 BiomedCLIP 构建三支决策提示"):
        img_path = IMAGE_DIR / filename
        clean_id = Path(filename).stem.replace("_segmentation", "")
        output_path = PROMPT_DIR / f"{clean_id}_segmentation.txt"

        try:
            image = Image.open(img_path).convert("RGB")
            W, H = image.size
        except Exception:
            continue

        candidate_bbox, query_feat = get_biomedclip_attention_bbox(
            model, preprocess_val, tokenizer, image, TEXT_LABEL, device
        )

        if candidate_bbox is None:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("null")
            counts["Reject"] += 1
            continue

        query_feat_norm = query_feat / (np.linalg.norm(query_feat) + 1e-8)
        sim_scores = np.dot(kb_features, query_feat_norm.T).squeeze()
        best_idx = np.argmax(sim_scores)
        best_sim = sim_scores[best_idx]

        best_kb_id = kb_ids[best_idx]
        best_kb_meta = kb_dict.get(best_kb_id, None)

        semantic_match = (best_kb_meta is not None)

        bbox_str = f"{candidate_bbox[0]} {candidate_bbox[1]} {candidate_bbox[2]} {candidate_bbox[3]}"
        box_area_ratio = ((candidate_bbox[2] - candidate_bbox[0]) * (candidate_bbox[3] - candidate_bbox[1])) / (W * H)

        # 🚀 彻底废除 pos_dist（位置约束），放宽面积上限到 0.70
        if semantic_match and best_sim >= SIM_ACCEPT_TH and box_area_ratio < 0.70:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(bbox_str)
            counts["Accept"] += 1

        elif semantic_match and best_sim >= SIM_DEFER_TH:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(bbox_str + "\nuncertain")
            counts["Defer"] += 1

        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("null")
            counts["Reject"] += 1

    print(f"\n✅ 提示标签生成完成！")
    print(f"📊 三支分布统计: {counts}")

if __name__ == "__main__":
    main()