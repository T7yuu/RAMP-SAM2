import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm
import open_clip

# ================= 🚀 配置区域 =================
SIM_ACCEPT_TH = 0.92
SIM_DEFER_TH = 0.6
# ===============================================

CLINICAL_PROMPT_TEMPLATES = {
    # 🌟 优化点：简化提示词，减少噪声干扰
    "CVC-ClinicDB": "a colon polyp"
}


class ThreeWayFeatureAdapter(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x): return F.normalize(self.projector(x), p=2, dim=-1)


def load_biomedclip_and_adapter(device, adapter_path):
    print("加载 BiomedCLIP 及 🚀 三支特征适配器...")
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    model, _, preprocess_val = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    adapter = ThreeWayFeatureAdapter().to(device)
    # 确保加载权重
    if os.path.exists(adapter_path):
        adapter.load_state_dict(torch.load(adapter_path, map_location=device, weights_only=True))
    return model, preprocess_val, tokenizer, adapter.eval()


def get_biomedclip_attention_bbox_and_point(model, preprocess_val, tokenizer, image, text_prompt, device):
    W, H = image.size
    img_area = W * H
    img_tensor = preprocess_val(image).unsqueeze(0).to(device)

    with torch.no_grad():
        text_features = F.normalize(model.encode_text(tokenizer([text_prompt]).to(device)), dim=-1)
        visual_model = model.visual
        x = visual_model.trunk.pos_drop(torch.cat(
            (visual_model.trunk.cls_token.expand(visual_model.trunk.patch_embed(img_tensor).shape[0], -1, -1),
             visual_model.trunk.patch_embed(img_tensor)), dim=1) + visual_model.trunk.pos_embed)
        for blk in visual_model.trunk.blocks: x = blk(x)
        x = visual_model.trunk.norm(x)
        cls_feat_proj = F.normalize(visual_model.head(x[:, 0, :]), dim=-1)

        # 提取 14x14 注意力图
        attn_map = torch.einsum('bnd,bd->bn', F.normalize(visual_model.head(x[:, 1:, :]), dim=-1), text_features).view(
            14, 14).cpu().numpy()

    # 1. 基础处理：归一化与缩放
    attn_map = np.clip(attn_map, 0, None)
    attn_map_resized = cv2.resize(attn_map, (W, H), interpolation=cv2.INTER_CUBIC)

    # 2. 视野 (FOV) 过滤：剔除内窥镜图像边缘的黑边
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, fov_mask = cv2.threshold(img_gray, 15, 1, cv2.THRESH_BINARY)
    fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    attn_map_resized = attn_map_resized * fov_mask

    # 🌟 3. 核心修复：高斯平滑。通过 51x51 的大核平滑，消除 ViT 的块状感和反光干扰
    attn_map_resized = cv2.GaussianBlur(attn_map_resized, (51, 51), 0)

    if attn_map_resized.max() > 0:
        attn_map_resized = (attn_map_resized / attn_map_resized.max() * 255).astype(np.uint8)
    else:
        return None, cls_feat_proj, text_features

    # 4. 动态多级阈值搜索最佳轮廓
    best_x, best_y, best_w, best_h = 0, 0, 0, 0
    valid_box_found = False

    # 从最热的核心区域开始向外搜索
    for th_ratio in np.arange(0.90, 0.20, -0.05):
        _, binary_map = cv2.threshold(attn_map_resized, int(255 * th_ratio), 255, cv2.THRESH_BINARY)

        # 使用闭运算连接可能碎裂的病灶区域
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue

        # 🌟 修复点：在有效候选轮廓中选择最像病灶的一个
        valid_candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0

            # 过滤逻辑：面积不能太小（噪点）也不能太大（背景）；比例不能太极端
            if area < img_area * 0.005 or area > img_area * 0.45: continue
            if aspect_ratio > 3.5 or aspect_ratio < 0.28: continue
            valid_candidates.append((cnt, area))

        if valid_candidates:
            # 选取当前阈值下最大的合理轮廓
            best_cnt = max(valid_candidates, key=lambda x: x[1])[0]
            best_x, best_y, best_w, best_h = cv2.boundingRect(best_cnt)
            valid_box_found = True
            # 如果框的尺寸已经达到一定规模（3%），认为已经包围了病灶主体，停止扩张
            if (best_w * best_h) > img_area * 0.03:
                break

    if not valid_box_found:
        return None, cls_feat_proj, text_features

    # 5. 生成最终 BBox 并略微外扩 (15%) 给 SAM2 留出边缘空间
    pw, ph = max(5, int(best_w * 0.15)), max(5, int(best_h * 0.15))
    final_bbox = [
        max(0, best_x - pw),
        max(0, best_y - ph),
        min(W - 1, best_x + best_w + pw),
        min(H - 1, best_y + best_h + ph)
    ]

    return final_bbox, cls_feat_proj, text_features


def load_knowledge_base(json_path, npz_path):
    with open(json_path, 'r', encoding='utf-8') as f: kb_data = json.load(f)
    features = np.load(npz_path)['features']
    ids = np.load(npz_path)['ids']
    return {item['image_path']: item for item in kb_data}, torch.tensor(features, dtype=torch.float32), ids


def main():
    DATASET_NAME = "CVC-ClinicDB"
    # 根据你的文件系统路径进行微调
    IMAGE_DIR = Path(rf"E:/Datasets/{DATASET_NAME}_Split/test/images_compressed")
    PROMPT_DIR = Path(rf"E:/Datasets/{DATASET_NAME}_Split/test/prompts_rag")
    VIS_DIR = Path(rf"E:/Datasets/{DATASET_NAME}_Split/test/vis_prompts")
    KB_JSON = r"E:\PythonD\ERA3\scripts\V9\CVC-ClinicDB\unified_medical_kb.json"
    KB_NPZ = r"E:/PythonD/ERA3/scripts/V9/scripts/outputs_medical/medical_features.npz"
    ADAPTER_PATH = r"E:/PythonD/ERA3/scripts/V9/scripts/3wd_adapter.pth"

    os.makedirs(PROMPT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    TEXT_LABEL = CLINICAL_PROMPT_TEMPLATES.get(DATASET_NAME, "medical image")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess_val, tokenizer, adapter = load_biomedclip_and_adapter(device, ADAPTER_PATH)
    kb_dict, kb_features_raw, _ = load_knowledge_base(KB_JSON, KB_NPZ)
    kb_list = list(kb_dict.values())
    kb_features_raw_np = kb_features_raw.numpy()

    with torch.no_grad():
        kb_features_3wd = adapter(kb_features_raw.to(device)).cpu().numpy()

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg'))]
    counts = {"Accept": 0, "Defer": 0, "Reject": 0}

    for filename in tqdm(image_files, desc="生成临床增强提示词"):
        img_path = IMAGE_DIR / filename
        clean_id = Path(filename).stem.replace("_segmentation", "")
        output_path = PROMPT_DIR / f"{clean_id}_segmentation.txt"

        try:
            image = Image.open(img_path).convert("RGB")
            W, H = image.size
            candidate_bbox, query_feat_raw, text_feat = get_biomedclip_attention_bbox_and_point(
                model, preprocess_val, tokenizer, image, TEXT_LABEL, device
            )
        except Exception as e:
            tqdm.write(f"⚠️ 处理 {filename} 失败: {e}")
            continue

        decision = "Reject"
        if candidate_bbox is not None:
            with torch.no_grad():
                query_feat_3wd = adapter(query_feat_raw).cpu().numpy().squeeze()
                text_feat_np = text_feat.cpu().numpy().squeeze()

            # 三支决策相似度计算
            sim_scores_img = np.dot(kb_features_3wd, query_feat_3wd.T)
            sim_scores_text = np.dot(kb_features_raw_np, text_feat_np.T)
            sim_scores = 0.90 * sim_scores_img + 0.10 * sim_scores_text
            best_idx = np.argmax(sim_scores)
            best_sim = float(sim_scores[best_idx])

            prompt_str = f"{int(candidate_bbox[0])} {int(candidate_bbox[1])} {int(candidate_bbox[2])} {int(candidate_bbox[3])}"

            if best_sim >= SIM_ACCEPT_TH:
                prior_mask_path = kb_list[best_idx].get('mask_path', 'none')
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"{prompt_str} {prior_mask_path}")
                decision = "Accept"
            elif best_sim >= SIM_DEFER_TH:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"{prompt_str}\nuncertain")
                decision = "Defer"
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("null")
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("null")

        counts[decision] += 1

        # 保存可视化结果
        draw = ImageDraw.Draw(image)
        color_map = {"Accept": "green", "Defer": "orange", "Reject": "red"}
        if candidate_bbox is not None:
            draw.rectangle(candidate_bbox, outline=color_map.get(decision, "red"), width=4)
        image.save(VIS_DIR / f"{clean_id}_vis_{decision}.jpg")

    print(f"\n✅ 提示词生成完成！三支统计: {counts}")


if __name__ == "__main__":
    main()