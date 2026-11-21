import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import open_clip

# ================= 🚀 临床医学增强提示词模板 (关键词堆叠) =================
CLINICAL_PROMPT_TEMPLATES = {
    "ISIC2018": (
        "dermoscopy, melanoma, skin lesion, blue-white veil, "
        "atypical pigment network, irregular borders, asymmetry, blotches"
    ),
    "Kvasir-SEG": (
        "endoscopy, gastrointestinal polyp, altered pit patterns, "
        "vascular intensity, mucosal protrusion, adenomatous polyp"
    )
}

class ThreeWayFeatureAdapter(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return F.normalize(self.projector(x), p=2, dim=-1)

def load_biomedclip_and_adapter(device, adapter_path):
    print("加载 BiomedCLIP 及 🚀 三支特征适配器...")
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    model, _, preprocess_val = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    adapter = ThreeWayFeatureAdapter().to(device)
    adapter.load_state_dict(torch.load(adapter_path, map_location=device, weights_only=True))
    adapter.eval()
    return model, preprocess_val, tokenizer, adapter

def get_biomedclip_attention_bbox_and_point(model, preprocess_val, tokenizer, image, text_prompt, device):
    W, H = image.size
    img_tensor = preprocess_val(image).unsqueeze(0).to(device)
    text_tokens = tokenizer([text_prompt]).to(device)

    with torch.no_grad():
        text_features = F.normalize(model.encode_text(text_tokens), dim=-1)
        visual_model = model.visual
        x = visual_model.trunk.patch_embed(img_tensor)
        cls_token = visual_model.trunk.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = visual_model.trunk.pos_drop(x + visual_model.trunk.pos_embed)
        for blk in visual_model.trunk.blocks: x = blk(x)
        x = visual_model.trunk.norm(x)
        patch_feats_proj = F.normalize(visual_model.head(x[:, 1:, :]), dim=-1)
        cls_feat_proj = F.normalize(visual_model.head(x[:, 0, :]), dim=-1)
        sim_scores = torch.einsum('bnd,bd->bn', patch_feats_proj, text_features)
        attn_map = sim_scores.view(14, 14).cpu().numpy()

    attn_map = np.clip(attn_map, 0, None)
    attn_map = (attn_map / (attn_map.max() + 1e-8) * 255).astype(np.float32)
    attn_map_resized = cv2.resize(attn_map, (W, H), interpolation=cv2.INTER_CUBIC)

    Y, X = np.ogrid[:H, :W]
    cy, cx = H / 2.0, W / 2.0
    sigma = min(H, W) / 2.5
    gaussian_weight = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
    attn_map_resized = (attn_map_resized * gaussian_weight)
    attn_map_resized = (attn_map_resized / (attn_map_resized.max() + 1e-8) * 255).astype(np.uint8)

    _, _, _, max_loc = cv2.minMaxLoc(attn_map_resized)
    point_x, point_y = max_loc

    thresh_val = attn_map_resized.max() * 0.60
    _, binary_map = cv2.threshold(attn_map_resized, thresh_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours: return None, None, cls_feat_proj

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    pad_w, pad_h = max(2, int(w * 0.03)), max(2, int(h * 0.03))
    candidate_bbox = [max(0, x - pad_w), max(0, y - pad_h), min(W - 1, x + w + pad_w), min(H - 1, y + h + pad_h)]
    return candidate_bbox, (point_x, point_y), cls_feat_proj

def load_knowledge_base(json_path, npz_path):
    with open(json_path, 'r', encoding='utf-8') as f: kb_data = json.load(f)
    kb_dict = {item['image_path']: item for item in kb_data}
    features_data = np.load(npz_path)
    return kb_dict, torch.tensor(features_data['features'], dtype=torch.float32), features_data['ids']

def main():
    DATASET_NAME = "ISIC2018"
    IMAGE_DIR = Path(r"E:/Datasets/ISIC2018/test/images_compressed")
    PROMPT_DIR = Path(r"E:/Datasets/ISIC2018/test/prompts_rag")
    TEXT_LABEL = CLINICAL_PROMPT_TEMPLATES.get(DATASET_NAME, "medical image of lesion")

    KB_JSON = r"E:\PythonD\ERA3\scripts\V9\scripts\unified_medical_kb.json"
    KB_NPZ = r"E:/PythonD/ERA3/scripts/V9/scripts/outputs_medical/medical_features.npz"
    ADAPTER_PATH = r"E:/PythonD/ERA3/scripts/V9/scripts/3wd_adapter.pth"

    SIM_ACCEPT_TH, SIM_DEFER_TH = 0.98, 0.70
    os.makedirs(PROMPT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess_val, tokenizer, adapter = load_biomedclip_and_adapter(device, ADAPTER_PATH)
    kb_dict, kb_features_raw, kb_ids = load_knowledge_base(KB_JSON, KB_NPZ)

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
        except:
            continue

        candidate_bbox, best_point, query_feat_raw = get_biomedclip_attention_bbox_and_point(
            model, preprocess_val, tokenizer, image, TEXT_LABEL, device
        )

        if candidate_bbox is None:
            with open(output_path, 'w', encoding='utf-8') as f: f.write("null")
            counts["Reject"] += 1
            continue

        with torch.no_grad():
            query_feat_3wd = adapter(query_feat_raw).cpu().numpy().squeeze()

        sim_scores = np.dot(kb_features_3wd, query_feat_3wd.T)
        best_sim = np.max(sim_scores)

        prompt_str = f"{candidate_bbox[0]} {candidate_bbox[1]} {candidate_bbox[2]} {candidate_bbox[3]} {best_point[0]} {best_point[1]}"
        area_ratio = ((candidate_bbox[2] - candidate_bbox[0]) * (candidate_bbox[3] - candidate_bbox[1])) / (W * H)

        if best_sim >= SIM_ACCEPT_TH and 0.20 < area_ratio < 0.70:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(prompt_str)
            counts["Accept"] += 1
        elif best_sim >= SIM_DEFER_TH:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(prompt_str + "\nuncertain")
            counts["Defer"] += 1
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("null")
            counts["Reject"] += 1

    print(f"\n✅ 提示词生成完成！三支统计: {counts}")

if __name__ == "__main__":
    main()