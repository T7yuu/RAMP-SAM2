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
from prior_transfer import is_prior_geometry_compatible

# ================= 🚀 路由配置 =================
ROUTING_TH = 0.50  # 填入 grid_search_val 跑出的最佳阈值
# ===============================================

CLINICAL_PROMPT_TEMPLATES = {
    "ISIC2018": "dermoscopy, melanoma, skin lesion, blue-white veil, atypical pigment network, irregular borders, asymmetry, blotches",
}


def get_bbox_from_mask(mask_2d):
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    if not np.any(rows): return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [xmin, ymin, xmax, ymax]


def calculate_iou(box1, box2):
    if box1 is None or box2 is None: return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = b1_area + b2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


class FeatureAdapter(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x): return F.normalize(self.projector(x), p=2, dim=-1)


def load_biomedclip_and_adapter(device, adapter_path):
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    model, _, preprocess_val = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    adapter = FeatureAdapter().to(device)
    adapter.load_state_dict(torch.load(adapter_path, map_location=device, weights_only=True))
    return model, preprocess_val, tokenizer, adapter.eval()


def get_biomedclip_attention_bbox_and_point(model, preprocess_val, tokenizer, image, text_prompt, device):
    W, H = image.size
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
        attn_map = torch.einsum('bnd,bd->bn', F.normalize(visual_model.head(x[:, 1:, :]), dim=-1), text_features).view(
            14, 14).cpu().numpy()

    attn_map = np.clip(attn_map, 0, None)
    attn_map_resized = cv2.resize((attn_map / (attn_map.max() + 1e-8) * 255).astype(np.float32), (W, H),
                                  interpolation=cv2.INTER_CUBIC)

    Y, X = np.ogrid[:H, :W]
    gaussian_weight = np.exp(-((X - W / 2.0) ** 2 + (Y - H / 2.0) ** 2) / (2 * (min(H, W) / 2.5) ** 2))
    attn_map_resized = ((attn_map_resized * gaussian_weight) / (
            (attn_map_resized * gaussian_weight).max() + 1e-8) * 255).astype(np.uint8)

    _, _, _, max_loc = cv2.minMaxLoc(attn_map_resized)
    _, binary_map = cv2.threshold(attn_map_resized, attn_map_resized.max() * 0.60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, cls_feat_proj, text_features

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    box_area = w * h
    img_area = W * H
    aspect_ratio = w / h if h > 0 else 0

    if box_area < img_area * 0.05 or aspect_ratio > 3.0 or aspect_ratio < 0.33:
        return None, None, cls_feat_proj, text_features

    cx = int(x + w / 2.0)
    cy = int(y + h / 2.0)
    best_point = [cx, cy]

    pw, ph = max(5, int(w * 0.15)), max(5, int(h * 0.15))
    return [max(0, x - pw), max(0, y - ph), min(W - 1, x + w + pw),
            min(H - 1, y + h + ph)], best_point, cls_feat_proj, text_features


def load_knowledge_base(json_path, npz_path):
    with open(json_path, 'r', encoding='utf-8') as f: kb_data = json.load(f)
    return {item['image_path']: item for item in kb_data}, torch.tensor(np.load(npz_path)['features'],
                                                                        dtype=torch.float32), np.load(npz_path)['ids']


def main():
    import argparse
    ap = argparse.ArgumentParser(description="RAMP-SAM2 检索提示词生成")
    ap.add_argument("--adapter", default=None, help="适配器权重路径 (默认用下面的 ADAPTER_PATH)")
    ap.add_argument("--image-dir", default=None, help="查询图像目录")
    ap.add_argument("--prompt-dir", default=None, help="提示词输出目录")
    ap.add_argument("--routing-th", type=float, default=None, help="覆盖路由阈值")
    args = ap.parse_args()

    DATASET_NAME = "ISIC2018"
    IMAGE_DIR = Path(args.image_dir or os.environ.get(
        "ISIC_IMAGE_DIR", r"E:/Datasets/ISIC2018/test/images_compressed"
    ))
    # Separate revision output preserves the pre-revision prompts for auditability.
    PROMPT_DIR = Path(args.prompt_dir or os.environ.get(
        "ISIC_PROMPT_DIR", r"E:/Datasets/ISIC2018/test/prompts_rag_revised_safe"
    ))
    TEXT_LABEL = CLINICAL_PROMPT_TEMPLATES.get(DATASET_NAME, "medical image")

    KB_JSON = r"E:\PythonD\ERA3\scripts\V9\3wd2\unified_medical_kb.json"
    KB_NPZ = r"E:/PythonD/ERA3/scripts/V9/scripts/outputs_medical/medical_features.npz"
    ADAPTER_PATH = args.adapter or r"E:/PythonD/ERA3/scripts/V9/scripts/2wd_adapter.pth"

    global ROUTING_TH
    if args.routing_th is not None:
        ROUTING_TH = args.routing_th

    os.makedirs(PROMPT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"adapter={ADAPTER_PATH}\nimages={IMAGE_DIR}\nprompts={PROMPT_DIR}\nROUTING_TH={ROUTING_TH}")

    model, preprocess_val, tokenizer, adapter = load_biomedclip_and_adapter(device, ADAPTER_PATH)
    kb_dict, kb_features_raw, _ = load_knowledge_base(KB_JSON, KB_NPZ)
    kb_list = list(kb_dict.values())
    kb_features_raw_np = kb_features_raw.numpy()

    with torch.no_grad():
        kb_features_2wd = adapter(kb_features_raw.to(device)).cpu().numpy()

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg'))]
    counts = {"Accept": 0, "Defer": 0}

    for filename in tqdm(image_files, desc="生成临床增强提示词"):
        img_path = IMAGE_DIR / filename
        clean_id = Path(filename).stem.replace("_segmentation", "")
        output_path = PROMPT_DIR / f"{clean_id}_segmentation.txt"

        try:
            image = Image.open(img_path).convert("RGB")
            W, H = image.size
            candidate_bbox, best_point, query_feat_raw, text_feat = get_biomedclip_attention_bbox_and_point(model,
                                                                                                            preprocess_val,
                                                                                                            tokenizer,
                                                                                                            image,
                                                                                                            TEXT_LABEL,
                                                                                                            device)
        except Exception as e:
            continue

        if candidate_bbox is None:
            # 万一连基础框都没提取到，强制走 Defer (v5 脚本中会对 box 设容错)
            with open(output_path, 'w', encoding='utf-8') as f: f.write("0 0 10 10 5 5\nuncertain")
            counts["Defer"] += 1
            continue

        with torch.no_grad():
            query_feat_2wd = adapter(query_feat_raw).cpu().numpy().squeeze()
            text_feat_np = text_feat.cpu().numpy().squeeze()

        sim_scores_img = np.dot(kb_features_2wd, query_feat_2wd.T)
        sim_scores_text = np.dot(kb_features_raw_np, text_feat_np.T)
        sim_scores = 0.95 * sim_scores_img + 0.05 * sim_scores_text

        best_idx = np.argmax(sim_scores)
        best_sim = float(sim_scores[best_idx])
        prompt_str = f"{candidate_bbox[0]} {candidate_bbox[1]} {candidate_bbox[2]} {candidate_bbox[3]} {best_point[0]} {best_point[1]}"

        prior_mask_path = kb_list[best_idx].get('mask_path', 'none')
        spatial_conflict = False

        if best_sim >= ROUTING_TH:
            # Prior transfer is allowed only after normalized position, scale, and
            # overlap checks. This gate is conservative; it is not image registration.
            spatial_conflict = not is_prior_geometry_compatible(
                prior_mask_path, candidate_bbox, (W, H)
            )

        # 严格执行双支决策路由
        if best_sim >= ROUTING_TH and not spatial_conflict:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"{prompt_str} {prior_mask_path}")
            counts["Accept"] += 1
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(prompt_str + "\nuncertain")
            counts["Defer"] += 1

    print(f"\n✅ 提示词生成完成！二支统计: {counts}")


if __name__ == "__main__":
    main()