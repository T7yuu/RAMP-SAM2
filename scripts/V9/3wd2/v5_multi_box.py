import sys
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sam2_repo_path = r"E:\PythonD\ERA\sam2_code"
if sam2_repo_path not in sys.path: sys.path.insert(0, sam2_repo_path)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from prior_transfer import prepare_aligned_prior

USE_ENSEMBLE = True
MAX_MASK_AREA_RATIO = 0.995
MIN_MASK_AREA_RATIO = 0.001
MAX_PROMPT_BOX_RATIO = 0.70

DATASET_NAME = "ISIC2018"
BASE_DIR = Path(r"E:/Datasets/ISIC2018/test")
IMAGE_DIR = BASE_DIR / "images_compressed"
PROMPT_DIR = BASE_DIR / "prompts_rag_revised_safe"
GT_DIR = BASE_DIR / "groundtruth"
OUTPUT_DIR = Path(rf"E:/PythonD/ERA3/scripts/V9/outputs_v5_{DATASET_NAME}_revised_safe_{'ensemble' if USE_ENSEMBLE else 'single'}")

SAM2_MODEL_CFG = Path(r"E:\PythonD\ERA\sam2_weights\config.yaml")
SAM2_CHECKPOINT = Path(r"E:\PythonD\ERA\sam2_weights\sam2.1_hiera_base_plus.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensemble_voting(masks, scores, threshold=0.5):
    if len(masks) == 0: return None
    weights = scores[:, np.newaxis, np.newaxis]
    prob_map = np.sum(masks.astype(np.float32) * weights, axis=0) / (np.sum(weights) + 1e-8)
    return prob_map > threshold



def clamp_box(box, width, height):
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    x1, x2 = sorted((float(np.clip(x1, 0, width - 1)), float(np.clip(x2, 0, width - 1))))
    y1, y2 = sorted((float(np.clip(y1, 0, height - 1)), float(np.clip(y2, 0, height - 1))))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def box_area_ratio(box, width, height):
    x1, y1, x2, y2 = clamp_box(box, width, height)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, width * height)


def derive_lesion_prompt(image, original_box=None):
    """Generate a conservative lesion seed/tight box for SAM2 from colour contrast."""
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    scale = min(1.0, 768.0 / max(height, width))
    small = cv2.resize(rgb, (max(1, int(width * scale)), max(1, int(height * scale))),
                       interpolation=cv2.INTER_AREA)
    small_h, small_w = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    blur_size = min(51, max(5, int(round(min(small_h, small_w) * 0.025)) | 1))
    smooth = cv2.GaussianBlur(lab, (blur_size, blur_size), 0)

    margin_y, margin_x = max(2, small_h // 12), max(2, small_w // 12)
    border = np.zeros((small_h, small_w), dtype=bool)
    border[:margin_y] = border[-margin_y:] = True
    border[:, :margin_x] = border[:, -margin_x:] = True
    lightness = smooth[..., 0]
    valid_border = border & (lightness > np.percentile(lightness, 20))
    skin_pixels = smooth[valid_border]
    if len(skin_pixels) < 100:
        skin_pixels = smooth.reshape(-1, 3)
    skin_lab = np.median(skin_pixels, axis=0)

    colour_distance = np.linalg.norm(smooth - skin_lab, axis=2)
    darkness = np.maximum(skin_lab[0] - smooth[..., 0], 0)

    def robust_normalize(values):
        low, high = np.percentile(values, (25, 95))
        return np.clip((values - low) / (high - low + 1e-6), 0, 1)

    saliency = 0.65 * robust_normalize(colour_distance) + 0.35 * robust_normalize(darkness)
    yy, xx = np.mgrid[:small_h, :small_w]
    centre_prior = np.exp(-0.5 * (((xx - small_w / 2) / (0.48 * small_w)) ** 2
                                  + ((yy - small_h / 2) / (0.48 * small_h)) ** 2))
    saliency *= 0.70 + 0.30 * centre_prior

    if original_box is not None:
        box = clamp_box(original_box, width, height) * scale
        x1, y1, x2, y2 = box.astype(int)
        box_region = np.zeros_like(saliency, dtype=bool)
        box_region[max(0, y1):min(small_h, y2 + 1), max(0, x1):min(small_w, x2 + 1)] = True
        saliency *= np.where(box_region, 1.0, 0.35)

    saliency_u8 = np.clip(saliency * 255, 0, 255).astype(np.uint8)
    otsu_threshold, _ = cv2.threshold(saliency_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(float(otsu_threshold), float(np.percentile(saliency_u8, 72)))
    proposal = (saliency_u8 >= threshold).astype(np.uint8)
    open_size = max(3, int(round(min(small_h, small_w) * 0.008)) | 1)
    close_size = max(5, int(round(min(small_h, small_w) * 0.025)) | 1)
    proposal = cv2.morphologyEx(proposal, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)))
    proposal = cv2.morphologyEx(proposal, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(proposal, connectivity=8)
    best_label, best_score = None, -np.inf
    target = np.array([small_w / 2, small_h / 2], dtype=np.float32)
    if original_box is not None:
        original = clamp_box(original_box, width, height) * scale
        target = np.array([(original[0] + original[2]) / 2, (original[1] + original[3]) / 2])

    for label in range(1, count):
        x, y, component_w, component_h, area = stats[label]
        area_ratio = area / max(1, small_w * small_h)
        if not MIN_MASK_AREA_RATIO <= area_ratio <= MAX_MASK_AREA_RATIO:
            continue
        component = labels == label
        border_fraction = np.concatenate((component[0], component[-1], component[:, 0], component[:, -1])).mean()
        distance = np.linalg.norm((centroids[label] - target) / np.array([small_w, small_h]))
        mean_saliency = float(saliency[component].mean())
        compactness = area / max(1, component_w * component_h)
        score = (2.0 * mean_saliency + 0.35 * np.log1p(area) + 0.5 * compactness
                 - 2.5 * distance - 5.0 * border_fraction)
        if score > best_score:
            best_label, best_score = label, score

    if best_label is None:
        return None, None, None

    component = (labels == best_label).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(component, contours, -1, 1, thickness=cv2.FILLED)
    distance_map = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    seed_y, seed_x = np.unravel_index(np.argmax(distance_map), distance_map.shape)
    x, y, component_w, component_h, _ = stats[best_label]
    padding = 0.18
    adaptive_box = np.array([
        max(0, x - int(component_w * padding)),
        max(0, y - int(component_h * padding)),
        min(small_w - 1, x + component_w + int(component_w * padding)),
        min(small_h - 1, y + component_h + int(component_h * padding)),
    ], dtype=np.float32) / scale
    adaptive_point = np.array([seed_x, seed_y], dtype=np.float32) / scale
    fallback_mask = cv2.resize(component, (width, height), interpolation=cv2.INTER_NEAREST) > 0
    return clamp_box(adaptive_box, width, height), adaptive_point, fallback_mask


def centred_prompt_boxes(point, reference_box, width, height,
                           scales=(0.18, 0.25, 0.35, 0.50, 0.70, 1.0)):
    """Create boxes of several sizes around a positive point."""
    reference_box = clamp_box(reference_box, width, height)
    point_x, point_y = np.asarray(point, dtype=np.float32)
    box_width = max(8.0, reference_box[2] - reference_box[0])
    box_height = max(8.0, reference_box[3] - reference_box[1])
    boxes = []
    for scale in scales:
        boxes.append(clamp_box([
            point_x - box_width * scale / 2,
            point_y - box_height * scale / 2,
            point_x + box_width * scale / 2,
            point_y + box_height * scale / 2,
        ], width, height))
    return boxes


def expand_box(box, width, height, scale=1.30):
    box = clamp_box(box, width, height)
    centre_x, centre_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    box_width, box_height = box[2] - box[0], box[3] - box[1]
    return clamp_box([
        centre_x - box_width * scale / 2,
        centre_y - box_height * scale / 2,
        centre_x + box_width * scale / 2,
        centre_y + box_height * scale / 2,
    ], width, height)


def _robust_unit_scale(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    low, high = np.percentile(values, (10, 90))
    if high - low < 1e-6:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0, 1)


def choose_best_candidate(masks, scores, boxes, points, image, prior_mask=None, reference_point=None):
    """Rank SAM2 candidates using confidence, appearance, shape and prompt fit."""
    if len(masks) == 0:
        return None

    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    scale = min(1.0, 512.0 / max(height, width))
    small_width, small_height = max(1, int(width * scale)), max(1, int(height * scale))
    small_rgb = cv2.resize(rgb, (small_width, small_height), interpolation=cv2.INTER_AREA)
    small_lab = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    small_gray = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
    gradient = cv2.magnitude(
        cv2.Sobel(small_gray, cv2.CV_32F, 1, 0),
        cv2.Sobel(small_gray, cv2.CV_32F, 0, 1)
    )
    ring_size = max(3, int(round(min(small_height, small_width) * 0.02)) | 1)
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_size, ring_size))

    features = []
    for mask, box, point in zip(masks, boxes, points):
        mask = np.asarray(mask) > 0
        area_ratio = float(mask.mean())
        small_mask = cv2.resize(mask.astype(np.uint8), (small_width, small_height),
                                interpolation=cv2.INTER_NEAREST) > 0
        small_u8 = small_mask.astype(np.uint8)
        dilated = cv2.dilate(small_u8, ring_kernel) > 0
        annulus = dilated & ~small_mask
        if small_mask.any() and annulus.any():
            colour_contrast = float(np.linalg.norm(
                small_lab[small_mask].mean(axis=0) - small_lab[annulus].mean(axis=0)
            ))
        else:
            colour_contrast = 0.0
        boundary = cv2.morphologyEx(
            small_u8, cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        ) > 0
        edge_alignment = float(gradient[boundary].mean()) if boundary.any() else 0.0

        x1, y1, x2, y2 = clamp_box(box, width, height).astype(int)
        box_fill = float(mask[y1:y2 + 1, x1:x2 + 1].mean())
        border_ratio = float(np.concatenate(
            (mask[0], mask[-1], mask[:, 0], mask[:, -1])
        ).mean())
        point_x, point_y = np.clip(
            np.asarray(point, dtype=int), [0, 0], [width - 1, height - 1]
        )
        contains_point = bool(mask[point_y, point_x])

        ys, xs = np.where(small_mask)
        if len(xs):
            bounding_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
            compactness = float(small_mask.sum() / max(1, bounding_area))
        else:
            compactness = 0.0
        features.append((area_ratio, box_fill, border_ratio, contains_point,
                         colour_contrast, edge_alignment, compactness))

    contrast_scores = _robust_unit_scale([f[4] for f in features])
    edge_scores = _robust_unit_scale([f[5] for f in features])
    qualities = []
    prior_is_usable = prior_mask is not None and 0.001 < np.mean(prior_mask) < 0.50
    for index, (mask, sam_score) in enumerate(zip(masks, scores)):
        area_ratio, box_fill, _, contains_point, _, _, compactness = features[index]
        # Full-set combination ablation selected no_seed_distance + no_border_penalty.
        quality = (
            3.0 * float(sam_score)
            + 1.00 * float(contrast_scores[index])
            + 0.05 * float(edge_scores[index])
            + 0.10 * compactness
            + (0.50 if contains_point else -3.0)
            - 5.0 * max(0.0, box_fill - 0.90)
            - 8.0 * max(0.0, area_ratio - 0.90)
            - 0.60 * box_area_ratio(boxes[index], width, height)
        )
        if area_ratio < MIN_MASK_AREA_RATIO:
            quality -= 3.0
        if prior_is_usable:
            candidate = np.asarray(mask) > 0
            intersection = np.logical_and(candidate, prior_mask).sum()
            union = np.logical_or(candidate, prior_mask).sum()
            quality += 0.15 * intersection / max(1, union)
        qualities.append(quality)
    return int(np.argmax(qualities))


def calculate_metrics_from_totals(tp, fp, fn, tn):
    eps = 1e-8
    return {
        "dice": (2. * tp) / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "sensitivity": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps)
    }


def evaluate_folder_macro(masks_dir: Path, gt_dir: Path):
    tp, fp, fn, tn = 0.0, 0.0, 0.0, 0.0
    for mask_file in tqdm([f for f in masks_dir.iterdir() if f.suffix in ('.jpg', '.png')], desc='Evaluating'):
        clean_id = mask_file.stem.replace("_segmentation", "")
        gt_file = next(
            (gt_dir / n for n in [f"{clean_id}.png", f"{clean_id}_segmentation.png"] if (gt_dir / n).exists()), None)
        if not gt_file: continue

        pm = np.array(Image.open(mask_file).convert('L')) > 127
        gm = np.array(Image.open(gt_file).convert('L')) > 127
        if pm.shape != gm.shape: pm = cv2.resize(pm.astype(np.uint8), (gm.shape[1], gm.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST) > 0

        tp += np.logical_and(pm, gm).sum()
        fp += np.logical_and(pm, ~gm).sum()
        fn += np.logical_and(~pm, gm).sum()
        tn += np.logical_and(~pm, ~gm).sum()
    return calculate_metrics_from_totals(tp, fp, fn, tn)


def load_prompts_2wd(prompt_file: Path):
    # 鏋佷綆姒傜巼鐨勪繚搴曡В鏋愶細濡傛灉鎵句笉鍒?prompt锛屼娇鐢ㄥ叏鍥句腑蹇冮檮杩戠殑铏氭嫙妗嗗苟杩涘叆 Defer銆?
    default_defer = (np.array([10, 10, 200, 200]), np.array([100, 100]), "Defer", None)

    if not prompt_file.exists(): return default_defer
    with open(prompt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if not lines: return default_defer

    decision = "Defer" if (len(lines) > 1 and "uncertain" in lines[1].lower()) else "Accept"
    parts = lines[0].strip().split()
    try:
        coords = [float(p) for p in parts[:6]]
        prior_mask = parts[6] if len(parts) > 6 else None
        return np.array(coords[:4]), np.array([coords[4], coords[5]]), decision, prior_mask
    except:
        return default_defer


def process_prior_mask(mask_path, device, query_box, image_size):
    """Build a geometry-normalized, low-magnitude prior for SAM2.

    The transferred mask is cropped to its own lesion box, affinely mapped into the
    current query box, and converted to local soft logits. It therefore cannot impose
    a global +/-10 background decision on an unrelated image.
    """
    return prepare_aligned_prior(mask_path, query_box, image_size, device)


def jitter_bbox(bbox, img_w, img_h, scales=(0.90, 1.0, 1.10)):
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cl = lambda b: [max(0, int(b[0])), max(0, int(b[1])), min(img_w - 1, int(b[2])), min(img_h - 1, int(b[3]))]
    return np.array([cl([cx - w * s / 2, cy - h * s / 2, cx + w * s / 2, cy + h * s / 2]) for s in scales])


def postprocess_medical_mask(mask, W, H, box_prompt=None, point_prompt=None):
    if mask is None or not np.any(mask):
        return np.zeros((H, W), dtype=bool)
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if np.sum(mask_u8) / (W * H) > MAX_MASK_AREA_RATIO:
        return np.zeros((H, W), dtype=bool)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels > 1:
        selected_label = None
        if point_prompt is not None:
            point_x, point_y = np.clip(np.asarray(point_prompt, dtype=int), [0, 0], [W - 1, H - 1])
            if labels[point_y, point_x] > 0:
                selected_label = labels[point_y, point_x]
        if selected_label is None:
            selected_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_u8 = (labels == selected_label).astype(np.uint8)

    kernel_size = max(5, min(21, int(round(min(W, H) * 0.008)) | 1))
    return cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (kernel_size, kernel_size))) > 0


def main():
    mode_str = "鏈夎瀺鍚?(Ensemble)" if USE_ENSEMBLE else "鏃犺瀺鍚?(Single Box)"
    print(f"鍚姩缁堟瀬闆嗘垚浠诲姟锛堝弻鏀矾鐢?- {mode_str}锛夛細{DATASET_NAME}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sam2_predictor = SAM2ImagePredictor(build_sam2(str(SAM2_MODEL_CFG), str(SAM2_CHECKPOINT)).to(DEVICE))
    image_files = sorted([f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    stats = {"Accept": 0, "Defer": 0}

    for image_path in tqdm(image_files, desc=f"鎵ц缁堟瀬鍒嗗壊 ({mode_str})"):
        try:
            image = Image.open(image_path).convert("RGB")
            W, H = image.size
            sam2_predictor.set_image(image)
            clean_id = image_path.stem.replace("_segmentation", "")

            box_prompt, point_prompt, decision, prior_mask_path = load_prompts_2wd(
                PROMPT_DIR / f"{clean_id}_segmentation.txt")

            best_mask = np.zeros((H, W), dtype=bool)
            original_box = clamp_box(box_prompt, W, H)
            original_point = np.clip(np.asarray(point_prompt, dtype=np.float32),
                                     [0, 0], [W - 1, H - 1])
            adaptive_box, adaptive_point, cv_prior = derive_lesion_prompt(image, original_box)

            seed_points = [original_point]
            if adaptive_point is not None and cv_prior is not None:
                point_x, point_y = original_point.astype(int)
                point_in_prior = bool(cv_prior[point_y, point_x])
                seed_distance = np.linalg.norm(
                    (adaptive_point - original_point) / np.array([W, H], dtype=np.float32)
                )
                if not point_in_prior and seed_distance > 0.02:
                    seed_points.append(adaptive_point)

            all_masks, all_scores, all_boxes, all_points, all_logits = [], [], [], [], []

            def add_candidates(box, point, mask_input=None):
                masks, scores, logits = sam2_predictor.predict(
                    point_coords=np.array([point]), point_labels=np.array([1]),
                    box=box, mask_input=mask_input, multimask_output=True
                )
                all_masks.extend(masks)
                all_scores.extend(scores)
                all_boxes.extend([np.asarray(box, dtype=np.float32)] * len(masks))
                all_points.extend([np.asarray(point, dtype=np.float32)] * len(masks))
                all_logits.extend(logits)

            aligned_prior_mask = None
            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                for seed_point in seed_points:
                    for candidate_box in centred_prompt_boxes(seed_point, original_box, W, H):
                        add_candidates(candidate_box, seed_point)

                # Keep the previous adaptive-box route as a competing hypothesis.
                if adaptive_box is not None and adaptive_point is not None:
                    for candidate_box in jitter_bbox(adaptive_box, W, H):
                        add_candidates(candidate_box, adaptive_point)

                if decision == "Accept":
                    # The prompt generator validated the retrieved prior against the
                    # attention-derived original_box, so use that same coordinate
                    # frame for transfer rather than a different adaptive proposal.
                    prior_box = original_box
                    prior_point = original_point
                    mask_input, aligned_prior_mask = process_prior_mask(
                        prior_mask_path, DEVICE, prior_box, (W, H)
                    )
                    if mask_input is not None:
                        add_candidates(prior_box, prior_point, mask_input=mask_input)

                # When an accepted retrieval prior is available, use its aligned
                # image-space mask for consistency scoring; otherwise retain the
                # image-derived CV prior as a fallback cue.
                ranking_prior = aligned_prior_mask if aligned_prior_mask is not None else cv_prior
                best_index = choose_best_candidate(
                    all_masks, all_scores, all_boxes, all_points, image, ranking_prior, original_point
                )
                if best_index is not None:
                    refine_box = expand_box(all_boxes[best_index], W, H, scale=1.30)
                    refine_logit = np.asarray(all_logits[best_index], dtype=np.float32)[None, ...]
                    add_candidates(refine_box, all_points[best_index], mask_input=refine_logit)
                    best_index = choose_best_candidate(
                        all_masks, all_scores, all_boxes, all_points, image, ranking_prior, original_point
                    )
                    best_mask = np.asarray(all_masks[best_index]) > 0
                    selected_box = all_boxes[best_index]
                    selected_point = all_points[best_index]
                else:
                    best_mask = cv_prior if cv_prior is not None else best_mask
                    selected_box = adaptive_box if adaptive_box is not None else original_box
                    selected_point = adaptive_point if adaptive_point is not None else original_point

            stats[decision] += 1
            best_mask = postprocess_medical_mask(best_mask, W, H, box_prompt=selected_box, point_prompt=selected_point)
            Image.fromarray((best_mask.squeeze() > 0).astype(np.uint8) * 255).save(OUTPUT_DIR / f"{clean_id}.jpg",
                                                                                   "JPEG", quality=100, subsampling=0)
        except Exception as e:
            tqdm.write(f"澶辫触 {image_path.name}: {e}")

    print(f"\n缁熻: {stats}")
    metrics = evaluate_folder_macro(OUTPUT_DIR, GT_DIR)
    print(f"\n--- 鍙屾敮璺敱璇勬祴鎶ュ憡 ({mode_str}) ---")
    print(f"Dice: {metrics.get('dice', 0):.4f}")
    print(f"IoU: {metrics.get('iou', 0):.4f}")
    print(f"SE: {metrics.get('sensitivity', 0):.4f}")
    print(f"SP: {metrics.get('specificity', 0):.4f}")
    print(f"AC: {metrics.get('accuracy', 0):.4f}")


if __name__ == "__main__":
    main()

