"""
适配器检索质量评估 (feature-space, 不需要 SAM2)
==============================================

对给定的一组适配器权重，在源域特征库上计算:

  1. 三元组几何: mean d+, mean d-, active 占比, triplet loss
  2. 均匀性: Eq.(7) uniformity loss
  3. 检索结构保真度: 投影空间中的最近邻与原始 512-D 空间中的最近邻一致的比例
     (adapter 若把检索结构破坏了, 这个数字会掉)
  4. 检索锐度: top-1 与 top-2 相似度之差 (越大越锐)

用法
----
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        D:/Python311/python.exe scripts/V9/3wd2/evaluate_adapter_retrieval.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NPZ = ROOT / "scripts" / "V9" / "scripts" / "outputs_medical" / "medical_features.npz"
DEFAULT_ABL = ROOT / "scripts" / "V9" / "negative_strategy_ablation"

MARGIN = 0.5


class FeatureAdapter(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.projector(x), p=2, dim=-1)


def global_partners(raw: torch.Tensor):
    """在原始 512-D 空间预计算确定性正/负伙伴 (论文设计)。返回 (pos_idx, neg_idx)。

    对角线屏蔽方向必须正确: 正样本取 argmax 需设 -inf; 负样本取 argmin 需设 +inf。
    两侧都设 -inf 会让负样本退化成 anchor 自身。
    """
    sim = torch.matmul(raw, raw.T)
    sim_pos = sim.clone()
    sim_pos.fill_diagonal_(-float("inf"))
    pos = torch.argmax(sim_pos, dim=1)
    sim_neg = sim.clone()
    sim_neg.fill_diagonal_(float("inf"))
    neg = torch.argmin(sim_neg, dim=1)
    return pos, neg


def evaluate(z: torch.Tensor, raw: torch.Tensor, pos_idx, neg_idx):
    d_pos = torch.norm(z - z[pos_idx], p=2, dim=1)
    d_neg = torch.norm(z - z[neg_idx], p=2, dim=1)
    trip_vec = F.relu(d_pos - d_neg + MARGIN)

    sq = torch.cdist(z, z, p=2).pow(2)
    mask = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    uni = torch.log(torch.mean(torch.exp(-2.0 * sq[mask])))

    # 投影空间的最近邻 vs 原始空间的最近邻
    def nn_index(feat):
        s = torch.matmul(feat, feat.T)
        s.fill_diagonal_(-float("inf"))
        return s

    s_proj = nn_index(z)
    s_raw = nn_index(raw)
    nn_proj = torch.argmax(s_proj, dim=1)
    nn_raw = torch.argmax(s_raw, dim=1)

    top2 = torch.topk(s_proj, k=2, dim=1).values
    sharp = (top2[:, 0] - top2[:, 1])

    # 原始空间最近邻在投影空间的排名 (1 = 完美保留)
    order_proj = torch.argsort(s_proj, descending=True, dim=1)
    rank = (order_proj == nn_raw.unsqueeze(1)).nonzero()[:, 1].float() + 1

    return {
        "mean_d_pos": float(d_pos.mean()),
        "mean_d_neg": float(d_neg.mean()),
        "triplet_loss": float(trip_vec.mean()),
        "active_frac": float((trip_vec > 0).float().mean()),
        "uniformity_loss": float(uni),
        "nn_agreement_raw": float((nn_proj == nn_raw).float().mean()),
        "mean_rank_of_raw_nn": float(rank.mean()),
        "top1_minus_top2": float(sharp.mean()),
        "mean_top1_sim": float(top2[:, 0].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABL)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--extra-weights", nargs="*", default=[
        str(ROOT / "scripts" / "V9" / "scripts" / "2wd_adapter.pth"),
        str(ROOT / "scripts" / "V9" / "feature_space_ablation_results" / "weights" / "adapter_uniformity.pth"),
    ])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = torch.tensor(np.load(args.npz)["features"], dtype=torch.float32)
    raw_n = F.normalize(raw, p=2, dim=-1).to(device)
    raw_dev = raw.to(device)
    pos_idx, neg_idx = global_partners(raw_n.cpu())
    pos_idx, neg_idx = pos_idx.to(device), neg_idx.to(device)

    print(f"源域: {raw.shape}  device={device}\n")

    rows = []

    # 基线 0: 原始 512-D 特征 (identity)
    rows.append({"name": "raw_512d_identity",
                 **evaluate(raw_n, raw_n, pos_idx, neg_idx)})

    # 基线 1: 未训练的随机初始化适配器
    torch.manual_seed(42)
    untrained = FeatureAdapter().to(device).eval()
    with torch.no_grad():
        rows.append({"name": "untrained_random_adapter",
                     **evaluate(untrained(raw_dev), raw_n, pos_idx, neg_idx)})

    # 消融训练出的四个适配器
    cands = []
    for s in ["farthest", "hardest", "semi_hard", "dist_weighted"]:
        p = args.ablation_dir / "weights" / f"adapter_{s}.pth"
        if p.exists():
            cands.append((f"adapter_{s}", p))
    for p in args.extra_weights:
        p = Path(p)
        if p.exists():
            cands.append((f"existing::{p.stem}", p))

    for name, path in cands:
        m = FeatureAdapter().to(device).eval()
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        with torch.no_grad():
            z = m(raw_dev)
        rows.append({"name": name, "weights": str(path),
                     **evaluate(z, raw_n, pos_idx, neg_idx)})
        print(f"  [ok] {name}")

    out = args.output or (args.ablation_dir / "retrieval_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"npz": str(args.npz), "n_source": int(raw.shape[0]), "results": rows}, f, indent=2)

    hdr = (f"{'variant':<34}{'triplet':>9}{'active':>8}{'d+':>7}{'d-':>7}"
           f"{'uni':>9}{'NNagree':>9}{'rank':>7}{'sharp':>8}")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<34}{r['triplet_loss']:>9.4f}{r['active_frac']*100:>7.1f}%"
              f"{r['mean_d_pos']:>7.3f}{r['mean_d_neg']:>7.3f}{r['uniformity_loss']:>9.3f}"
              f"{r['nn_agreement_raw']*100:>8.1f}%{r['mean_rank_of_raw_nn']:>7.2f}"
              f"{r['top1_minus_top2']:>8.4f}")
    print("=" * len(hdr))
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
