import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ================= 1. 三支特征适配器网络 =================
class ThreeWayFeatureAdapter(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        # 降维并提纯特征，剥离无用的背景噪声
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        out = self.projector(x)
        return F.normalize(out, p=2, dim=-1)  # 将特征投影到超球面上


# ================= 2. 三支决策对比损失函数 =================
class ThreeWayDecisionLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.8, margin=0.1):
        super().__init__()
        self.alpha = alpha  # Accept 域半径: 距离必须小于此值
        self.beta = beta  # Reject 域半径: 距离必须大于此值
        self.margin = margin

    def forward(self, anchor, positive, negative, boundary):
        dist_pos = torch.norm(anchor - positive, p=2, dim=1)
        dist_neg = torch.norm(anchor - negative, p=2, dim=1)
        dist_bnd = torch.norm(anchor - boundary, p=2, dim=1)

        # Accept 损失：强行拉近
        loss_pos = F.relu(dist_pos - self.alpha + self.margin)
        # Reject 损失：强行推开
        loss_neg = F.relu(self.beta - dist_neg + self.margin)
        # Defer 损失 (核心！)：强迫边界样本悬浮在 alpha 和 beta 之间
        loss_bnd_lower = F.relu(self.alpha - dist_bnd + self.margin)
        loss_bnd_upper = F.relu(dist_bnd - self.beta + self.margin)
        loss_bnd = loss_bnd_lower + loss_bnd_upper

        return loss_pos.mean() + loss_neg.mean() + loss_bnd.mean()


# ================= 3. 数据集构建 =================
class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.num_samples = len(features)

        # [极简自监督策略]：利用原始余弦距离挖掘样本
        # 实际打榜中，这里应该用真实的 Mask IoU 来严格定义
        print("正在自动挖掘 Positive, Negative, Boundary 样本...")
        similarity_matrix = torch.matmul(self.features, self.features.T)

        self.pos_idx = []
        self.neg_idx = []
        self.bnd_idx = []

        for i in tqdm(range(self.num_samples), desc="构建三支元组"):
            sims = similarity_matrix[i]
            # 最相似的是自己，取第二到第五相似的作为 Positive
            sorted_idx = torch.argsort(sims, descending=True)
            self.pos_idx.append(sorted_idx[1:2].item())

            # 最不相似的作为 Negative (Reject域)
            self.neg_idx.append(sorted_idx[-1].item())

            # 相似度居中的作为 Boundary (Defer域 - 模棱两可的图)
            mid_idx = self.num_samples // 2
            self.bnd_idx.append(sorted_idx[mid_idx].item())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.features[self.pos_idx[idx]],
            self.features[self.neg_idx[idx]],
            self.features[self.bnd_idx[idx]]
        )


# ================= 4. 主训练循环 =================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    NPZ_PATH = r"E:/PythonD/ERA3/scripts/V9/scripts/outputs_medical/medical_features.npz"
    SAVE_PATH = r"E:/PythonD/ERA3/scripts/V9/scripts/3wd_adapter.pth"

    print("加载原始 BiomedCLIP 特征...")
    data = np.load(NPZ_PATH)
    raw_features = data['features']

    dataset = FeatureDataset(raw_features)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    adapter = ThreeWayFeatureAdapter(input_dim=512, output_dim=128).to(device)
    criterion = ThreeWayDecisionLoss(alpha=0.3, beta=0.8).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    epochs = 50
    print(f"\n🚀 开始在 {device} 上训练三支特征适配器 (Epochs: {epochs})...")

    adapter.train()
    for epoch in range(epochs):
        total_loss = 0
        for anchor, pos, neg, bnd in dataloader:
            anchor, pos, neg, bnd = anchor.to(device), pos.to(device), neg.to(device), bnd.to(device)

            # 映射到全新的三支决策空间
            z_anchor = adapter(anchor)
            z_pos = adapter(pos)
            z_neg = adapter(neg)
            z_bnd = adapter(bnd)

            loss = criterion(z_anchor, z_pos, z_neg, z_bnd)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

    # 保存打通任督二脉的权重
    torch.save(adapter.state_dict(), SAVE_PATH)
    print(f"✅ 训练完成！三支空间映射器已保存至: {SAVE_PATH}")


if __name__ == "__main__":
    main()