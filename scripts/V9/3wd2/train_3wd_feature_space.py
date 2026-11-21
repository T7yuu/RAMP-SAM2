import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class FeatureAdapter(nn.Module):
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


class TripletUniformityLoss(nn.Module):
    def __init__(self, margin=0.5, uniformity_weight=2.0):
        super().__init__()
        self.margin = margin
        self.uniformity_weight = uniformity_weight

    def forward(self, anchor, positive, negative):
        # 1. 经典三元组损失 (Triplet Loss)：保证同类靠近，异类推开
        dist_pos = torch.norm(anchor - positive, p=2, dim=1)
        dist_neg = torch.norm(anchor - negative, p=2, dim=1)
        loss_triplet = F.relu(dist_pos - dist_neg + self.margin).mean()

        # 2. 反坍缩大招 (Uniformity Loss)：强制所有特征在超球面上排布均匀
        sq_dist = torch.cdist(anchor, anchor, p=2).pow(2)
        mask = ~torch.eye(anchor.size(0), dtype=torch.bool, device=anchor.device)
        sq_dist = sq_dist[mask]
        loss_uniformity = torch.log(torch.mean(torch.exp(-2.0 * sq_dist)))

        total_loss = loss_triplet + self.uniformity_weight * loss_uniformity
        return total_loss, loss_triplet, loss_uniformity


class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.num_samples = len(features)

        similarity_matrix = torch.matmul(self.features, self.features.T)

        self.pos_idx, self.neg_idx = [], []
        for i in tqdm(range(self.num_samples), desc="构建对比元组"):
            sims = similarity_matrix[i]
            sorted_idx = torch.argsort(sims, descending=True)
            self.pos_idx.append(sorted_idx[1:2].item())  # 最相似的作正样本
            self.neg_idx.append(sorted_idx[-1].item())  # 最不相似的作负样本

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        return (self.features[idx], self.features[self.pos_idx[idx]], self.features[self.neg_idx[idx]])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    NPZ_PATH = r"E:/PythonD/ERA3/scripts/V9/scripts/outputs_medical/medical_features.npz"
    SAVE_PATH = r"E:/PythonD/ERA3/scripts/V9/scripts/2wd_adapter.pth"  # 改名 2wd_adapter

    data = np.load(NPZ_PATH)
    dataset = FeatureDataset(data['features'])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    adapter = FeatureAdapter().to(device)
    criterion = TripletUniformityLoss(uniformity_weight=2.0).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    adapter.train()
    print(f"\n🚀 开始训练 (Two-Way 对比特征适配器)...")

    for epoch in range(50):
        total_epoch_loss, total_trip_loss, total_uni_loss = 0, 0, 0
        for anc, pos, neg in dataloader:
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
            loss, loss_trip, loss_uni = criterion(adapter(anc), adapter(pos), adapter(neg))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            total_trip_loss += loss_trip.item()
            total_uni_loss += loss_uni.item()

        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch + 1}/50] ➜ Total: {total_epoch_loss / len(dataloader):.4f} | Triplet: {total_trip_loss / len(dataloader):.4f} | Uniformity: {total_uni_loss / len(dataloader):.4f}")

    torch.save(adapter.state_dict(), SAVE_PATH)
    print(f"\n✅ 二支适配器已保存至: {SAVE_PATH}")


if __name__ == "__main__":
    main()