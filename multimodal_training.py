# multimodal_training.py
# Run: python multimodal_training.py --mode debug --epochs 2 --batch_size 8

import os
import argparse
import random
import numpy as np
import math
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix
)

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except:
        return float('nan')


def safe_auprc(y_true, y_score):
    try:
        return average_precision_score(y_true, y_score)
    except:
        return float('nan')


def specificity_score(y_true, y_pred):
    # handle degenerate cases
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        return tn / (tn + fp + 1e-12)
    except:
        return float('nan')


# ---------------------------
# Synthetic Dataset for Debug Testing
# ---------------------------
class SyntheticMultimodalDataset(Dataset):
    """Tiny synthetic dataset for debugging/trial runs."""
    def __init__(self, n=200, num_labels=5, image_size=256, num_ehr_features=16):
        self.n = n
        self.num_labels = num_labels
        self.num_ehr_features = num_ehr_features

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        rng = np.random.RandomState(0)

        # Random images
        self.images = [
            Image.fromarray((rng.rand(image_size, image_size, 3) * 255).astype("uint8"))
            for _ in range(n)
        ]

        # Random EHR features
        self.ehr = rng.normal(size=(n, num_ehr_features)).astype(np.float32)

        # Random multi-label targets with imbalance
        p = np.linspace(0.2, 0.6, num_labels)
        self.targets = (rng.rand(n, num_labels) < p).astype(np.float32)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = self.transform(self.images[idx])
        ehr = torch.tensor(self.ehr[idx])
        target = torch.tensor(self.targets[idx])
        return img, ehr, target


# ---------------------------
# Image Encoder (DenseNet-121)
# ---------------------------
class ImageEncoderDenseNet(nn.Module):
    def __init__(self, pretrained=True, out_dim=128):
        super().__init__()
        # Use torchvision new API safely
        try:
            backbone = models.densenet121(weights="IMAGENET1K_V1" if pretrained else None)
        except Exception:
            backbone = models.densenet121(pretrained=pretrained)
        self.features = backbone.features
        self.conv_proj = nn.Conv2d(1024, out_dim, kernel_size=1)

    def forward(self, x):
        f = self.features(x)                             # [B,1024,H,W]
        f_proj = self.conv_proj(f)                       # [B,out_dim,H,W]
        B, C, H, W = f_proj.shape
        tokens = f_proj.flatten(2).permute(0,2,1)        # [B,H*W,C]
        pooled = F.adaptive_avg_pool2d(f_proj, (1,1)).view(B, C)
        return tokens, pooled


# ---------------------------
# EHR Transformer Encoder
# ---------------------------
class EHRTransformerEncoder(nn.Module):
    def __init__(self, num_features, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        # create a small fixed tokenization
        self.num_tokens = 4
        # ensure token_dim >=1
        self.token_dim = max(1, num_features // self.num_tokens)
        self.input_dim = self.num_tokens * self.token_dim

        self.linear = nn.Linear(self.token_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        B = x.size(0)
        # pad or trim to input_dim
        if x.size(1) < self.input_dim:
            pad = torch.zeros(B, self.input_dim - x.size(1), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        else:
            x = x[:, :self.input_dim]
        x = x.view(B, self.num_tokens, self.token_dim)
        tokens = self.linear(x)               # [B,T,d]
        out = self.transformer(tokens)
        pooled = out.mean(dim=1)
        return out, pooled


# ---------------------------
# Cross-Attention Fusion Block
# ---------------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim_img=128, dim_ehr=128, dim_hidden=256, num_heads=4):
        super().__init__()
        # image tokens query EHR, and EHR tokens query image
        self.attn_img2ehr = nn.MultiheadAttention(embed_dim=dim_ehr, num_heads=num_heads, batch_first=True)
        self.attn_ehr2img = nn.MultiheadAttention(embed_dim=dim_img, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(dim_img + dim_ehr, dim_hidden)

    def forward(self, img_tokens, ehr_tokens):
        # img_tokens: [B, Nimg, Cimg]
        # ehr_tokens: [B, Nehr, Cehr]
        # attend: query, key, value
        try:
            img_to_ehr, _ = self.attn_img2ehr(img_tokens, ehr_tokens, ehr_tokens)
            ehr_to_img, _ = self.attn_ehr2img(ehr_tokens, img_tokens, img_tokens)
        except Exception:
            # fallback: simple avg pooling if shapes mismatch
            img_pool = img_tokens.mean(dim=1)
            ehr_pool = ehr_tokens.mean(dim=1)
            fused = torch.cat([img_pool, ehr_pool], dim=1)
            return F.relu(self.fc(fused))

        img_pool = img_to_ehr.mean(dim=1)
        ehr_pool = ehr_to_img.mean(dim=1)

        fused = torch.cat([img_pool, ehr_pool], dim=1)
        fused = F.relu(self.fc(fused))
        return fused


# ---------------------------
# Final Classifier
# ---------------------------
class MultimodalClassifier(nn.Module):
    def __init__(self, in_dim=256, num_labels=5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels)
        )

    def forward(self, fused_vector):
        return self.classifier(fused_vector)


# ---------------------------
# Metrics
# ---------------------------
def compute_metrics(y_true, y_scores):
    y_pred = (y_scores >= 0.5).astype(int)
    C = y_true.shape[1]

    results = {"AUROC": [], "AUPRC": [], "Precision": [], "Recall": [], "F1": [], "Specificity": [], "Accuracy": []}

    for c in range(C):
        yt = y_true[:, c]
        ys = y_scores[:, c]

        auc = safe_auc(yt, ys)
        auprc = safe_auprc(yt, ys)

        p, r, f, _ = precision_recall_fscore_support(yt, y_pred[:,c], average="binary", zero_division=0)
        acc = accuracy_score(yt, y_pred[:,c])
        spec = specificity_score(yt, y_pred[:,c]) if len(np.unique(yt)) > 1 else float('nan')

        results["AUROC"].append(auc)
        results["AUPRC"].append(auprc)
        results["Precision"].append(p)
        results["Recall"].append(r)
        results["F1"].append(f)
        results["Specificity"].append(spec)
        results["Accuracy"].append(acc)

    macro = {m: float(np.nanmean(results[m])) for m in results}

    return results, macro


# ---------------------------
# Training one epoch
# ---------------------------
def train_epoch(img_enc, ehr_enc, fusion, clf, loader, optimizer, device):
    img_enc.train()
    ehr_enc.train()
    fusion.train()
    clf.train()

    criterion = nn.BCEWithLogitsLoss()

    for img, ehr, targets in loader:
        img, ehr, targets = img.to(device), ehr.to(device), targets.to(device)

        img_tokens, _ = img_enc(img)
        ehr_tokens, _ = ehr_enc(ehr)

        fused = fusion(img_tokens, ehr_tokens)
        logits = clf(fused)

        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ---------------------------
# Evaluation
# ---------------------------
def evaluate(img_enc, ehr_enc, fusion, clf, loader, device):
    img_enc.eval()
    ehr_enc.eval()
    fusion.eval()
    clf.eval()

    all_scores = []
    all_targets = []

    with torch.no_grad():
        for img, ehr, targets in loader:
            img, ehr = img.to(device), ehr.to(device)

            img_tokens, _ = img_enc(img)
            ehr_tokens, _ = ehr_enc(ehr)

            logits = clf(fusion(img_tokens, ehr_tokens))
            probs = torch.sigmoid(logits).cpu().numpy()

            all_scores.append(probs)
            all_targets.append(targets.numpy())

    all_scores = np.vstack(all_scores)
    all_targets = np.vstack(all_targets)

    return compute_metrics(all_targets, all_scores)


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="debug")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # synthetic data for testing
    train_ds = SyntheticMultimodalDataset(n=200)
    val_ds   = SyntheticMultimodalDataset(n=80)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)

    img_enc = ImageEncoderDenseNet().to(device)
    ehr_enc = EHRTransformerEncoder(num_features=16).to(device)
    fusion  = CrossAttentionFusion(dim_img=128, dim_ehr=128, dim_hidden=256, num_heads=4).to(device)
    clf     = MultimodalClassifier(in_dim=256, num_labels=5).to(device)

    optimizer = torch.optim.Adam(
        list(img_enc.parameters()) +
        list(ehr_enc.parameters()) +
        list(fusion.parameters()) +
        list(clf.parameters()),
        lr=1e-4
    )

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_epoch(img_enc, ehr_enc, fusion, clf, train_loader, optimizer, device)
        (per_class, macro) = evaluate(img_enc, ehr_enc, fusion, clf, val_loader, device)
        print("Macro metrics:", macro)

    print("\nTraining completed.")


if __name__ == "__main__":
    main()
