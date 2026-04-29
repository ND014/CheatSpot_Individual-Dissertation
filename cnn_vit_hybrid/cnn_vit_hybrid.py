# ==================== CHEATSPOT - CNN + ViT Hybrid ====================

import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import ViTModel
from PIL import Image
from sklearn.metrics import classification_report
from google.colab import drive

drive.mount('/content/drive')

SAVE_PATH = '/content/drive/MyDrive/CheatSpot/cnn_vit_hybrid_best.pt'
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS    = 20


# ==================== Dataset ====================

class YOLOFrameDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir   = img_dir
        self.transform = transform
        self.samples   = []

        for label, cls in enumerate(['not_cheating', 'cheating']):
            cls_dir = os.path.join(img_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = YOLOFrameDataset('/content/drive/MyDrive/CheatSpot/train', transform)
val_dataset   = YOLOFrameDataset('/content/drive/MyDrive/CheatSpot/valid', transform)

train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=2)
val_loader    = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=2)


# ==================== Model ====================

class CNNViTHybrid(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        resnet          = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn_branch = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_proj   = nn.Linear(512, 256)

        self.vit_branch = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.vit_proj   = nn.Linear(768, 256)

        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn_branch(x).squeeze(-1).squeeze(-1)
        cnn_feat = self.cnn_proj(cnn_feat)

        vit_feat = self.vit_branch(pixel_values=x).last_hidden_state[:, 0]
        vit_feat = self.vit_proj(vit_feat)

        fused = torch.cat([cnn_feat, vit_feat], dim=1)
        return self.classifier(fused)


# ==================== Training Setup ====================

model     = CNNViTHybrid(num_classes=2).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()


# ==================== Training Loop ====================

best_val_acc = 0.0

for epoch in range(EPOCHS):

    # ── Train ──
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    train_acc = correct / total

    # ── Validate ──
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            val_loss    += loss.item()
            preds        = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / val_total

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✅ Best model saved (val_acc={val_acc:.4f})")

    scheduler.step()


# ==================== Evaluation ====================

print(f"\n📊 Classification Report (Best Val Acc: {best_val_acc:.4f}):")
print(classification_report(all_labels, all_preds,
                             target_names=['not_cheating', 'cheating']))