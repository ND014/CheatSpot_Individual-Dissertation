# ==================== CHEATSPOT - ViT Standalone ====================

import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from google.colab import drive

drive.mount('/content/drive')

DATA_PATH  = '/content/drive/MyDrive/CheatSpot/dataset'
SAVE_PATH  = '/content/drive/MyDrive/CheatSpot/vit_standalone_best.pt'
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 2e-4
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")


# ==================== Dataset ====================

class YOLOFrameDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.transform = transform
        self.samples   = []

        for fname in os.listdir(img_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path   = os.path.join(img_dir, fname)
            label_path = os.path.join(label_dir, fname.rsplit('.', 1)[0] + '.txt')
            label      = 1 if (os.path.exists(label_path) and os.path.getsize(label_path) > 0) else 0
            self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ==================== Transforms ====================

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# ==================== Data Loaders ====================

train_dataset = YOLOFrameDataset(f'{DATA_PATH}/train/images', f'{DATA_PATH}/train/labels', train_transform)
val_dataset   = YOLOFrameDataset(f'{DATA_PATH}/valid/images', f'{DATA_PATH}/valid/labels', val_transform)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")


# ==================== Model ====================

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=2,
    ignore_mismatched_sizes=True
).to(DEVICE)


# ==================== Training Setup ====================

optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion    = nn.CrossEntropyLoss()
best_val_acc = 0.0
history      = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}


# ==================== Training Loop ====================

for epoch in range(EPOCHS):

    # ── Train ──
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(pixel_values=imgs).logits
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
            outputs     = model(pixel_values=imgs).logits
            loss        = criterion(outputs, labels)
            val_loss   += loss.item()
            preds       = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / val_total

    history['train_loss'].append(train_loss / len(train_loader))
    history['val_loss'].append(val_loss / len(val_loader))
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    scheduler.step()

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss/len(train_loader):.4f} | Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss/len(val_loader):.4f} | Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✅ Saved best model (val_acc={val_acc:.4f})")


# ==================== Evaluation ====================

print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds,
                             target_names=['not_cheating', 'cheating']))