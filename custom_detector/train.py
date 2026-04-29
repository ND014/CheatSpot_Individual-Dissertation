import torch
import os
from model   import CheatNet
from dataset import get_loaders
from loss    import DetectionLoss

# ── Config ────────────────────────────────────────────
EPOCHS     = 50
LR         = 1e-3
BATCH_SIZE = 16
SAVE_DIR   = '/content/drive/MyDrive/custom_detector'
DATA_BASE  = '/content/drive/MyDrive/dataset'
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥  Using: {device}")

train_loader, val_loader = get_loaders(DATA_BASE, BATCH_SIZE)

model     = CheatNet(num_classes=1).to(device)
criterion = DetectionLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_loss = float('inf')

for epoch in range(EPOCHS):
    # ── Train ──────────────────────────────────────
    model.train()
    total_loss = 0

    for imgs, targets in train_loader:
        imgs    = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds  = model(imgs)
        loss, box_l, obj_l, noobj_l, cls_l = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_train = total_loss / len(train_loader)

    # ── Validate ───────────────────────────────────
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs    = imgs.to(device)
            targets = targets.to(device)
            preds   = model(imgs)
            loss, *_ = criterion(preds, targets)
            val_loss += loss.item()

    avg_val = val_loss / len(val_loader)
    scheduler.step()

    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
          f"Train Loss: {avg_train:.4f} | "
          f"Val Loss: {avg_val:.4f}")

    if avg_val < best_loss:
        best_loss = avg_val
        torch.save(model.state_dict(), f'{SAVE_DIR}/best_model.pth')
        print(f"   💾 Saved! (Val Loss: {avg_val:.4f})")

print(f"\n✅ Done! Best Val Loss: {best_loss:.4f}")
