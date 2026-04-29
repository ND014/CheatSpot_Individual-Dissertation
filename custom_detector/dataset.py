import os, torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class CheatDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, img_size=416, S=13):
        self.img_dir  = img_dir
        self.lbl_dir  = lbl_dir
        self.img_size = img_size
        self.S        = S          # grid size

        self.images = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        print(f"✅ Loaded {len(self.images)} images from {img_dir}")

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        lbl_path = os.path.join(
            self.lbl_dir,
            img_name.rsplit('.', 1)[0] + '.txt'
        )

        # Load image
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # Build target grid (6, S, S)
        # 6 = [tx, ty, tw, th, objectness, class]
        target = torch.zeros(6, self.S, self.S)

        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                lines = f.read().strip().split('\n')

            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue

                cls, cx, cy, w, h = map(float, parts[:5])

                # Find which grid cell this box belongs to
                grid_x = int(cx * self.S)
                grid_y = int(cy * self.S)

                # Clamp to valid range
                grid_x = min(grid_x, self.S - 1)
                grid_y = min(grid_y, self.S - 1)

                # Only assign if cell not already taken
                if target[4, grid_y, grid_x] == 0:
                    # tx, ty = offset within cell
                    tx = cx * self.S - grid_x
                    ty = cy * self.S - grid_y

                    target[0, grid_y, grid_x] = tx
                    target[1, grid_y, grid_x] = ty
                    target[2, grid_y, grid_x] = w
                    target[3, grid_y, grid_x] = h
                    target[4, grid_y, grid_x] = 1.0  # objectness
                    target[5, grid_y, grid_x] = cls  # class

        return img, target


def get_loaders(base='/content/drive/MyDrive/dataset', batch_size=16):
    train_dataset = CheatDataset(
        img_dir=f'{base}/train/images',
        lbl_dir=f'{base}/train/labels'
    )
    val_dataset = CheatDataset(
        img_dir=f'{base}/val/images',
        lbl_dir=f'{base}/val/labels'
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader
