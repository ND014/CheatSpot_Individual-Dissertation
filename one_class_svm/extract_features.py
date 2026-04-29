import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights
from PIL import Image
import os, numpy as np

class ImageDataset(Dataset):
    def __init__(self, folder, transform):
        self.transform = transform
        self.images = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        print(f"✅ Found {len(self.images)} images in {folder}")

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        return self.transform(img)


def extract(folder):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(folder, transform)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model  = torch.nn.Sequential(*list(model.children())[:-1])
    model  = model.to(device).eval()

    features = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out   = model(batch)
            out   = out.squeeze(-1).squeeze(-1)
            features.append(out.cpu().numpy())

    return np.vstack(features)


if __name__ == '__main__':
    BASE     = '/content/drive/MyDrive/dataset'
    SAVE_DIR = '/content/drive/MyDrive/one_class_svm'

    print("📦 Extracting train features...")
    train_feats = extract(f'{BASE}/train/cheating')
    np.save(f'{SAVE_DIR}/train_features.npy', train_feats)

    print("📦 Extracting val features...")
    val_feats = extract(f'{BASE}/val/cheating')
    np.save(f'{SAVE_DIR}/val_features.npy', val_feats)

    print(f"✅ Train features: {train_feats.shape}")
    print(f"✅ Val features  : {val_feats.shape}")
