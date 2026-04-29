import torch
import numpy as np
from model   import CheatNet
from dataset import get_loaders
from utils   import decode_predictions, iou

SAVE_DIR  = '/content/drive/MyDrive/custom_detector'
DATA_BASE = '/content/drive/MyDrive/dataset'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CheatNet(num_classes=1).to(device)
model.load_state_dict(torch.load(f'{SAVE_DIR}/best_model.pth',
                                  map_location=device))
model.eval()

_, val_loader = get_loaders(DATA_BASE, batch_size=16)

TP = FP = FN = 0
iou_threshold = 0.5

with torch.no_grad():
    for imgs, targets in val_loader:
        imgs    = imgs.to(device)
        preds   = model(imgs)
        results = decode_predictions(preds.cpu(), conf_threshold=0.5)

        for b in range(imgs.shape[0]):
            detections = results[b]
            target     = targets[b]

            # Get true boxes from target grid
            obj_mask = target[4] == 1
            true_boxes = []
            for i in range(target.shape[1]):
                for j in range(target.shape[2]):
                    if target[4, i, j] == 1:
                        S  = target.shape[1]
                        cx = (j + target[0, i, j].item()) / S
                        cy = (i + target[1, i, j].item()) / S
                        w  = target[2, i, j].item()
                        h  = target[3, i, j].item()
                        true_boxes.append([cx - w/2, cy - h/2,
                                           cx + w/2, cy + h/2])

            if len(detections) == 0 and len(true_boxes) == 0:
                continue
            elif len(detections) == 0:
                FN += len(true_boxes)
                continue
            elif len(true_boxes) == 0:
                FP += len(detections)
                continue

            matched = set()
            for det in detections:
                pred_box = torch.tensor([det['box']])
                best_iou = 0
                best_idx = -1
                for idx, tb in enumerate(true_boxes):
                    if idx in matched:
                        continue
                    iou_val = iou(pred_box,
                                  torch.tensor([tb])).item()
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_idx = idx

                if best_iou >= iou_threshold and best_idx not in matched:
                    TP += 1
                    matched.add(best_idx)
                else:
                    FP += 1

            FN += len(true_boxes) - len(matched)

precision = TP / (TP + FP + 1e-6) * 100
recall    = TP / (TP + FN + 1e-6) * 100
f1        = 2 * precision * recall / (precision + recall + 1e-6)

print("=" * 50)
print("   CheatNet Custom Detector — Results")
print("=" * 50)
print(f"  TP: {TP} | FP: {FP} | FN: {FN}")
print(f"  Precision : {precision:.2f}%")
print(f"  Recall    : {recall:.2f}%")
print(f"  F1 Score  : {f1:.2f}%")
print("=" * 50)
print("\n→ Plug into analytics.html:")
print(f"   Precision : {precision:.2f}%")
print(f"   Recall    : {recall:.2f}%")
print(f"   F1        : {f1:.2f}%")
