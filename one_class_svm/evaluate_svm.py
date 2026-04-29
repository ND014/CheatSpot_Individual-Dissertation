import numpy as np
from sklearn.metrics import (precision_score, recall_score,
                              f1_score, confusion_matrix)
import joblib

SAVE_DIR = '/content/drive/MyDrive/one_class_svm'

svm    = joblib.load(f'{SAVE_DIR}/svm_model.pkl')
scaler = joblib.load(f'{SAVE_DIR}/svm_scaler.pkl')

val_feats = np.load(f'{SAVE_DIR}/val_features.npy')
val_feats = scaler.transform(val_feats)

preds = svm.predict(val_feats)

true_labels  = np.ones(len(val_feats))
preds_binary = (preds == 1).astype(int)
true_binary  = true_labels.astype(int)

correct      = (preds_binary == true_binary).sum()
total        = len(true_binary)
accuracy     = correct / total * 100
detected     = (preds_binary == 1).sum()
not_detected = (preds_binary == 0).sum()

p  = precision_score(true_binary, preds_binary, zero_division=0) * 100
r  = recall_score(true_binary, preds_binary,    zero_division=0) * 100
f1 = f1_score(true_binary, preds_binary,        zero_division=0) * 100
cm = confusion_matrix(true_binary, preds_binary)

print("=" * 50)
print("   CheatSpot One-Class SVM — Results")
print("=" * 50)
print(f"  Total val images   : {total}")
print(f"  Correctly detected : {detected}  ({accuracy:.2f}%)")
print(f"  Missed detections  : {not_detected}")
print(f"  Precision          : {p:.2f}%")
print(f"  Recall             : {r:.2f}%")
print(f"  F1 Score           : {f1:.2f}%")
print(f"  Confusion Matrix   :\n{cm}")
print("=" * 50)
print("\n→ Plug into analytics.html benchmark:")
print(f"   Precision : {p:.2f}%")
print(f"   Recall    : {r:.2f}%")
print(f"   F1        : {f1:.2f}%")
print(f"   mAP proxy : {accuracy:.2f}%")
