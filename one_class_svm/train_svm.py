import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib

SAVE_DIR = '/content/drive/MyDrive/one_class_svm'

print("📂 Loading features...")
train_feats = np.load(f'{SAVE_DIR}/train_features.npy')
print(f"✅ Train features shape: {train_feats.shape}")

scaler      = StandardScaler()
train_feats = scaler.fit_transform(train_feats)

print("🏋️ Training One-Class SVM...")
svm = OneClassSVM(
    kernel='rbf',
    nu=0.1,
    gamma='scale'
)
svm.fit(train_feats)

joblib.dump(svm,    f'{SAVE_DIR}/svm_model.pkl')
joblib.dump(scaler, f'{SAVE_DIR}/svm_scaler.pkl')

print("✅ SVM trained and saved!")
print(f"   Support vectors: {svm.support_vectors_.shape[0]}")
