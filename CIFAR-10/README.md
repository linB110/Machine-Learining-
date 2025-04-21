# 📦 CIFAR-10 Image Classification with MobileNetV2

這個專案使用 PyTorch 和預訓練的 MobileNetV2 模型，對 CIFAR-10 資料集進行影像分類。訓練過程中整合了：

- ✅ 自動混合精度訓練（AMP）
- ✅ ReduceLROnPlateau 學習率調整策略
- ✅ Early Stopping
- ✅ 模型微調（fine-tuning）
- ✅ 訓練可視化與混淆矩陣分析

---

### ✅ 環境需求

- Python 3.10+
- PyTorch 2.x
- torchvision
- tqdm
- seaborn
- scikit-learn
- matplotlib
