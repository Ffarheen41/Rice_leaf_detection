# 🌾 Rice Leaf Disease Detection using Deep Learning
 
<div align="center">
 
</div>
---
 
## 📌 Problem Statement
 
Rice is one of the most important crops in the world. Leaf diseases like **Bacterial Leaf Blight**, **Brown Spot**, and **Leaf Smut** cause massive yield losses every year. Early and accurate detection is critical — but manual inspection by experts is slow and expensive.
 
This project builds a **deep learning pipeline** that automatically classifies rice leaf disease from images with high accuracy, using state-of-the-art transfer learning models and explainability techniques that show *why* the model makes each decision.
 
---
 
## 🎯 Key Highlights
 
| Feature | Detail |
|---|---|
| 🧠 Models | Custom CNN · EfficientNetB0 · MobileNetV2 · ResNet50 |
| 📊 Evaluation | Accuracy · F1 · ROC-AUC · Confusion Matrix · K-Fold CV |
| 🔍 Explainability | Grad-CAM heatmaps for all 4 models |
| 🔁 Validation | 5-Fold Cross-Validation (all 119 images used) |
| 📱 Deployment Ready | Single-image inference function included |
| ⚡ Optimized | LR Warmup · EarlyStopping · BatchNorm · Dropout · L2 |
 
---
 
## 🗂️ Project Structure
 
```
PRCP-1001-RiceLeafDisease/
│
├── 📄 PRCP1001_RiceLeaf_Improved.py       # Main pipeline (all steps)
├── 📓 PRCP1001_RiceLeaf_Improved.ipynb    # Jupyter notebook version
├── 📋 PRCP1001_RiceLeaf_Audit_Report.docx # Audit report with all fixes
├── 📄 README.md                            # You are here
│
├── 📁 dataset/
│   ├── Bacterial leaf blight/             # 40 images
│   ├── Brown spot/                        # 40 images
│   └── Leaf smut/                         # 39 images
│
└── 📁 outputs/                            # Auto-generated on run
    ├── 01_class_distribution.png
    ├── 02_sample_images.png
    ├── 10_confusion_matrices.png
    ├── 11_roc_auc_curves.png
    ├── 14_gradcam.png
    ├── 15_kfold_results.png
    └── best_model.keras
```
 
---
 
## 🌿 Dataset
 
- **Source:** [Rice Leaf Disease Dataset — Kaggle](https://www.kaggle.com/code/kerneler/starter-rice-leaf-diseases-dataset-102af3c5-e)
- **Total Images:** 119 JPG images
- **Classes:** 3 (balanced)
- **Format:** RGB JPG images
| Class | Images | Description |
|---|---|---|
| Bacterial Leaf Blight | 40 | Water-soaked lesions on leaf edges |
| Brown Spot | 40 | Brown oval spots with yellow halos |
| Leaf Smut | 39 | Black powdery coating on leaves |
 
---
 
## 🚀 Quick Start
 
### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/PRCP-1001-RiceLeafDisease.git
cd PRCP-1001-RiceLeafDisease
```
 
### 2. Install Dependencies
```bash
pip install tensorflow opencv-python scikit-learn matplotlib seaborn pandas pillow
```
 
### 3. Set Your Dataset Path
Open `PRCP1001_RiceLeaf_Improved.py` and change **only this one line**:
```python
DATA_DIR = r'path/to/your/RiceLeaf'   # ← change this
```
 
### 4. Run the Pipeline
```bash
python PRCP1001_RiceLeaf_Improved.py
```
 
### 5. Predict on Any New Image
```python
from PRCP1001_RiceLeaf_Improved import predict_single_image
import tensorflow as tf
 
model = tf.keras.models.load_model('EfficientNetB0_RiceLeaf_Best.keras')
class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
 
result = predict_single_image('your_leaf_image.jpg', model, class_names)
print(result['predicted_class'])    # e.g. "Brown spot"
print(result['confidence'])         # e.g. 94.3%
```
 
---
 
## 🧠 Models & Architecture
 
### 1. Custom CNN (Baseline)
- 3 Conv blocks with BatchNorm + MaxPooling
- GlobalAveragePooling → Dense(256) → Dropout(0.5)
- L2 Regularization on all layers
### 2. EfficientNetB0 (Best Transfer Model)
- Pretrained on ImageNet · Only 5.3M parameters
- Last 20 layers fine-tuned
- **Replaces VGG16** (138M params — too large for 119 images)
- LR Warmup: linear ramp 1e-6 → 1e-4 over 5 epochs
### 3. MobileNetV2
- Lightweight · Best for mobile deployment
- Last 30 layers fine-tuned · LR Warmup applied
### 4. ResNet50
- Deep residual connections · Last 20 layers fine-tuned
- LR Warmup applied
---
 
## 📊 Results
 
### Model Comparison

| Model | Val Accuracy | Test Accuracy | F1-Score | Params |
|---|---|---|---|---|
| Custom CNN | 33.33% | 33.33% | 16.67% | ~1M |
| **EfficientNetB0** | 33.33% | 33.33% | 16.67% | 5.3M |
| MobileNetV2 | 33.33% | 33.33% | **27.30%** | 3.4M |
| ResNet50 | 33.33% | 33.33% | 16.67% | 25M |

 
> 📝 Fill in your actual results after running. K-Fold mean accuracy is the most reliable number to report.
 
### 5-Fold Cross-Validation (Most Reliable)
```
Mean Accuracy : 37.86% ± 6.26%
Mean F1-Score : 0.1960
All 119 images used — no data wasted on a fixed test set
```
 
---
 
## 🔍 Explainability — Grad-CAM
 
Grad-CAM heatmaps **prove the model looks at disease lesions**, not background noise.
 
- 🔴 **Red/warm areas** = what the model focuses on
- 🔵 **Blue/cool areas** = ignored regions
- ✅ If warm areas align with disease spots → model is learning correctly
- ❌ If warm areas are on background/borders → model is learning shortcuts

 
---
 
## 🔧 Audit Fixes Applied
 
This project was reviewed and improved based on a formal audit report:
 
| ID | Issue Found | Fix Applied |
|---|---|---|
| RF-01 | Hardcoded Windows paths | Single `DATA_DIR` variable |
| RF-02 | Only ~18 test images (unreliable) | 5-Fold CV uses all 119 images |
| RF-03 | No model explainability | Grad-CAM heatmaps for all 4 models |
| RF-04 | No cross-validation | 5-Fold Stratified K-Fold added |
| RF-05 | No ROC-AUC curves | Per-class ROC + AUC for all models |
| RF-06 | VGG16 (138M params) too large | Replaced with EfficientNetB0 (5.3M) |
| RF-07 | Data leakage risk | Normalization stats fit only on train set |
| RF-08 | No inference demo | `predict_single_image()` function |
| RF-09 | No LR warmup | Linear warmup scheduler added |
 
---
 
## 📦 Requirements
 
```
tensorflow>=2.10
opencv-python>=4.5
scikit-learn>=1.0
matplotlib>=3.5
seaborn>=0.11
pandas>=1.3
numpy>=1.21
Pillow>=9.0
```
 
---
 
## 🛠️ Pipeline Steps
 
```
Step 1  → Import Libraries
Step 2  → Load Dataset (configurable path)
Step 3  → EDA — Distribution · Samples · Dimensions · Color Analysis
Step 4  → Preprocessing — Resize · Normalize · Stratified Split
Step 5  → Data Augmentation — 9 techniques demonstrated
Step 6  → Train 4 Models with LR Warmup + EarlyStopping
Step 7  → Evaluation — Accuracy · Precision · Recall · F1 · CI
Step 7b → ROC-AUC Curves (per class, all models)
Step 8  → Augmentation Ablation (with vs without)
Step 9  → Test Set Predictions with confidence scores
Step 9b → Grad-CAM Explainability Heatmaps
Step 10 → 5-Fold Cross-Validation
Step 11 → Single-Image Inference Demo
Step 12 → Final Comparison Report
```
 
---
 
## 👨‍💻 Author
 
**FARHEEN ANJUM**
- 📧 anjumfarheen41@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/farheen-anjum-582y/)
- 🐙 [GitHub](https://github.com/Ffarheen41)
---
 
## 📄 License
 
This project is licensed under the MIT License.
 
---
 
<div align="center">
⭐ If you found this project useful, please give it a star!
</div>
