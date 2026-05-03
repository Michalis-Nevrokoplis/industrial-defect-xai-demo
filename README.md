# Industrial Defect Classification with Explainable AI

A computer vision portfolio project for **industrial quality control**, focused on detecting steel surface defects using deep learning and preparing the model for explainability with Grad-CAM.

## Project Overview

This project applies a pretrained convolutional neural network to an industrial visual inspection task. The goal is to classify steel surface images as either **Defect** or **No Defect**, and then use Explainable AI methods to understand which image regions influenced the model's decision.

The project is designed as a practical demonstration of how deep learning and XAI can support quality control in manufacturing environments.

## Why This Project Matters

In industrial inspection, model accuracy alone is not enough. A model may correctly classify images, but engineers and operators also need to understand **why** a decision was made, especially when decisions affect product quality, rework, scrap, or customer risk.

This project combines:

- Industrial visual inspection
- Transfer learning with CNNs
- Binary defect classification
- Model evaluation using quality-control-relevant metrics
- Explainability with Grad-CAM as the next development stage

## Dataset

The project uses the **Severstal Steel Defect Detection** dataset from Kaggle.

The original dataset contains steel surface images and defect annotations. For this project, the task was simplified into a binary classification problem:

- `0` = No Defect
- `1` = Defect

The raw dataset is not included in this repository because of file size and licensing considerations.

Expected local data structure:

```text
data/raw/
├── train_images/
├── test_images/
├── train.csv
└── sample_submission.csv
```

## Project Structure

```text
industrial-defect-xai-demo/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_gradcam_explainability.ipynb   # planned / in progress
│
├── models/
│   └── mobilenetv2_baseline.keras
│
├── results/
│   ├── training_accuracy.png
│   ├── training_loss.png
│   └── confusion_matrix.png
│
├── data/                                # ignored by GitHub
├── .gitignore
├── requirements.txt
└── README.md
```

## Stage 1 — Data Exploration

The first notebook explores the dataset and prepares the binary labels used for training.

Main steps:

- Loaded and inspected the original annotation file
- Cleaned one problematic row with missing values
- Created image-level binary labels
- Checked the distribution of defective and non-defective images
- Saved the binary labels locally for training

Dataset summary after preprocessing:

| Category | Number of images | Percentage |
|---|---:|---:|
| Defect | 6,666 | 53.04% |
| No Defect | 5,902 | 46.96% |
| Total | 12,568 | 100% |

The dataset is reasonably balanced for a first binary classification baseline.

## Stage 2 — Model Training

The second notebook trains a baseline deep learning model for binary defect classification.

Because TensorFlow was not available locally, the actual training was performed in **Google Colab with GPU acceleration**. The notebook and project structure were organized locally using **VS Code** and then pushed to GitHub.

### Model

The model uses transfer learning with **MobileNetV2 pretrained on ImageNet**.

Architecture summary:

- MobileNetV2 backbone
- Frozen pretrained feature extractor
- Global Average Pooling layer
- Dropout layer
- Dense sigmoid output for binary classification

### Training Setup

| Setting | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| Split | 70% train / 15% validation / 15% test |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss | Binary crossentropy |
| Epochs | 10 |

## Results

The MobileNetV2 baseline achieved the following results on the test set:

| Metric | Score |
|---|---:|
| Accuracy | 85.15% |
| Precision | 84.48% |
| Recall | 88.20% |
| F1-score | 86.30% |

### Confusion Matrix

|  | Predicted No Defect | Predicted Defect |
|---|---:|---:|
| Actual No Defect | 724 | 162 |
| Actual Defect | 118 | 882 |

From an industrial quality control perspective, the most critical error type is the **false negative** case: a defective image classified as non-defective.

In this baseline model, there were **118 false negatives** on the test set.

This makes recall especially important, because recall measures how many actual defective images were successfully detected.

## Current Limitations

This is an initial baseline model, so there are several important limitations:

- The original steel strip images are very wide, but they were resized to 224 × 224 for compatibility with MobileNetV2.
- Square resizing may distort the aspect ratio and reduce sensitivity to small or thin defects.
- The model currently performs binary classification only, not defect localization or segmentation.
- Grad-CAM explainability is the next stage and is not yet fully completed.
- The model has not yet been optimized to reduce false negatives specifically.

## Next Stage — Grad-CAM Explainability

The next step is to add Explainable AI using Grad-CAM.

Planned workflow:

```text
load trained model
choose sample steel images
predict No Defect / Defect
generate Grad-CAM heatmaps
overlay heatmaps on original images
save Grad-CAM examples
```

The goal is to visualize which image regions influenced the CNN prediction. This is important in industrial inspection because a model should ideally focus on actual defect regions rather than irrelevant background patterns.

Planned output:

```text
results/gradcam_examples.png
```

## Possible Future Improvements

Future extensions could include:

- Error analysis of false positives and false negatives
- Threshold tuning to reduce false negatives
- Fine-tuning the MobileNetV2 backbone
- Comparison with other pretrained CNNs such as EfficientNetB0 or ResNet
- Patch-based classification to better handle wide steel strip images
- Aspect-ratio-preserving preprocessing
- Additional XAI methods such as Grad-CAM++, Score-CAM, Integrated Gradients, Occlusion Sensitivity, LIME, or SHAP
- Comparison between Grad-CAM heatmaps and true defect masks from the original annotations

## Skills Demonstrated

This project demonstrates practical experience with:

- Python
- Data exploration and preprocessing
- Computer vision for industrial inspection
- Transfer learning
- CNN-based binary classification
- TensorFlow / Keras
- Google Colab GPU training
- Model evaluation with accuracy, precision, recall, F1-score, and confusion matrix
- Git and GitHub project organization
- Explainable AI for quality control applications

## Status

| Stage | Status |
|---|---|
| Data exploration | Completed |
| MobileNetV2 training | Completed |
| Model evaluation | Completed |
| Grad-CAM explainability | Planned / in progress |
| README documentation | Completed |

## Project Goal

The purpose of this repository is not only to build a high-performing classifier, but also to show how AI models can become more useful and trustworthy in industrial quality control when combined with explainability methods.

This project is part of my broader interest in **Industrial AI, visual inspection, quality control, and Explainable AI**.