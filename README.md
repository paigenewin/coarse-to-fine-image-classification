# Coarse-to-Fine Image Classification with Machine Learning Approaches and Deep Visual Features

This project investigates image classification under a coarse-to-fine setting, comparing handcrafted image features, pretrained ResNet18 visual embeddings, and multiple classification approaches across two tasks of increasing visual difficulty.

- **Task 1 – Coarse-grained animal classification:** classification of 10 animal categories.
- **Task 2 – Fine-grained bird classification:** classification of 10 visually similar bird species.

---

## Project Overview

Both tasks follow a similar classification pipeline:

1. Load image metadata and provided image features.
2. Extract additional engineered image features.
3. Extract deep visual embeddings using an ImageNet-pretrained ResNet18.
4. Compare provided/engineered features, ResNet18 embeddings, and their combination.
5. Compare multiple classification models using stratified 5-fold cross-validation.
6. Tune the strongest-performing model.
7. Evaluate the selected model on a stratified holdout validation set.
8. Train the final model on the full training set and generate test predictions.

ResNet18 is used as a fixed feature extractor rather than being fine-tuned on the target datasets.

---

## Tasks

### Task 1: Coarse-Grained Animal Classification

Task 1 classifies images into 10 animal categories:

- Bird
- Butterfly
- Cat
- Deer
- Dog
- Elephant
- Frog
- Horse
- Sheep
- Spider

The dataset contains **3,750 training images** and **1,250 test images**.

### Task 2: Fine-Grained Bird Classification

Task 2 classifies bird images into 10 visually similar species:

- Cardinal
- Blue Jay
- American Goldfinch
- Red-winged Blackbird
- House Sparrow
- Song Sparrow
- Herring Gull
- Ring-billed Gull
- Yellow Warbler
- Wilson Warbler

The dataset contains **417 training images** and **180 test images**.

---

## Feature Representations

Three feature representations were evaluated.

### 1. Provided and Engineered Features

The provided features include:

- Colour histograms
- HOG features reduced using PCA
- Edge density
- Texture variance
- Average colour channels

Additional image features were extracted directly from the original images.

### 2. ResNet18 Embeddings

A ResNet18 model pretrained on ImageNet was used as a fixed feature extractor.

The final classification layer was removed, producing a **512-dimensional deep visual representation** for each image.

### 3. Combined Features

The provided and engineered image features were combined with the ResNet18 embeddings to create the final feature representation.

The combined representation achieved the strongest feature performance across both tasks.

---

## Models

### Task 1

The following classifiers were evaluated:

- Logistic Regression
- Random Forest
- LightGBM
- XGBoost
- CatBoost
- RBF Support Vector Machine (SVM)
- Soft-voting ensemble

The final selected model was an **RBF SVM**.

### Task 2

The following classifiers were evaluated:

- Extra Trees
- LightGBM
- CatBoost
- Soft-voting ensemble

The final selected model was **Extra Trees**.

---

## Results

| Task | Selected Model | Mean CV Accuracy | Holdout Accuracy | Kaggle Accuracy |
| --- | --- | ---: | ---: | ---: |
| Task 1: Coarse-grained | RBF SVM | 87.28% | 89.73% | 87.25% |
| Task 2: Fine-grained | Extra Trees | 86.34% | 82.14% | 84.44% |

ResNet18 embeddings substantially improved classification performance compared with the provided and engineered features, particularly for fine-grained bird classification. Combining the handcrafted and deep visual features produced the strongest feature representation for both tasks.

Error analysis showed that the most difficult predictions were concentrated among visually similar classes. Prominent confusions included dog–cat and deer–horse in Task 1, and Herring Gull–Ring-billed Gull and Wilson Warbler–Yellow Warbler in Task 2.

---

## Repository Structure

```text
coarse-to-fine-image-classification/
├── data/
│   ├── task1_data/
│   └── task2_data/
│
├── notebooks/
│   ├── coarse_grained_classification.ipynb
│   └── fine_grained_classification.ipynb
│
├── report/
│   └── coarse_to_fine_grained_classification.pdf
│
├── results/
│   ├── task1_predictions.csv
│   └── task2_predictions.csv
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Data

The datasets are not included in this repository.

To reproduce the experiments, place the required files inside `data/task1_data/` and `data/task2_data/` as described below.

### Task 1 Data

Task 1 requires the following input data:

```text
data/task1_data/
├── images/
├── additional_features.csv
├── color_histogram.csv
├── hog_pca.csv
├── test_metadata.csv
└── train_metadata.csv
```

The `images/` directory contains the original images referenced by the training and test metadata.

During execution, the Task 1 notebook generates and caches:

```text
engineered_image_features_task1.csv
resnet18_features_task1.csv
```

These files do not need to be manually added before running the notebook.

### Task 2 Data

Task 2 requires the following input data:

```text
data/task2_data/
├── images/
├── additional_features.csv
├── class_mapping.csv
├── color_histogram.csv
├── hog_pca.csv
├── test_metadata.csv
└── train_metadata.csv
```

Task 2 additionally uses `class_mapping.csv` for the fine-grained bird class mappings.

During execution, the Task 2 notebook generates and caches:

```text
engineered_image_features_task2.csv
resnet18_features_task2.csv
```

These files do not need to be manually added before running the notebook.

> **Note:** The repository contains the complete feature extraction, modelling and evaluation code. However, the original images and input feature files are required to execute the notebooks from start to finish.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd coarse-to-fine-image-classification
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

After placing the required datasets in the `data/` directories, start Jupyter:

```bash
jupyter notebook
```

The notebooks are located in:

```text
notebooks/
```

The Task 1 notebook expects:

```python
data_dir = Path("../data/task1_data")
```

The Task 2 notebook expects:

```python
data_dir = Path("../data/task2_data")
```

Run:

```text
notebooks/coarse_grained_classification.ipynb
```

for Task 1 and:

```text
notebooks/fine_grained_classification.ipynb
```

for Task 2.

On the first run, the ImageNet-pretrained ResNet18 weights may be downloaded automatically by `torchvision`.

The first execution can take longer because the engineered features and ResNet18 embeddings are extracted from the original images. These features are cached locally and reused on subsequent runs.

---

## Prediction Outputs

The final test-set predictions generated by the selected models are available in:

```text
results/
├── task1_predictions.csv
└── task2_predictions.csv
```

These predictions achieved Kaggle accuracies of:

- **Task 1:** 87.25%
- **Task 2:** 84.44%

---

## Evaluation

Models were compared using **stratified 5-fold cross-validation**.

Accuracy was used as the primary model-selection metric because the datasets were approximately balanced across the classes and classification errors were treated equally.

The selected models were additionally evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices
- Misclassification analysis

---

## Requirements

The main Python dependencies are:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
torch
torchvision
xgboost
lightgbm
catboost
jupyter
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Report

The full project report is available at:

```text
report/coarse_to_fine_grained_classification.pdf
```

It contains the complete methodology, experimental results, error analysis, discussion, limitations and conclusions.

---

## Author

**Ha Phuong Nguyen**