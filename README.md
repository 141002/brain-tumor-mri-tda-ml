# Brain Tumor MRI Classification Using Persistent Homology

A machine learning project that uses **Topological Data Analysis (TDA)** and
**persistent homology** to transform brain MRI images into topological feature
vectors and classify them as tumor or non-tumor.

## Project Overview

This project processes grayscale brain MRI images with the GUDHI library,
constructs a cubical complex, computes persistent homology in dimensions
H0 and H1, converts the persistence diagrams into persistence images, and
uses the resulting 800-dimensional feature representation for classification.

### Dataset

The original experiment contains:

- **1,500 tumor MRI images**
- **1,500 non-tumor MRI images**
- **3,000 images total**

The dataset is in the google drive, here is the link: https://drive.google.com/drive/folders/17qlAb3d7I9ZOcyvO7KnpJTV621FGjAub?usp=sharing

## Methodology

```text
Brain MRI Images
       │
       ▼
Grayscale Conversion
       │
       ▼
Intensity Normalization
       │
       ▼
Intensity Inversion
       │
       ▼
Cubical Complex
       │
       ▼
Persistent Homology
       │
       ├──────────────┐
       ▼              ▼
      H0             H1
Connected          Holes /
Components         Loops
       │              │
       ▼              ▼
Persistence Image  Persistence Image
   20 × 20           20 × 20
       │              │
       └──────┬───────┘
              ▼
       400 + 400 = 800
       TDA Features
              │
              ▼
      Train/Test Split
          80 / 20
              │
       ┌──────┼──────────────┐
       ▼      ▼              ▼
   Logistic  RBF SVM    Random Forest
  Regression
       │      │              │
       └──────┼──────────────┘
              ▼
       Model Evaluation
```

## Persistent Homology Feature Extraction

The notebook uses GUDHI's `CubicalComplex` for the image filtration.

The preprocessing follows these steps:

1. Convert the MRI to grayscale.
2. Normalize intensity values.
3. Invert intensity for the filtration.
4. Pad the vertex grid.
5. Build a cubical complex.
6. Compute H0 and H1 persistence intervals.

The persistence-image configuration is:

```python
PersistenceImage(
    bandwidth=0.1,
    resolution=[20, 20],
    im_range=[0, 1, 0, 1]
)
```

Each persistence diagram becomes a **20 × 20 = 400-dimensional vector**.
H0 and H1 are concatenated, producing:

```text
H0 = 400 features
H1 = 400 features
-----------------
Total = 800 features
```

## Models

The original notebook evaluates:

### Logistic Regression

```python
LogisticRegression(
    max_iter=300000,
    random_state=42
)
```

### RBF Support Vector Machine

```python
SVC(
    kernel="rbf",
    C=1.0,
    random_state=42
)
```

### Random Forest

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
```

The split is stratified and uses:

```python
test_size=0.2
random_state=42
```

## Results

Results recorded from the uploaded notebook:

| Model | Accuracy | Tumor Precision | Tumor Recall | Tumor F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.83% | 80% | 82% | 81% |
| RBF SVM | 77.17% | 81% | 72% | 76% |
| **Random Forest** | **95.83%** | **94%** | **97%** | **96%** |

See [`results/README.md`](results/README.md) for confusion matrices and
additional details.

## Repository Structure

```text
brain-tumor-mri-ml/
│
├── README.md
├── brain_tumor_mri.ipynb
├── requirements.txt
├── data/
│   └── README.md
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   └── model.py
│
├── results/
│   └── README.md
│
└── .gitignore
```

## Installation

```bash
git clone <your-repository-url>
cd brain-tumor-mri-ml
pip install -r requirements.txt
```

The notebook was originally developed in Google Colab and uses Google Drive
for the MRI dataset.

## Running the Notebook

Open:

```text
brain_tumor_mri.ipynb
```

in Google Colab.

Mount Google Drive and make sure the dataset directory matches the
`DATASET_DIR` variable in the notebook.

## Source Modules

The `src/` directory separates reusable parts of the project:

- `preprocessing.py` — image normalization, filtration preparation, cubical
  complex construction, and persistence computation.
- `feature_extraction.py` — persistence diagrams to persistence-image vectors
  and the combined 800-feature representation.
- `model.py` — the three classifiers and reusable evaluation logic.

The notebook remains the main experimental walkthrough.

## Future Work

Potential next experiments include:

- StandardScaler + model pipelines
- Linear SVM
- RBF SVM hyperparameter tuning
- Extra Trees
- Gradient boosting
- L1-based feature selection
- SelectKBest feature selection
- 5-fold stratified cross-validation
- ROC-AUC comparison
- Comparing H0-only, H1-only, and H0+H1 features
- Comparing 50/100/200/400/800 selected features

## Disclaimer

This is an academic machine learning project and is **not a clinical diagnostic
system**. The model should not be used for medical diagnosis or treatment
decisions.
