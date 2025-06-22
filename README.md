# Quora Question Pairs Duplicate Detection

## Project Overview

This project addresses the problem of detecting whether two questions from Quora are semantically equivalent (i.e., duplicates). The solution leverages a combination of traditional NLP, machine learning, and deep learning techniques, including GloVe embeddings, TF-IDF, classical classifiers, and BERT-based models.

The workflow is organized into several Jupyter notebooks, each focusing on a different stage or modeling approach. The project is modular, with reusable utility scripts and a clear separation between data, models, and code.

---

## Directory Structure

```
quora_project/
│
├── 1. EDA_preprocessing_LR_XGBoost.ipynb
├── 2. GloVe_CBOW_classifier.ipynb
├── 3. BERT_embeddings.ipynb
├── 4. BERT_finetuning.ipynb
│
└── utils/
    ├── metrics_utils.py
    ├── plot_utils.py
    └── preprocess_utils.py
```

---

## Notebooks Overview

### 1. EDA_preprocessing_LR_XGBoost.ipynb
- **Purpose:** Data loading, cleaning, exploratory data analysis (EDA), feature engineering, and baseline modeling with Logistic Regression and XGBoost.
- **Key Steps:**
  - Loads and inspects the Quora question pairs dataset.
  - Handles missing values and explores class distribution.
  - Analyzes question lengths, special patterns, and word statistics.
  - Performs text preprocessing using spaCy and NLTK.
  - Generates features using TF-IDF and GloVe embeddings.
  - Trains and evaluates Logistic Regression and XGBoost models on TF-IDF and GloVe features.
  - Prepares data for downstream modeling.

### 2. GloVe_CBOW_classifier.ipynb
- **Purpose:** Training a neural network classifier (CBOW-style) using GloVe embeddings and engineered features.
- **Key Steps:**
  - Loads precomputed GloVe vectors and engineered features.
  - Splits data into training and validation sets.
  - Defines and trains a PyTorch neural network (CBOW architecture).
  - Evaluates model performance and saves the best model.

### 3. BERT_embeddings.ipynb
- **Purpose:** Feature extraction using BERT and training a classifier on top of BERT embeddings.
- **Key Steps:**
  - Loads data and splits into train/validation/test.
  - Extracts sentence embeddings for each question using a pretrained BERT model.
  - Concatenates, subtracts, and multiplies embeddings to form feature vectors.
  - Trains a neural network classifier on these features.
  - Evaluates and visualizes results.

### 4. BERT_finetuning.ipynb
- **Purpose:** End-to-end fine-tuning of a BERT model for duplicate question detection.
- **Key Steps:**
  - Loads and splits data.
  - Prepares PyTorch datasets and dataloaders.
  - Fine-tunes a BERT model using HuggingFace Transformers.
  - Evaluates and visualizes model performance.

---

## Model Performance Comparison

| Model                      | Validation F1 | Validation Log-Loss | Test F1 | Test Log-Loss |
|----------------------------|:-------------:|:-------------------:|:-------:|:-------------:|
| Logistic Regression (TF-IDF) | 0.7135 | 0.4192 | 0.7110 | 0.4262 |
| XGBoost (TF-IDF)           | 0.7192 | 0.3990 | 0.7231 | 0.4048 |
| Logistic Regression (GloVe) | 0.6048 | 0.5163 | 0.6067 | 0.5127 |
| XGBoost (GloVe)            | 0.7231 | 0.4048 | 0.7231 | 0.4048 |
| CBOW Neural Net (GloVe)    | 0.7721 | 0.3722 | 0.7717 | 0.3720 |
| BERT Embeddings + NN       | 0.7747 | 0.3139 | 0.7771 | 0.3029 |
| BERT Fine-tuned            | 0.8650 | 0.3537 | 0.8659 | 0.3515 |

