# Quora Question Pairs Duplicate Detection

## Project Overview

This project addresses the problem of detecting whether two questions from Quora are semantically equivalent (i.e., duplicates). The solution is a combination of traditional NLP, machine learning, and deep learning techniques, including GloVe embeddings, TF-IDF, classical classifiers, and BERT-based models.

The workflow is organized into several Jupyter notebooks, each focusing on a different stage or modeling approach. The project is modular, with reusable utility scripts and a clear separation between data, models, and code.

---

## Directory Structure

```
quora_project/
│
├── 1. EDA_preprocessing_LR_XGBoost.ipynb
├── 2. GloVe_NN_classifier.ipynb
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
  - Loading and inspecting the Quora question pairs dataset.
  - Handling missing values and exploring class distribution.
  - Analyzing question lengths, special patterns, and word statistics.
  - Performing text preprocessing using spaCy and NLTK.
  - Generating features using TF-IDF and GloVe embeddings.
  - Training and evaluating Logistic Regression and XGBoost models on TF-IDF and GloVe features.
  - Preparing data for downstream modeling.

### 2. GloVe_NN_classifier.ipynb
- **Purpose:** Training a neural network classifier using GloVe embeddings and engineered features.
- **Key Steps:**
  - Loading precomputed GloVe vectors and engineered features.
  - Splitting data into training and validation sets.
  - Defining and training a PyTorch neural network.
  - Evaluating model performance and saving the best model.

### 3. BERT_embeddings.ipynb
- **Purpose:** Feature extraction using BERT and training a classifier on top of BERT embeddings.
- **Key Steps:**
  - Loading data and splitting into train/validation/test.
  - Extracting sentence embeddings for each question using a pretrained BERT model.
  - Concatenating, subtracting, and multiplying embeddings to form feature vectors.
  - Training a neural network classifier on these features.
  - Evaluating and visualizing results.

### 4. BERT_finetuning.ipynb
- **Purpose:** End-to-end fine-tuning of a BERT model for duplicate question detection.
- **Key Steps:**
  - Loading and splitting data.
  - Preparing PyTorch datasets and dataloaders.
  - Fine-tuning a BERT model using HuggingFace Transformers.
  - Evaluating and visualizing model performance.

---

## Model Performance Comparison

| Model                      | Validation F1 | Validation Log-Loss | Test F1 | Test Log-Loss |
|----------------------------|:-------------:|:-------------------:|:-------:|:-------------:|
| Logistic Regression (TF-IDF) | 0.7135 | 0.4192 | 0.7110 | 0.4262 |
| XGBoost (TF-IDF)           | 0.7192 | 0.3990 | 0.7208 | 0.4045 |
| Logistic Regression (GloVe) | 0.6048 | 0.5163 | 0.6067 | 0.5127 |
| XGBoost (GloVe)            | 0.7231 | 0.4048 | 0.7249 | 0.4054 |
| Neural Network (GloVe)    | 0.7721 | 0.3722 | 0.7708 | 0.3760 |
| BERT Embeddings + NN       | 0.7771 | 0.3910 | 0.7793 | 0.3880 |
| BERT Fine-tuned            | 0.8650 | 0.3537 | 0.8659 | 0.3515 |

