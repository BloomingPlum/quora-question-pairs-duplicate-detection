import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)


def display_metrics(y_true, y_pred, y_prob):
    """
    Print standard classification metrics.

    Parameters:
    - y_true: array-like of shape (n_samples,) — Ground truth labels.
    - y_pred: array-like of shape (n_samples,) — Predicted class labels.
    - y_prob: array-like of shape (n_samples,) — Predicted probabilities for the positive class.

    Outputs:
    - Prints accuracy, precision, recall, F1 score, log loss, and ROC AUC.
    """
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred):.4f}")
    print(f"Log-Loss : {log_loss(y_true, y_prob):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y_true, y_prob):.4f}")


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """
    Plot a confusion matrix with normalized percentages and counts.

    Parameters:
    - y_true: array-like — Ground truth labels.
    - y_pred: array-like — Predicted class labels.
    - labels: list — Optional label names for axes.
    - title: str — Title for the plot.

    Output:
    - Displays a seaborn heatmap of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_perc = cm / cm.sum(axis=1, keepdims=True)
    annot = [[f"{p:.2f}\n({c})" for p, c in zip(row_perc, row_cnt)]
             for row_perc, row_cnt in zip(cm_perc, cm)]
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm_perc, annot=annot, fmt='', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()


def plot_roc_curve(y_true, y_prob, title="ROC Curve"):
    """
    Plot the ROC curve for a binary classifier.

    Parameters:
    - y_true: array-like — Ground truth labels.
    - y_prob: array-like — Predicted probabilities for the positive class.
    - title: str — Plot title.

    Output:
    - Displays the ROC curve with AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR");  plt.ylabel("TPR")
    plt.title(title);   plt.legend();  plt.grid(True);  plt.show()


def predict_on_loader(model, loader, device):
    """
    Run a model on a dataloader and collect predictions.

    Parameters:
    - model: torch.nn.Module — Trained PyTorch model.
    - loader: DataLoader — Dataloader for the dataset (validation/test).
    - device: torch.device — Device to run inference on (e.g., 'cuda' or 'cpu').

    Returns:
    - y_true: np.ndarray — True labels.
    - y_pred: np.ndarray — Predicted class labels.
    - y_prob: np.ndarray — Predicted probabilities for the positive class.
    """
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            probs  = F.softmax(logits, dim=1)[:, 1]   # prob of positive class
            y_true.extend(yb.numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)