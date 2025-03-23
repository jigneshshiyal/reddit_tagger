from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
import numpy as np

def evaluate_model(y_true, y_pred, y_probs=None, average="weighted"):
    """
    Computes and returns various evaluation metrics for multi-class classification.

    Parameters:
    - y_true: Actual class labels (list or array)
    - y_pred: Predicted class labels (list or array)
    - y_probs: Predicted probabilities (for AUC & Log Loss, optional, shape: [n_samples, n_classes])
    - average: Averaging method for multi-class metrics ('micro', 'macro', 'weighted', 'samples')

    Returns:
    - Dictionary with all metrics
    """
    metrics = {}

    cm = confusion_matrix(y_true, y_pred)
    metrics["Confusion Matrix"] = cm.tolist()  # Convert to list for better readability

    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred, average=average)
    metrics["Recall"] = recall_score(y_true, y_pred, average=average)
    metrics["F1-Score"] = f1_score(y_true, y_pred, average=average)

    if y_probs is not None:
        metrics["AUC-ROC"] = roc_auc_score(y_true, y_probs, multi_class="ovr", average=average)
        metrics["Log Loss"] = log_loss(y_true, y_probs)

    return metrics
