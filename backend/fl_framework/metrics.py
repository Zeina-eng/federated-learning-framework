"""
Evaluation and Clinical Metrics Computation:
Confusion Matrix, Accuracy, Precision, Sensitivity/Recall, Specificity, F1-Score, and AUC-ROC.
"""
import numpy as np
from typing import Dict, List, Tuple


def compute_binary_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Computes comprehensive clinical validation metrics from predicted probabilities.
    Returns: accuracy, precision, sensitivity, specificity, f1_score, auc, and loss.
    """
    y_true = np.array(y_true, dtype=int).flatten()
    y_pred_probs = np.array(y_pred_probs, dtype=float).flatten()

    y_pred = (y_pred_probs >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if (precision + sensitivity) > 0:
        f1 = (2.0 * precision * sensitivity) / (precision + sensitivity)
    else:
        f1 = 0.0

    # Cross-Entropy / Log Loss
    eps = 1e-12
    clipped_probs = np.clip(y_pred_probs, eps, 1.0 - eps)
    bce_loss = -np.mean(y_true * np.log(clipped_probs) + (1.0 - y_true) * np.log(1.0 - clipped_probs))

    # Area Under ROC Curve (AUC) via Mann-Whitney U / Trapezoidal Integration
    auc = compute_roc_auc(y_true, y_pred_probs)

    return {
        "accuracy": round(float(accuracy * 100.0), 2),
        "precision": round(float(precision * 100.0), 2),
        "sensitivity": round(float(sensitivity * 100.0), 2),
        "specificity": round(float(specificity * 100.0), 2),
        "f1": round(float(f1 * 100.0), 2),
        "auc": round(float(auc), 3),
        "loss": round(float(bce_loss), 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Computes Area Under Receiver Operating Characteristic Curve."""
    if len(y_true) == 0:
        return 0.5

    pos_count = int(np.sum(y_true == 1))
    neg_count = int(np.sum(y_true == 0))

    if pos_count == 0 or neg_count == 0:
        return 1.0

    # Sort descending by prediction score
    order = np.argsort(-y_scores)
    sorted_y = y_true[order]

    current_pos = 0
    auc_sum = 0

    for label in sorted_y:
        if label == 1:
            current_pos += 1
        else:
            auc_sum += current_pos

    return float(auc_sum / (pos_count * neg_count))
