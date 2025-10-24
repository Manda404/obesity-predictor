"""
visualization.py
=========================
Matplotlib-based utilities for common ML visualizations.

Design goals
------------
- Keep matplotlib only (compatible with Streamlit; no style assumptions)
- Return `matplotlib.figure.Figure` so the caller can render (Streamlit or notebooks)
- Cover essential visualizations for classification: confusion matrix, ROC (multiclass),
  and feature importances for tree models.

Author: Rostand Surel
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Optional[Sequence[str]] = None,
    normalize: Optional[str] = "true",
    title: str = "Confusion Matrix",
):
    """
    Plot a confusion matrix.

    Parameters
    ----------
    y_true : Sequence
        Ground truth labels.
    y_pred : Sequence
        Predicted labels.
    labels : Sequence[str] | None
        Class labels order for the axes.
    normalize : {"true","pred","all", None}
        Normalization mode as in sklearn ConfusionMatrixDisplay.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, colorbar=True)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_multiclass_roc(
    y_true: Sequence,
    y_proba: np.ndarray,
    class_names: Sequence[str],
    title: str = "ROC Curves (Multiclass)",
):
    """
    Plot One-vs-Rest ROC curves for multiclass predictions.

    Parameters
    ----------
    y_true : Sequence
        Ground truth labels (categorical).
    y_proba : np.ndarray
        Predicted probabilities of shape (n_samples, n_classes).
    class_names : Sequence[str]
        Class label names in the same order as columns of y_proba.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true_bin = label_binarize(y_true, classes=class_names)
    n_classes = y_true_bin.shape[1]

    fig, ax = plt.subplots()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_feature_importance(
    model,
    feature_names: Sequence[str],
    top_n: int = 20,
    title: str = "Top Feature Importances",
):
    """
    Plot top-N feature importances for tree-based models.

    Parameters
    ----------
    model : Any
        Fitted model exposing `feature_importances_` or `booster_.feature_importance()`.
    feature_names : Sequence[str]
        Feature names aligned with the model input order.
    top_n : int
        Number of features to display (sorted by importance).
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    importances = None

    # Try common APIs
    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_)
    elif hasattr(model, "booster_") and hasattr(model.booster_, "feature_importance"):
        importances = np.array(model.booster_.feature_importance())
    else:
        raise AttributeError(
            "Model does not expose feature importances via 'feature_importances_' "
            "or 'booster_.feature_importance()'."
        )

    importances = importances.astype(float)
    idx = np.argsort(importances)[::-1][:top_n]
    imp_vals = importances[idx]
    imp_names = np.array(feature_names)[idx]

    fig, ax = plt.subplots()
    ax.barh(range(len(imp_vals))[::-1], imp_vals[::-1])
    ax.set_yticks(range(len(imp_names))[::-1])
    ax.set_yticklabels(imp_names[::-1])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return fig
