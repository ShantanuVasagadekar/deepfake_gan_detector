"""
evaluate_model.py — Benchmark evaluation for the Deepfake Detection System.

Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC AUC

Usage:
    python evaluate_model.py --dataset dataset --model saved_models/deepfake_classifier.pth
"""

import os
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from classifier_model import DeepfakeClassifier
from face_preprocessing import DeepfakeDataset
from utils import get_device

SAVE_DIR = Path(__file__).resolve().parent / "saved_models"
IMG_SIZE = 256


def get_eval_transforms():
    """Standard evaluation transforms at 256x256."""
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def evaluate(dataset_root: str, model_path: str = None):
    """Run full evaluation and print metrics."""
    device = get_device()

    # Load model
    if model_path is None:
        model_path = str(SAVE_DIR / "deepfake_classifier.pth")

    model = DeepfakeClassifier(pretrained=False).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"[EVAL] Loaded model from {model_path}")

    # Load dataset
    dataset = DeepfakeDataset(
        root=dataset_root,
        transform=get_eval_transforms(),
        image_size=IMG_SIZE,
        balance=False,
        use_label_conditional_aug=False,
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())

    all_labels = []
    all_scores = []
    all_preds = []

    print(f"[EVAL] Evaluating on {len(dataset):,} images...")

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels_np = labels.numpy()

            # Forward pass (model auto-computes FFT and noise residual)
            logits = model(images)
            scores = torch.sigmoid(logits).view(-1).cpu().numpy()
            preds = (scores >= 0.5).astype(int)

            all_labels.extend(labels_np)
            all_scores.extend(scores)
            all_preds.extend(preds)

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    all_preds = np.array(all_preds)

    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    accuracy = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # ROC AUC (manual calculation)
    auc = _compute_auc(all_labels, all_scores)

    # Print results
    print("\n" + "=" * 55)
    print("  DEEPFAKE DETECTION — EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Dataset         : {dataset_root}")
    print(f"  Total Images    : {len(all_labels):,}")
    print(f"  Real Images     : {int((all_labels == 0).sum()):,}")
    print(f"  Fake Images     : {int((all_labels == 1).sum()):,}")
    print("-" * 55)
    print(f"  Accuracy        : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  Precision       : {precision:.4f}")
    print(f"  Recall          : {recall:.4f}")
    print(f"  F1 Score        : {f1:.4f}")
    print(f"  ROC AUC         : {auc:.4f}")
    print("-" * 55)
    print(f"  True Positives  : {tp}")
    print(f"  True Negatives  : {tn}")
    print(f"  False Positives : {fp}")
    print(f"  False Negatives : {fn}")
    print("=" * 55)

    # Plot confusion matrix
    _plot_confusion_matrix(tp, tn, fp, fn)
    _plot_roc_curve(all_labels, all_scores, auc)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def _compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC using the trapezoidal rule."""
    # Sort by descending score
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    tpr_list = [0.0]
    fpr_list = [0.0]

    tp_count = 0
    fp_count = 0

    for label in sorted_labels:
        if label == 1:
            tp_count += 1
        else:
            fp_count += 1
        tpr_list.append(tp_count / n_pos)
        fpr_list.append(fp_count / n_neg)

    # Trapezoidal rule
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

    return auc


def _plot_confusion_matrix(tp, tn, fp, fn):
    """Save confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    matrix = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(matrix, cmap='Blues')

    labels_axis = ["Real (0)", "Fake (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels_axis)
    ax.set_yticklabels(labels_axis)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]),
                    ha='center', va='center', fontsize=18,
                    color='white' if matrix[i, j] > matrix.max() / 2 else 'black')

    plt.colorbar(im)
    plt.tight_layout()
    save_path = Path(__file__).resolve().parent / "confusion_matrix.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Confusion matrix saved to {save_path}")


def _plot_roc_curve(labels: np.ndarray, scores: np.ndarray, auc: float):
    """Save ROC curve plot."""
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp_count = 0
    fp_count = 0

    for label in sorted_labels:
        if label == 1:
            tp_count += 1
        else:
            fp_count += 1
        tpr_list.append(tp_count / n_pos)
        fpr_list.append(fp_count / n_neg)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_list, tpr_list, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Deepfake Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = Path(__file__).resolve().parent / "roc_curve.png"
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] ROC curve saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deepfake classifier.")
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    evaluate(args.dataset, args.model)
