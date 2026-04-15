"""Evaluate the best checkpoint on the held-out test set.

Prints accuracy, macro-F1, weighted-F1, a full classification report,
and saves a confusion matrix heatmap as a PNG.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.preprocess import LabelEncoder, prepare_datasets
from model.classifier import ClinicalDataset, ClinicalNoteClassifier


def load_best_model(device: torch.device) -> tuple[ClinicalNoteClassifier, LabelEncoder]:
    """Load the best saved checkpoint and its label encoder.

    Args:
        device: Compute device to map the model onto.

    Returns:
        A ``(model, label_encoder)`` tuple.
    """
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    encoder = LabelEncoder.load(checkpoint["label_encoder_path"])
    model = ClinicalNoteClassifier(num_labels=checkpoint["num_labels"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, encoder


@torch.no_grad()
def run_test(
    model: ClinicalNoteClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Run inference over the test DataLoader.

    Args:
        model: The classifier in eval mode.
        loader: Test DataLoader.
        device: Compute device.

    Returns:
        Tuple of ``(all_true_labels, all_predicted_labels)`` as integer lists.
    """
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in tqdm(loader, desc="Testing"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return all_labels, all_preds


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
    save_path: str,
) -> None:
    """Plot and save a confusion matrix heatmap.

    Args:
        y_true: Ground-truth label indices.
        y_pred: Predicted label indices.
        label_names: Human-readable class names in index order.
        save_path: Output PNG file path.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Clinical Note Classifier")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")


def evaluate() -> None:
    """Load the best model, run on the test set, and print results."""
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    model, encoder = load_best_model(device)
    datasets, _ = prepare_datasets()

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    test_ds = ClinicalDataset(
        datasets["test"]["text"],
        datasets["test"]["label"],
        tokenizer,
        config.MAX_LENGTH,
    )
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    y_true, y_pred = run_test(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*50}")
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  Macro-F1:     {macro_f1:.4f}")
    print(f"  Weighted-F1:  {weighted_f1:.4f}")
    print(f"{'='*50}\n")

    label_names = [encoder.decode(i) for i in range(len(encoder))]
    present_labels = sorted(set(y_true + y_pred))
    present_names = [encoder.decode(i) for i in present_labels]

    print(classification_report(
        y_true, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    ))

    os.makedirs(config.LOG_DIR, exist_ok=True)
    cm_path = os.path.join(config.LOG_DIR, "confusion_matrix.png")
    save_confusion_matrix(y_true, y_pred, label_names, cm_path)


if __name__ == "__main__":
    evaluate()
