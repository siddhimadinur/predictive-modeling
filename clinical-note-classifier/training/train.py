"""Training loop for the clinical note classifier.

Loads preprocessed data, fine-tunes Bio_ClinicalBERT with a linear warmup
schedule, evaluates on the validation set each epoch, and saves the best
checkpoint by macro-F1.
"""

import csv
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.preprocess import prepare_datasets
from model.classifier import ClinicalDataset, ClinicalNoteClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloader(
    texts: list[str],
    labels: list[int],
    tokenizer: AutoTokenizer,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Wrap texts/labels in a ClinicalDataset and DataLoader."""
    ds = ClinicalDataset(texts, labels, tokenizer, config.MAX_LENGTH)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: ClinicalNoteClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch.

    Args:
        model: The classifier.
        loader: Training DataLoader.
        optimizer: AdamW optimizer.
        scheduler: Linear warmup scheduler.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(loader.dataset)  # type: ignore[arg-type]


@torch.no_grad()
def evaluate(
    model: ClinicalNoteClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on a validation/test set.

    Args:
        model: The classifier.
        loader: Evaluation DataLoader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Tuple of ``(average_loss, macro_f1)``.
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in tqdm(loader, desc="  eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item() * input_ids.size(0)

        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)  # type: ignore[arg-type]
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


def train() -> None:
    """Run the full training pipeline: preprocess, train, checkpoint."""
    device = get_device()
    print(f"Using device: {device}")

    # Data
    datasets, encoder = prepare_datasets()
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    train_loader = build_dataloader(
        datasets["train"]["text"],
        datasets["train"]["label"],
        tokenizer,
        config.BATCH_SIZE,
        shuffle=True,
    )
    val_loader = build_dataloader(
        datasets["validation"]["text"],
        datasets["validation"]["label"],
        tokenizer,
        config.BATCH_SIZE,
        shuffle=False,
    )

    # Model
    model = ClinicalNoteClassifier(num_labels=len(encoder)).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    # Logging setup
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_path = os.path.join(config.LOG_DIR, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "val_loss", "val_macro_f1", "elapsed_s"])

    best_f1 = 0.0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    start = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start
        log_writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_f1:.4f}", f"{elapsed:.1f}"])
        log_file.flush()

        print(
            f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_labels": len(encoder),
                    "label_encoder_path": config.LABEL_ENCODER_PATH,
                },
                checkpoint_path,
            )
            print(f"  -> saved best checkpoint (F1={best_f1:.4f})")

    log_file.close()
    print(f"\nTraining complete. Best val macro-F1: {best_f1:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    train()
