"""CLI inference script for the clinical note classifier.

Usage:
    python predict.py --note "Patient presented with chest pain..."
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data.preprocess import LabelEncoder
from model.classifier import ClinicalNoteClassifier


def load_model(
    device: torch.device,
) -> tuple[ClinicalNoteClassifier, LabelEncoder, AutoTokenizer]:
    """Load the trained model, label encoder, and tokenizer.

    Args:
        device: Compute device to map the model onto.

    Returns:
        Tuple of ``(model, label_encoder, tokenizer)``.
    """
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    if not os.path.isfile(ckpt_path):
        print(f"Error: No checkpoint found at {ckpt_path}")
        print("Run training first: python training/train.py")
        sys.exit(1)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    encoder = LabelEncoder.load(checkpoint["label_encoder_path"])

    model = ClinicalNoteClassifier(num_labels=checkpoint["num_labels"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    return model, encoder, tokenizer


@torch.no_grad()
def predict(note: str, model: ClinicalNoteClassifier, encoder: LabelEncoder,
            tokenizer: AutoTokenizer, device: torch.device, top_k: int = 3) -> list[tuple[str, float]]:
    """Predict ICD-9 chapter categories for a clinical note.

    Args:
        note: Raw clinical note text.
        model: Trained classifier.
        encoder: Label encoder for decoding predictions.
        tokenizer: HuggingFace tokenizer.
        device: Compute device.
        top_k: Number of top predictions to return.

    Returns:
        List of ``(category_name, confidence)`` tuples, sorted by confidence descending.
    """
    encoding = tokenizer(
        note.lower(),
        max_length=config.MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    logits = model(input_ids, attention_mask)
    probs = F.softmax(logits, dim=-1).squeeze(0)

    top_probs, top_indices = probs.topk(top_k)
    results: list[tuple[str, float]] = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        results.append((encoder.decode(idx), prob))
    return results


def main() -> None:
    """Parse CLI arguments and print top-3 predictions."""
    parser = argparse.ArgumentParser(
        description="Predict ICD-9 diagnosis categories from a clinical note."
    )
    parser.add_argument(
        "--note",
        type=str,
        required=True,
        help="Clinical note text to classify.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to display (default: 3).",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )

    model, encoder, tokenizer = load_model(device)
    results = predict(args.note, model, encoder, tokenizer, device, args.top_k)

    print(f"\n{'='*50}")
    print("  Top predictions:")
    print(f"{'='*50}")
    for rank, (category, confidence) in enumerate(results, 1):
        print(f"  {rank}. {category:20s}  {confidence:.4f} ({confidence*100:.1f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
