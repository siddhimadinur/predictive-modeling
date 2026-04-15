"""ClinicalBERT-based classifier and dataset for ICD-9 chapter prediction."""

import sys
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ClinicalNoteClassifier(nn.Module):
    """Bio_ClinicalBERT encoder with a linear classification head.

    Architecture:
        ClinicalBERT pooler_output (768) -> Dropout(0.3) -> Linear(768, NUM_LABELS)
    """

    def __init__(self, num_labels: int = config.NUM_LABELS, model_name: str = config.MODEL_NAME) -> None:
        """Initialise the classifier.

        Args:
            num_labels: Number of output classes.
            model_name: HuggingFace model identifier for the BERT backbone.
        """
        super().__init__()
        self.bert: nn.Module = AutoModel.from_pretrained(model_name, use_safetensors=True)
        self.dropout: nn.Dropout = nn.Dropout(0.3)
        self.classifier: nn.Linear = nn.Linear(768, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return raw logits.

        Args:
            input_ids: Token ids of shape ``(batch, seq_len)``.
            attention_mask: Attention mask of shape ``(batch, seq_len)``.

        Returns:
            Logits tensor of shape ``(batch, num_labels)``.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled: torch.Tensor = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits: torch.Tensor = self.classifier(pooled)
        return logits


class ClinicalDataset(TorchDataset):
    """PyTorch dataset that tokenizes clinical notes on the fly.

    Each sample returns ``input_ids``, ``attention_mask``, and ``label``
    as tensors ready for the model.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = config.MAX_LENGTH,
    ) -> None:
        """Create the dataset.

        Args:
            texts: Raw clinical note strings.
            labels: Integer-encoded class labels.
            tokenizer: HuggingFace tokenizer instance.
            max_length: Maximum token sequence length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Tokenize and return a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with ``input_ids``, ``attention_mask``, and ``label``.
        """
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
