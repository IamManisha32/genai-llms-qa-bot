"""
dataset.py

Custom PyTorch Dataset + DataLoader for the course-knowledge-base chatbot.

Reads data/knowledge_base.json, tokenizes each (question, answer) pair,
and batches them using a custom collate function that pads sequences
to the same length within each batch (a core LLM data-pipeline skill).

Run directly to see a sanity-check batch printed to console:
    python src/dataset.py
"""

import json
from typing import List, Dict, Callable, Optional

import torch
from torch.utils.data import Dataset, DataLoader


class QADataset(Dataset):
    """Wraps the knowledge_base.json Q&A pairs as a PyTorch Dataset.

    Each item returns the raw question/answer text plus tokenized
    input_ids for the question (tokenization is pluggable via `tokenizer_fn`,
    so you can swap in BertTokenizer, a simple whitespace tokenizer, etc.
    without changing this class).
    """

    def __init__(self, json_path: str, tokenizer_fn: Callable[[str], List[int]]):
        with open(json_path, "r", encoding="utf-8") as f:
            self.records: List[Dict] = json.load(f)
        self.tokenizer_fn = tokenizer_fn

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        question_ids = self.tokenizer_fn(record["question"])
        return {
            "id": record["id"],
            "topic": record["topic"],
            "question": record["question"],
            "answer": record["answer"],
            "input_ids": question_ids,
        }


def collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict:
    """Custom collate function: pads variable-length input_ids to the
    longest sequence in the batch, and returns an attention_mask so the
    model knows which tokens are real vs. padding.

    This is the exact mechanic the course covers: "batch, shuffle, and
    feed data to models" plus "reads, tokenizes, and batches text data".
    """
    ids = [item["input_ids"] for item in batch]
    lengths = [len(seq) for seq in ids]
    max_len = max(lengths)

    padded_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, seq in enumerate(ids):
        padded_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, : len(seq)] = 1

    return {
        "ids": [item["id"] for item in batch],
        "topics": [item["topic"] for item in batch],
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
        "input_ids": padded_ids,
        "attention_mask": attention_mask,
    }


def build_dataloader(
    json_path: str,
    tokenizer_fn: Callable[[str], List[int]],
    batch_size: int = 4,
    shuffle: bool = True,
    pad_token_id: int = 0,
) -> DataLoader:
    """Convenience factory: builds the Dataset + DataLoader in one call."""
    dataset = QADataset(json_path, tokenizer_fn)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=pad_token_id),
    )


# ---------------------------------------------------------------------------
# A minimal whitespace tokenizer used ONLY for local testing without internet.
# In the real app, swap this for BertTokenizer.encode (see comment below).
# ---------------------------------------------------------------------------
def dummy_whitespace_tokenizer(text: str) -> List[int]:
    """Maps each whitespace-split word to a fake ID (hash-based, deterministic).
    This is NOT a real tokenizer -- just here so the pipeline is testable
    without downloading a model. Replace with a real tokenizer in production."""
    return [abs(hash(word)) % 30000 for word in text.lower().split()]


if __name__ == "__main__":
    import os

    json_path = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.json")

    # --- For local testing (no internet needed) ---
    loader = build_dataloader(json_path, dummy_whitespace_tokenizer, batch_size=4)

    print(f"Dataset size: {len(loader.dataset)} records")
    print(f"Number of batches: {len(loader)}\n")

    for batch_num, batch in enumerate(loader):
        print(f"--- Batch {batch_num + 1} ---")
        print("Topics:", batch["topics"])
        print("input_ids shape:", batch["input_ids"].shape)
        print("attention_mask shape:", batch["attention_mask"].shape)
        print("Sample input_ids row 0:", batch["input_ids"][0].tolist())
        print()

    # --- To use the REAL BertTokenizer instead (run on your machine, needs internet) ---
    # from transformers import BertTokenizer
    # bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    # real_loader = build_dataloader(
    #     json_path,
    #     tokenizer_fn=lambda text: bert_tok.encode(text, add_special_tokens=True),
    #     batch_size=4,
    #     pad_token_id=bert_tok.pad_token_id,
    # )
