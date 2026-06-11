import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# AG News topic names, in the dataset's label order
CLASSES = ["World", "Sports", "Business", "Sci/Tech"]
NUM_CLASSES = len(CLASSES)


class FakeTextData(Dataset):
    """Random token ids and labels — used by the smoke run, no downloads."""

    def __init__(self, n, seq_len=32, vocab_size=256):
        g = torch.Generator().manual_seed(0)
        self.input_ids = torch.randint(0, vocab_size, (n, seq_len), generator=g)
        self.labels = torch.randint(0, NUM_CLASSES, (n,), generator=g)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i],
                "attention_mask": torch.ones_like(self.input_ids[i]),
                "labels": self.labels[i]}


def build_tokenizer(backbone):
    """Tokenizer for the backbone; decoder-only models need a pad token picked."""
    tok = AutoTokenizer.from_pretrained(backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def make_collate(tokenizer, max_length):
    """Collate raw texts into padded token batches on the fly."""
    def collate(batch):
        enc = tokenizer([b["text"] for b in batch], truncation=True,
                        max_length=max_length, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([b["label"] for b in batch])
        return dict(enc)
    return collate


def build_loaders(cfg):
    """Build the train/val dataloaders for the configured dataset."""
    if cfg.data.fake:
        return (DataLoader(FakeTextData(64), batch_size=cfg.data.batch_size, shuffle=True),
                DataLoader(FakeTextData(32), batch_size=cfg.data.batch_size), NUM_CLASSES)

    ds = load_dataset("ag_news")
    train_ds = ds["train"].shuffle(seed=cfg.seed)
    if cfg.data.train_subset:
        train_ds = train_ds.select(range(cfg.data.train_subset))

    collate = make_collate(build_tokenizer(cfg.model.backbone), cfg.data.max_length)
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers, collate_fn=collate,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(ds["test"], batch_size=cfg.data.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers, collate_fn=collate,
                            pin_memory=True)
    return train_loader, val_loader, NUM_CLASSES


def sample_rows(cfg, n=6):
    """A few (text, label) pairs from the train split for the samples figure."""
    rows = load_dataset("ag_news")["train"].shuffle(seed=cfg.seed).select(range(n))
    return [(row["text"], CLASSES[row["label"]]) for row in rows]
