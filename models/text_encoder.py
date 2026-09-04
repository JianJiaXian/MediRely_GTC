"""Lightweight text encoders for clinical reports.

Default ``BoWTextEncoder``: a whitespace/regex tokenizer + learned word
embedding + masked average pooling. It carries its own vocabulary which is
built from the training reports and saved inside the model checkpoint, so no
external tokenizer or large language model is required.

An optional ClinicalBERT interface is provided behind a flag but is NOT
required (and is not downloaded by default).

All encoders:
  * accept a list[str] of raw reports,
  * handle empty / missing reports (return a valid embedding),
  * expose ``.embed_dim`` and return (B, embed_dim).
"""
import re
from typing import List

import torch
import torch.nn as nn

_TOKEN_RE = re.compile(r"[a-z0-9]+")
PAD, UNK = "<pad>", "<unk>"


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


class Vocabulary:
    def __init__(self, max_size: int = 8000):
        self.max_size = max_size
        self.itos = [PAD, UNK]
        self.stoi = {PAD: 0, UNK: 1}

    def build(self, texts: List[str]):
        from collections import Counter
        counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        for tok, _ in counter.most_common(self.max_size - len(self.itos)):
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
        return self

    def encode(self, text: str, max_tokens: int) -> List[int]:
        ids = [self.stoi.get(t, 1) for t in tokenize(text)][:max_tokens]
        if not ids:
            ids = [0]  # pad token -> masked out, yields zero-ish embedding
        return ids

    def state_dict(self):
        return {"itos": self.itos, "max_size": self.max_size}

    def load_state_dict(self, state):
        self.itos = state["itos"]
        self.max_size = state.get("max_size", len(self.itos))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)


class BoWTextEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, vocab_size: int = 8000,
                 max_tokens: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_tokens = max_tokens
        self.vocab = Vocabulary(max_size=vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True))

    def fit_vocab(self, texts: List[str]):
        self.vocab.build(texts)

    @property
    def device(self):
        return self.embedding.weight.device

    def _encode_batch(self, reports: List[str]):
        max_len = 1
        encoded = []
        for r in reports:
            ids = self.vocab.encode(r, self.max_tokens)
            encoded.append(ids)
            max_len = max(max_len, len(ids))
        batch = torch.zeros(len(reports), max_len, dtype=torch.long)
        mask = torch.zeros(len(reports), max_len, dtype=torch.float32)
        for i, ids in enumerate(encoded):
            batch[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask[i, :len(ids)] = 1.0
            if not (batch[i] > 0).any():  # entirely pad (empty report)
                mask[i, :] = 0.0
        return batch.to(self.device), mask.to(self.device)

    def forward(self, reports: List[str]) -> torch.Tensor:
        ids, mask = self._encode_batch(reports)
        emb = self.embedding(ids)                       # (B, L, D)
        mask = mask.unsqueeze(-1)                        # (B, L, 1)
        summed = (emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / denom                          # masked mean
        return self.proj(pooled)

    # checkpoint helpers -------------------------------------------------
    def export_vocab(self):
        return self.vocab.state_dict()

    def import_vocab(self, state):
        self.vocab.load_state_dict(state)


class ClinicalBERTTextEncoder(nn.Module):
    """Optional ClinicalBERT wrapper. NOT used by default.

    Requires ``transformers`` and network access to download weights. Kept thin
    so the rest of the pipeline is agnostic to the text-encoder type.
    """

    def __init__(self, embed_dim: int = 256,
                 model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer  # lazy import
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.bert.config.hidden_size, embed_dim)
        self.embed_dim = embed_dim

    @property
    def device(self):
        return self.proj.weight.device

    def fit_vocab(self, texts):  # no-op, kept for interface compatibility
        return self

    def forward(self, reports: List[str]) -> torch.Tensor:
        reports = [r if r.strip() else "[PAD]" for r in reports]
        tok = self.tokenizer(reports, padding=True, truncation=True,
                             max_length=256, return_tensors="pt").to(self.device)
        out = self.bert(**tok).last_hidden_state[:, 0]   # CLS token
        return self.proj(out)


def build_text_encoder(cfg_model) -> nn.Module:
    te = cfg_model.text_encoder
    kind = te.get("type", "bow")
    embed_dim = int(te.get("embed_dim", 256))
    if kind == "clinicalbert":
        return ClinicalBERTTextEncoder(embed_dim=embed_dim)
    return BoWTextEncoder(embed_dim=embed_dim,
                          vocab_size=int(te.get("vocab_size", 8000)),
                          max_tokens=int(te.get("max_tokens", 128)))
