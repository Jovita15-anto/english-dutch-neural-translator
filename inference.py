"""Inference for the English -> Dutch Transformer translator."""
import json
from functools import lru_cache
from pathlib import Path

import torch

from model import Transformer

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
MAX_STEPS = 20
SPECIAL = {"<PAD>", "<SOS>", "<EOS>", "<UNK>"}


@lru_cache(maxsize=1)
def load():
    """Load vocabularies and trained weights once; returns a dict of everything needed."""
    src_w2i = json.loads((MODEL_DIR / "src_vocab.json").read_text(encoding="utf-8"))
    tgt_w2i = json.loads((MODEL_DIR / "tgt_vocab.json").read_text(encoding="utf-8"))

    model = Transformer(len(src_w2i), len(tgt_w2i), d_model=64, heads=4)
    state = torch.load(MODEL_DIR / "transformer.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return {
        "model": model,
        "src_w2i": src_w2i,
        "tgt_w2i": tgt_w2i,
        "tgt_i2w": {int(v): k for k, v in tgt_w2i.items()},
    }


def supported_words():
    """English words the model knows (excluding special tokens)."""
    return [w for w in load()["src_w2i"] if w not in SPECIAL]


def tokenize(sentence):
    """Return (words, ids, unknown_words) for an English sentence."""
    w2i = load()["src_w2i"]
    words = sentence.lower().split()
    unknown = [w for w in words if w not in w2i]
    ids = [w2i["<SOS>"]] + [w2i.get(w, w2i["<UNK>"]) for w in words] + [w2i["<EOS>"]]
    return words, ids, unknown


def translate_with_details(sentence):
    """Greedy autoregressive decoding. Returns a dict with translation, token ids and unknown words."""
    m = load()
    model, tgt_w2i, tgt_i2w = m["model"], m["tgt_w2i"], m["tgt_i2w"]

    _, src_ids, unknown = tokenize(sentence)
    src = torch.tensor([src_ids], dtype=torch.long)

    tgt_ids = [tgt_w2i["<SOS>"]]
    for _ in range(MAX_STEPS):
        tgt = torch.tensor([tgt_ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(src, tgt)
        next_token = logits[:, -1, :].argmax(dim=-1).item()
        tgt_ids.append(next_token)
        if next_token == tgt_w2i["<EOS>"]:
            break

    words = [tgt_i2w[i] for i in tgt_ids if tgt_i2w[i] not in SPECIAL]
    return {
        "translation": " ".join(words),
        "src_ids": src_ids,
        "tgt_ids": tgt_ids,
        "unknown": unknown,
    }


def translate(sentence):
    return translate_with_details(sentence)["translation"]


if __name__ == "__main__":
    for s in ["i love mathematics", "calculus is great"]:
        print(f"{s!r} -> {translate(s)!r}")
