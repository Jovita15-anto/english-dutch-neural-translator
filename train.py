"""Train the translator from scratch on a tiny parallel corpus.

NOTE: the pairs below are reconstructed from the shipped vocabularies. Replace PAIRS with
your own data to train on something larger. By default the result is saved to
model/transformer_retrained.pth so the original weights are never overwritten.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from model import Transformer

PAIRS = [
    ("i love mathematics", "ik hou van wiskunde"),
    ("i love calculus", "ik hou van calculus"),
    ("calculus is great", "calculus is geweldig"),
    ("calculus is useful", "calculus is nuttig"),
    ("mathematics is great", "wiskunde is geweldig"),
    ("mathematics is useful", "wiskunde is nuttig"),
]
SPECIALS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]


def build_vocab(sentences):
    words = sorted({w for s in sentences for w in s.split()})
    return {w: i for i, w in enumerate(SPECIALS + words)}


def encode(sentence, w2i):
    return [w2i["<SOS>"]] + [w2i[w] for w in sentence.split()] + [w2i["<EOS>"]]


def pad(batch, pad_id):
    n = max(len(x) for x in batch)
    return torch.tensor([x + [pad_id] * (n - len(x)) for x in batch])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="model/transformer_retrained.pth")
    args = ap.parse_args()
    torch.manual_seed(0)

    src_w2i = build_vocab([p[0] for p in PAIRS])
    tgt_w2i = build_vocab([p[1] for p in PAIRS])
    src = pad([encode(s, src_w2i) for s, _ in PAIRS], 0)
    tgt = pad([encode(t, tgt_w2i) for _, t in PAIRS], 0)

    model = Transformer(len(src_w2i), len(tgt_w2i), d_model=64, heads=4)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, args.epochs + 1):
        logits = model(src, tgt[:, :-1])          # teacher forcing
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out)
    (out.parent / "src_vocab_retrained.json").write_text(json.dumps(src_w2i, ensure_ascii=False))
    (out.parent / "tgt_vocab_retrained.json").write_text(json.dumps(tgt_w2i, ensure_ascii=False))
    print("saved", out)


if __name__ == "__main__":
    main()
