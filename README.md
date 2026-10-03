# 🌍 Transformer Translator

An English-to-Dutch neural machine translator built **from scratch in PyTorch** (no `nn.Transformer`), with a Streamlit web app.

![![Screenshot](../image.png)]

**Input:** `i love mathematics` → **Output:** `ik hou van wiskunde`

The app has two modes:
- **Pretrained model:** type any English text. Uses Helsinki-NLP `opus-mt-en-nl` (MarianMT) via Hugging Face.
- **From-scratch Transformer:** my own PyTorch implementation, limited to a tiny vocabulary.

Comparing the two shows how far data and scale matter: same architecture family, very different capability.

## Why this project
I wanted to understand every component of the Transformer from "Attention Is All You Need", so each block is implemented by hand in [`model.py`](model.py): positional encoding, multi-head attention, feed-forward layers, layer norm, encoder and decoder layers, and causal masking.

## Architecture

```
English sentence → tokenize (+<SOS>/<EOS>) → embedding + positional encoding
  → Encoder (self-attention → feed-forward)
  → Decoder (masked self-attention → cross-attention → feed-forward)
  → Linear → argmax → next Dutch token (repeat until <EOS>)
```

| Setting | Value |
|---|---|
| Embedding dimension | 64 |
| Attention heads | 4 |
| Encoder / decoder layers | 1 / 1 |
| Source / target vocabulary | 11 / 12 tokens |
| Decoding | Greedy, max 20 steps |

## Features
- Multi-head self-attention and encoder-decoder cross-attention
- Sinusoidal positional encoding
- Causal masking and autoregressive decoding
- Streamlit app with example buttons, unknown-word warnings and a token-ID viewer
- `train.py` to retrain the model, and `pytest` tests for the model and inference

## Project structure
```
Transformer/
├── model/                 # vocabularies + trained weights
├── tests/test_model.py    # shape, causal-mask and inference tests
├── app.py                 # Streamlit UI
├── .streamlit/config.toml # theme
├── inference.py           # loading + greedy decoding
├── model.py               # Transformer implementation (from scratch)
├── pretrained.py          # pretrained MarianMT translator (any text)
├── train.py               # training script (teacher forcing)
├── requirements.txt
└── README.md
```

## Run it
```bash
pip install -r requirements.txt
python inference.py          # quick check of the from-scratch model
python pretrained.py         # quick check of the pretrained model (downloads ~300 MB once)
streamlit run app.py         # open http://localhost:8501
pytest                       # run tests
python train.py              # retrain (saves to model/transformer_retrained.pth)
```

## Limitations
This is a learning project. My from-scratch model knows only a handful of words (`i, love, mathematics, calculus, is, great, useful`) and was trained on a tiny corpus, so it only translates sentences built from that vocabulary. Unknown words are mapped to `<UNK>` and the app warns about them. Next steps: a larger parallel corpus (e.g. Tatoeba or OPUS), subword tokenization, more layers, beam search and BLEU evaluation.

## Author
**Anto Jovita** · B.Tech Artificial Intelligence & Data Science
