"""General-purpose English -> Dutch translation using a pretrained MarianMT model.

The first call downloads the model (~300 MB) from Hugging Face and caches it.
"""
from functools import lru_cache

MODEL_NAME = "Helsinki-NLP/opus-mt-en-nl"


@lru_cache(maxsize=1)
def _load():
    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def translate_pretrained(text, num_beams=4, max_length=256):
    """Translate any English text (word, sentence or short paragraph) to Dutch."""
    import torch

    tokenizer, model = _load()
    batch = tokenizer([text], return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        out = model.generate(**batch, num_beams=num_beams, max_length=max_length)
    return tokenizer.decode(out[0], skip_special_tokens=True)


if __name__ == "__main__":
    for s in ["Hello, how are you?", "I love mathematics", "Where is the nearest train station?"]:
        print(f"{s!r} -> {translate_pretrained(s)!r}")
