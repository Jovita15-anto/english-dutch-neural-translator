import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import Transformer  # noqa: E402
from inference import translate  # noqa: E402


def test_output_shape():
    m = Transformer(11, 12, d_model=64, heads=4)
    out = m(torch.randint(0, 11, (2, 5)), torch.randint(0, 12, (2, 4)))
    assert out.shape == (2, 4, 12)


def test_causal_mask_is_lower_triangular():
    mask = Transformer(11, 12).causal_mask(4)[0, 0]
    assert torch.equal(mask, torch.tril(torch.ones(4, 4)))


def test_future_tokens_do_not_change_earlier_outputs():
    m = Transformer(11, 12).eval()
    src = torch.randint(0, 11, (1, 5))
    a = torch.tensor([[2, 4, 5, 6]])
    b = torch.tensor([[2, 4, 9, 9]])
    with torch.no_grad():
        assert torch.allclose(m(src, a)[:, :2], m(src, b)[:, :2], atol=1e-5)


def test_trained_model_translates_demo_sentence():
    assert translate("i love mathematics") == "ik hou van wiskunde"
