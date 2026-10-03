import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)

        div = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        # buffer (not persistent) so .to(device) works and saved weights still load
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()

        self.h = heads
        self.d = d_model // heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        B = q.size(0)

        Q = self.q(q).view(
            B, -1, self.h, self.d
        ).transpose(1, 2)

        K = self.k(k).view(
            B, -1, self.h, self.d
        ).transpose(1, 2)

        V = self.v(v).view(
            B, -1, self.h, self.d
        ).transpose(1, 2)

        scores = torch.matmul(
            Q, K.transpose(-2, -1)
        ) / math.sqrt(self.d)

        if mask is not None:
            scores = scores.masked_fill(
                mask == 0,
                -1e9
            )

        attn = torch.softmax(scores, -1)

        ctx = torch.matmul(attn, V)

        ctx = (
            ctx.transpose(1, 2)
            .contiguous()
            .view(B, -1, self.h * self.d)
        )

        return self.out(ctx)


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()

        self.attn = MultiHeadAttention(
            d_model,
            heads
        )

        self.n1 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model)

        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x):

        x = self.n1(
            x + self.attn(x, x, x)
        )

        x = self.n2(
            x + self.ff(x)
        )

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model,
            heads
        )

        self.n1 = nn.LayerNorm(d_model)

        self.cross = MultiHeadAttention(
            d_model,
            heads
        )

        self.n2 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model)

        self.n3 = nn.LayerNorm(d_model)

    def forward(self, x, enc, mask):

        x = self.n1(
            x + self.self_attn(
                x, x, x, mask
            )
        )

        x = self.n2(
            x + self.cross(
                x, enc, enc
            )
        )

        x = self.n3(
            x + self.ff(x)
        )

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        d_model=64,
        heads=4
    ):
        super().__init__()

        self.src_emb = nn.Embedding(
            src_vocab,
            d_model
        )

        self.tgt_emb = nn.Embedding(
            tgt_vocab,
            d_model
        )

        self.pos = PositionalEncoding(
            d_model
        )

        self.encoder = EncoderLayer(
            d_model,
            heads
        )

        self.decoder = DecoderLayer(
            d_model,
            heads
        )

        self.fc = nn.Linear(
            d_model,
            tgt_vocab
        )

    def causal_mask(self, n, device=None):

        m = torch.tril(
            torch.ones(n, n, device=device)
        )

        return m.unsqueeze(0).unsqueeze(0)

    def forward(self, src, tgt):

        src = self.pos(
            self.src_emb(src)
        )

        enc = self.encoder(src)

        tgt = self.pos(
            self.tgt_emb(tgt)
        )

        mask = self.causal_mask(
            tgt.size(1),
            tgt.device
        )

        dec = self.decoder(
            tgt,
            enc,
            mask
        )

        return self.fc(dec)
