import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# reference: https://github.com/karpathy/nanoGPT


class AttendToEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        linear = lambda: nn.Linear(self.n_embd, self.n_embd, bias=config.bias)

        self.q = linear()
        self.k = linear()
        self.v = linear()
        self.projection = linear()
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residue_dropout = nn.Dropout(config.dropout)

    def forward(self, decoder_hs, encoder_hs):
        (b, s, d) = decoder_hs.size()
        q = self.q(decoder_hs)
        k = self.k(encoder_hs)
        v = self.v(encoder_hs)

        heads_view = lambda tensor: tensor.view(
            b, s, self.n_head, d // self.n_head
        ).transpose(1, 2)

        q = heads_view(q)
        k = heads_view(k)
        v = heads_view(v)

        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attention_dropout(att)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v
        y = (
            y.transpose(1, 2).contiguous().view(b, s, d)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.residue_dropout(self.projection(y))
        return y
