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

        self.q = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.k = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.v = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.projection = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residue_dropout = nn.Dropout(config.dropout)

    def forward(self, decoder_hs, encoder_hs):
        (db, dt, dc) = decoder_hs.size()
        (eb, et, ec) = encoder_hs.size()
        assert db == eb
        q = self.q(decoder_hs)
        k = self.k(encoder_hs)
        v = self.v(encoder_hs)

        q = q.view(db, dt, self.n_head, dc // self.n_head).transpose(1, 2)
        k = k.view(eb, et, self.n_head, ec // self.n_head).transpose(1, 2)
        v = v.view(eb, et, self.n_head, ec // self.n_head).transpose(1, 2)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attention_dropout(att)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v
        y = (
            y.transpose(1, 2).contiguous().view(db, dt, dc)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.residue_dropout(self.projection(y))
        return y
