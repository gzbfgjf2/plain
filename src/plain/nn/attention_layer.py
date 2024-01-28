import torch.nn as nn
from torch.nn import functional as F
from plain.nn.functional import (
    causal_mask,
    attention_forward,
)


# reference: https://github.com/karpathy/nanoGPT
class AttentionIngredient(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d % config.n_head == 0
        self.d = config.d
        self.n_head = config.n_head
        self.head_d = self.d // self.n_head
        self.dropout = config.dropout
        self.bias = config.bias
        self.sequence_length = config.sequence_length
        linear = lambda: nn.Linear(self.d, self.d, bias=self.bias)
        self.q = linear()
        self.k = linear()
        self.v = linear()
        self.projection = linear()
        self.attention_dropout = nn.Dropout(self.dropout)
        self.residue_dropout = nn.Dropout(self.dropout)
        self.register_buffer("mask", None)


class Attention(AttentionIngredient):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, state_for_value, state_for_query):
        return attention_forward(self, state_for_value, state_for_query)


class SelfAttention(AttentionIngredient):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, hs):
        return attention_forward(self, hs, hs)


class EncoderDecoderAttention(AttentionIngredient):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, encoder_hs, decoder_hs):
        return attention_forward(self, encoder_hs, decoder_hs)


class CausalSelfAttention(AttentionIngredient):
    def __init__(self, config):
        super().__init__(config)
        self.mask = causal_mask(self.sequence_length)

    def forward(self, decoder_hs):
        return attention_forward(self, decoder_hs, decoder_hs)
