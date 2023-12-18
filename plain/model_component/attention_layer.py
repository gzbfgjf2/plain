import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# reference: https://github.com/karpathy/nanoGPT


def causal_mask(sequence_length):
    mask = torch.ones(sequence_length, sequence_length)
    mask = mask.view(1, 1, sequence_length, sequence_length)
    mask = torch.tril(mask)
    return mask


def scaled_dot_product_attention(q, k, v, dropout, mask):
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    if mask:
        (b, s, d) = k.size()
        att = att.masked_fill(mask[:, :, :s, :s] == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    att = dropout(att)
    y = att @ v
    return y


def head_split(tensor, b, s, n_head, head_d):
    return tensor.view(b, s, n_head, head_d).transpose(1, 2)


def attention_forward(layer, decoder_hs, encoder_hs):
    # batch, sequence, dimension
    (b, s, d) = decoder_hs.size()
    q = layer.q(decoder_hs)
    k = layer.k(encoder_hs)
    v = layer.v(encoder_hs)

    args = b, s, layer.n_head, layer.head_d
    q = head_split(q, *args)
    k = head_split(k, *args)
    v = head_split(v, *args)

    y = scaled_dot_product_attention(q, k, v, layer.dropout, layer.mask)
    y = y.transpose(1, 2).contiguous().view(b, s, d)
    y = layer.residue_dropout(layer.projection(y))
    return y


class AttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d % config.n_head == 0
        self.d = config.d
        self.n_head = config.n_head
        self.head_d = self.d // self.n_head
        self.dropout = config.dropout
        self.bias = config.bias
        self.sequence_length = config.sequence_length
        self.mask_name = config.mask_name

        linear = lambda: nn.Linear(self.d, self.d, bias=self.bias)
        self.q = linear()
        self.k = linear()
        self.v = linear()
        self.projection = linear()
        self.attention_dropout = nn.Dropout(self.dropout)
        self.residue_dropout = nn.Dropout(self.dropout)
        self.mask = None

    def forward(self, decoder_hs, encoder_hs):
        attention_forward(self, decoder_hs, encoder_hs)


class EncoderAttentionLayer(AttentionLayer):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, decoder_hs):
        attention_forward(self, decoder_hs, decoder_hs)


class EncoderDecoderAttentionLayer(AttentionLayer):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, encoder_hs, decoder_hs):
        attention_forward(self, encoder_hs, decoder_hs)


class DecoderAttentionLayer(AttentionLayer):
    def __init__(self, config):
        super().__init__(config)
        self.mask = causal_mask(self.sequence_length)

    def forward(self, decoder_hs):
        attention_forward(self, decoder_hs, decoder_hs)