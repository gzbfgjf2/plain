import torch.nn as nn
from torch.nn import functional as F
from plain.nn.layer import LayerNorm, MLP
from plain.nn.attention import CausalSelfAttention, Attention


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)

        self.causal_self_attention = CausalSelfAttention(config)
        self.encoder_decoder_attention = Attention(config)
        self.mlp = MLP(config)

    def forward(self, word_memory_batches, x):
        x = x + self.causal_self_attention(self.ln_1(x))
        x = x + self.encoder_decoder_attention(
            word_memory_batches,
            self.ln_2(x),
        )
        x = x + self.mlp(self.ln_3(x))
        return x
