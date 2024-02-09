import torch
import torch.nn as nn
from torch.nn import functional as F
from plain.nn import (
    LayerNorm,
    Mlp,
    CausalSelfAttention,
)

import math
from .mixins.optimizer import OptimizerMixin


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.d, bias=config.bias)
        self.ln_2 = LayerNorm(config.d, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.mlp = Mlp(config)

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(OptimizerMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.n_vocab, config.d)
        self.wpe = nn.Embedding(config.sequence_length, config.d)
        self.lm_head = nn.Linear(config.d, config.n_vocab, bias=False)
        self.wte.weight = self.lm_head.weight
        self.drop = nn.Dropout(config.dropout)
        self.ln = LayerNorm(config.d, bias=config.bias)

        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )

        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        n_params = self.get_num_params()
        print("number of parameters: %.2fM" % (n_params / 1e6,))

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.config.dtype == "float16")
        )
        self.loss_function = F.cross_entropy

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_optimizer(self):
        optimizer = self.configure_optimizers(
            self.config.weight_decay,
            self.config.learning_rate,
            (self.config.beta1, self.config.beta2),
            self.config.device,
        )
        return optimizer

    def training_step(self, data):
        x, y = data
        logits = self(x)
        loss = self.loss_function(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1
        )
        return logits, loss

    @torch.no_grad()
    def evaluation_step(self, data):
        x, y = data
        logits = self(x)
        loss = self.loss_function(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1
        )
        ids = torch.argmax(logits, dim=-1)
        return ids, loss

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        return logits
