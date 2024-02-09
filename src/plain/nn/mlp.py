import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.d, 4 * config.d, bias=config.bias)
        self.gelu = nn.GELU()
        self.projection = nn.Linear(4 * config.d, config.d, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.gelu(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x
