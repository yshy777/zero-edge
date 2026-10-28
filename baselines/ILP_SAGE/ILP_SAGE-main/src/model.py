# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super().__init__()
        self.layers = nn.ModuleList([
            SAGEConv(in_size, hidden_size, "mean"),
            SAGEConv(hidden_size, hidden_size, "mean")
        ])
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1))

    def forward(self, blocks, x):
        hidden_x = x
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hidden_x = layer(block, hidden_x)
            if layer_idx < len(self.layers) - 1:
                hidden_x = F.relu(hidden_x)
        return hidden_x
