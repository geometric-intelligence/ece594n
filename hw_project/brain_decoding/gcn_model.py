import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import numpy as np


class GCN(torch.nn.Module):
    def __init__(self, edge_index, edge_weight, n_roi, batch_size=16, n_timepoints=1, n_classes=9):
        super().__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.n_roi = n_roi
        self.batch_size = batch_size

        self.conv1 = tg.nn.ChebConv(
            in_channels=n_timepoints, out_channels=32, K=2, bias=True
        )
        self.conv2 = tg.nn.ChebConv(in_channels=32, out_channels=32, K=2, bias=True)
        self.conv3 = tg.nn.ChebConv(in_channels=32, out_channels=batch_size, K=2, bias=True)
        self.fc1 = nn.Linear(self.n_roi * batch_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        batch_vector = torch.arange(x.size(0), dtype=int)
        x = torch.flatten(x, 1)
        x = tg.nn.global_mean_pool(x, batch_vector)
        x = x.view(-1, self.n_roi * self.batch_size)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x