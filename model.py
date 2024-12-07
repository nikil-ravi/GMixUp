import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn
from torch_geometric.nn import global_mean_pool


# GIN
class GIN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(GIN, self).__init__()

        # using the built-in GIN model from PyTorch Geometric, with parameters 
        # from page 15 appendix F https://arxiv.org/pdf/2202.07179 
        self.gin = torch_geometric.nn.GIN(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            num_layers=5,
            out_channels=hidden_dim,
            act='relu',
            norm='batch_norm',
            dropout=0.0,
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.gin(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def predict(self, x, edge_index, batch):
        x = self.forward(x, edge_index, batch)
        return torch.argmax(x, dim=1)


# GCN
class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(GCN, self).__init__()

        # using the built-in GCN model from PyTorch Geometric, with parameters 
        # from page 15 appendix F https://arxiv.org/pdf/2202.07179 
        self.gcn = torch_geometric.nn.GCN(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            num_layers=4,
            out_channels=output_dim,
            act='relu',
        )

    def forward(self, x, edge_index, batch):
        x = self.gcn(x, edge_index)
        return global_mean_pool(x, batch)
    
    def predict(self, x, edge_index, batch):
        x = self.forward(x, edge_index, batch)
        return torch.argmax(x, dim=1)
    