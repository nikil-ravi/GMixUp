import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCN, GIN, global_mean_pool

# GIN
class GIN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(GIN, self).__init__()

        # using the built-in GIN model from PyTorch Geometric, with parameters 
        # from page 15 appendix F https://arxiv.org/pdf/2202.07179 
        self.gin = GIN(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            num_layers=5,
            out_channels=output_dim,
            act='relu',
            norm='batch_norm',
            dropout=0.0,
        )

    def forward(self, x, edge_index, batch):
        x = self.gin(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)
    
# GCN
class GCN(nn.Module):
    def __init__(self):
        pass

    def forward(self, x, edge_index):
        pass