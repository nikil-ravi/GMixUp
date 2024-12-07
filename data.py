
from pathlib import Path
from typing import Tuple

from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
import torch_geometric.utils
import torch
import torch.nn.functional as F


class DatasetFromList(InMemoryDataset):
    '''
    From: https://stackoverflow.com/a/78488771
    '''
    def __init__(self, listOfDataObjects):
        super().__init__()
        self.data, self.slices = self.collate(listOfDataObjects)
    
    def __getitem__(self, idx):
        return self.get(idx)



def load_data(
    dataset: str,
    data_cache_path: Path
) -> Dataset:
    '''
    Parameters:
        dataset: the name of the TU dataset
        data_cache_path: directory for downloading/reading the data
        train_val_test_split: tuple indicating the data split ratio (doesn't )
    Returns the full dataset
    '''
    ds = TUDataset(str(data_cache_path), dataset)
    print(f'Loaded dataset {dataset}: ', end='')
    ds.print_summary()
    return ds
    

def split_data(
    dataset: Dataset,
    train_amount: float,
    test_amount: float
):
    '''
    Parameters:
        dataset: input dataset
        train/test_amount: amount of train/test data
            - don't have to sum to 1
    '''
    t = train_amount + test_amount
    train_amount /= t
    test_amount /= t

    N = len(dataset)
    ntrain = int(train_amount * N)

    train = dataset.index_select(slice(0, ntrain))
    test = dataset.index_select(slice(ntrain, N))

    print(f'Train: {len(train)} graphs | Test: {len(test)} graphs')
    return train, test


def create_node_features(dataset: Dataset):
    '''
    Based on https://github.com/ahxt/g-mixup/blob/0813bb32319abbfb29fe82b9ee0c353a6862a1ad/src/gmixup.py#L32
    '''
    assert dataset[0].x is None
    
    dataset = list(dataset)
    # Get node degrees: list of tensors of shape (num_nodes,)
    all_degrees = []
    for graph in dataset:
        all_degrees.append(torch_geometric.utils.degree(graph.edge_index[0], dtype=torch.long))
        graph.num_nodes = int(torch.max(graph.edge_index)) + 1
    max_degree = max(d.max().item() for d in all_degrees)
    # If sparse enough, use onehot
    if max_degree < 2000:
        print('Generating node features with one-hot degrees')
        for graph, graph_degrees in zip(dataset, all_degrees):
            graph.x = F.one_hot(graph_degrees, num_classes=max_degree+1).float()
    # Else use degree z-score
    else:
        print('Generating node features degree z-scores')
        std, mean = torch.std_mean(torch.cat(all_degrees).float())
        std, mean = std.item(), mean.item()
        for graph, graph_degrees in zip(dataset, all_degrees):
            graph.x = ((graph_degrees - mean) / std).view(-1, 1)
    dataset = DatasetFromList(dataset)
    return dataset

            
    
    max_degree = 0
    degs = []
    for data in dataset:
        degs += [degree(data.edge_index[0], dtype=torch.long)]
        max_degree = max( max_degree, degs[-1].max().item() )
        data.num_nodes = int( torch.max(data.edge_index) ) + 1

    if max_degree < 2000:
        # dataset.transform = T.OneHotDegree(max_degree)

        for data in dataset:
            degs = degree(data.edge_index[0], dtype=torch.long)
            data.x = F.one_hot(degs, num_classes=max_degree+1).to(torch.float)
    else:
        deg = torch.cat(degs, dim=0).to(torch.float)
        mean, std = deg.mean().item(), deg.std().item()
        for data in dataset:
            degs = degree(data.edge_index[0], dtype=torch.long)
            data.x = ( (degs - mean) / std ).view( -1, 1 )
    return dataset

