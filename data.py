
from pathlib import Path
from typing import Tuple

from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


def load_data(
    dataset: str,
    data_cache_path: Path,
    train_split_ratio: float
) -> Tuple[Dataset, Dataset]:
    '''
    Return train, test
    '''
    ds = TUDataset(str(data_cache_path), dataset)
    print(f'Loaded dataset {dataset}: ', end='')
    ds.print_summary()
    train = ds.index_select(slice(int(len(ds)*train_split_ratio)))
    test = ds.index_select(slice(len(train),len(ds)))
    print(f'Train: {len(train)} graphs | Test: {len(test)} graphs')
    return train, test
    

def split_data(graphs, labels):
    # given graphs and labels, split them into training and testing sets
    pass




