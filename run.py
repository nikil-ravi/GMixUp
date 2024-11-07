
import argparse
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader

from gmixup import GMixup
from data import load_data
from model import GIN, GCN


def main(args):

    # get data
    train, test = load_data(
        args.dataset,
        args.data_cache_path,
        args.train_split_ratio
    )

    # initialize GMixUp object
    gmixup = GMixup(train)

    # generate synthetic data
    synthetic = gmixup.generate(aug_ratio=0.5, num_samples=5)

    # mix real and synthetic data
    combined_train_data = DataLoader(
        [train, synthetic],
        batch_size=args.batch_size,
        shuffle=True
    )

    # train model
    if args.model == 'GIN':
        model = GIN
    elif args.model == 'GCN':
        model = GCN
    model = model(input_dim=train.num_features, output_dim=train.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 10

    for epoch in range(num_epochs):
        
        for batch in combined_train_data:
            # batch is type torch_geometric.data.Batch
            pass


    # evaluate model
    # ...


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GIN', choices=['GIN', 'GCN'])
    parser.add_argument('--dataset', default='REDDIT-BINARY',
        choices=['REDDIT-BINARY', 'REDDIT-MULTI-5K', 'IMDB-BINARY', 'IMDB-MULTI'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--train-split-ratio', type=float, default=0.80)
    parser.add_argument('--data-cache-path', type=Path, default=Path('./'))
    args = parser.parse_args()

    main(args)
