
import argparse
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR

from gmixup import GMixup
from data import load_data
from model import GIN, GCN, mixup_cross_entropy_loss


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

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # set optimizer, epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    num_epochs = 10

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in combined_train_data:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)

            targets = y = batch.y.view(-1, model.num_classes)

            # Compute loss (assumes labels are in batch.y)
            loss = mixup_cross_entropy_loss(out, targets)
            total_loss += loss.item()

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        avg_loss = total_loss / len(combined_train_data)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')


    # TODO: evaluate model
    # ...


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GIN', choices=['GIN', 'GCN'])
    parser.add_argument('--dataset', default='REDDIT-BINARY',
        choices=['REDDIT-BINARY', 'REDDIT-MULTI-5K', 'IMDB-BINARY', 'IMDB-MULTI'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--train-split-ratio', type=float, default=0.80)
    parser.add_argument('--data-cache-path', type=Path, default=Path('./'))
    parser.add_argument('--aug-ratio', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    main(args)
