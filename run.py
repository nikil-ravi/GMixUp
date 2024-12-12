
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR

from gmixup import GMixup, mixup_cross_entropy_loss
from data import *
from model import GIN, GCN


def main(args):
    torch.manual_seed(args.seed)

    # get data
    dataset = load_data(args.dataset, args.data_cache_path)
    dataset = create_node_features(dataset)
    dataset = make_labels_one_hot(dataset)
    train, test = split_data(dataset, *args.data_split)
    num_features = dataset.num_features
    num_classes = dataset.num_classes


    if args.use_mixup:

        gmixup = GMixup(train)
        synthetic = gmixup.generate(
            aug_ratio=0.5,
            num_samples=5,
            interpolation_lambda=args.interpolation_lambda,
        )
        combined_graphs = train + synthetic

        for i, g in enumerate(combined_graphs):
            # assert shape of y is (1, num_classes)
            assert g.y.shape == (1, num_classes)

        dataloader = DataLoader(
            combined_graphs,
            batch_size=args.batch_size,
            shuffle=True
        )
        def loss_fn(preds, target):
            preds = F.log_softmax(preds, dim=1)
            return mixup_cross_entropy_loss(preds, target)

    else:
        dataloader = DataLoader(
            train,
            batch_size=args.batch_size,
            shuffle=True
        )
        loss_fn = F.cross_entropy


    # train model
    if args.model == 'GIN':
        model = GIN
    elif args.model == 'GCN':
        model = GCN
    else:
        raise ValueError(f'Unknown model: {model}')
    model = model(input_dim=num_features, output_dim=num_classes)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).train()

    # set optimizer, epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

    # main training loop
    for epoch in range(args.epochs):
        
        total_loss, total_samples = 0, 0
        for batch in dataloader:
            batch = batch.to(device)
            out = model.forward(batch.x, batch.edge_index, batch.batch)
            
            # Compute loss (assumes labels are in batch.y)
            loss = loss_fn(out, batch.y)
            total_loss += len(batch) * loss.item()
            total_samples += len(batch)

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        
        avg_loss = total_loss / total_samples
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}')


    return evaluate(model, DataLoader(test, batch_size=args.batch_size), device)



def evaluate(model: nn.Module, test: DataLoader, device):
    model.eval()
    total_correct, total_samples = 0, 0
    print('Evaluating ...')
    for batch in test:
        batch = batch.to(device)
        out = model.predict(batch.x, batch.edge_index, batch.batch)

        labels = torch.argmax(batch.y, dim=1)  # Convert one-hot to class indices
        total_correct += (out == labels).sum()
        total_samples += len(batch)
    
    acc = total_correct / total_samples
    print(f'Test Accuracy: {100*acc:.3f}%')
    return acc.cpu().item()




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--model', default='GIN', choices=['GIN', 'GCN'])
    parser.add_argument('--dataset', default='REDDIT-BINARY',
        choices=['REDDIT-BINARY', 'REDDIT-MULTI-5K', 'IMDB-BINARY', 'IMDB-MULTI'])
    parser.add_argument('--data-cache-path', type=Path, default=Path('./datasets/'))
    parser.add_argument('--data-split', type=float, nargs=2, default=(7.,3.),
        help="ratio of train-test data (doesn't have to sum to 1)")
    parser.add_argument('--seed', type=int, default=42)
    # Training
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=0.01)
    # GMixup
    parser.add_argument('--vanilla', dest='use_mixup', action='store_false')
    parser.add_argument('--aug-ratio', type=float, default=0.5)
    parser.add_argument('--interpolation-lambda', type=float, default=0.1)
    args = parser.parse_args()

    main(args)
