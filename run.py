import torch
from gmixup import GMixUp
from data import load_data, split_data

from model import GIN

def main():

    # get data
    graphs, labels = load_data() # TODO: implement load_data function (Jared)
    train_graphs, train_labels, test_graphs, test_labels = split_data(graphs, labels) # TODO: Jared

    # initialize GMixUp object
    gmixup = GMixUp(train_graphs, train_labels)

    # generate synthetic data
    synthetic_graphs = gmixup.generate(aug_ratio=0.5, num_samples=5)

    combined_dataset = train_graphs + synthetic_graphs
    # combined_labels = train_labels + synthetic_labels

    # train model
    model = GIN(input_dim=None, output_dim=None) # TODO: fill in input_dim and output_dim
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 10

    for epoch in range(num_epochs):
        # ...
        # TODO: implement training loop
        # ...
        break

    # evaluate model
    # ...

if __name__ == '__main__':
    main()