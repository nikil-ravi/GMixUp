from gmixup import GMixUp


def main():

    # get data
    graphs, labels = load_data()
    train_graphs, train_labels, test_graphs, test_labels = split_data(graphs, labels)

    # initialize GMixUp object
    gmixup = GMixUp(train_graphs, train_labels)

    # generate synthetic data
    synthetic_graphs = gmixup.generate(aug_ratio=0.5, num_samples=5)


if __name__ == '__main__':
    main()