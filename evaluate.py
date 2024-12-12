
import argparse
from pathlib import Path
from itertools import product
import numpy as np

from run import main



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Eval
    parser.add_argument('--models', nargs='*', default=['GIN', 'GCN'])
    parser.add_argument('--datasets', nargs='*', default=['REDDIT-BINARY', 'REDDIT-MULTI-5K', 'IMDB-BINARY', 'IMDB-MULTI'])
    parser.add_argument('--num-runs', type=int, default=10)
    # General
    parser.add_argument('--data-cache-path', type=Path, default=Path('./datasets/'))
    parser.add_argument('--data-split', type=float, nargs=2, default=(7.,3.),
        help="ratio of train-test data (doesn't have to sum to 1)")
    parser.add_argument('--seed', type=int, default=42)
    # Training
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.01)
    # GMixup
    parser.add_argument('--vanilla', dest='use_mixup', action='store_false')
    parser.add_argument('--aug-ratio', type=float, default=0.5)
    parser.add_argument('--interpolation-range', nargs=2, type=float, default=(0.1,0.2))
    args = parser.parse_args()


    # model -> dataset -> list[runs]
    results = { m: { d: [] for d in args.datasets } for m in args.models }

    iter = product(
        args.models,
        args.datasets,
        list(range(args.seed, args.seed+args.num_runs))
    )

    for model, dataset, seed in iter:
        print('Running:')
        print('\tmodel:', model)
        print('\tdataset:', dataset)
        print('\tseed:', seed)
        
        args.model = model
        args.dataset = dataset
        args.seed = seed

        results[model][dataset].append(main(args))


    print('=== Results ===')
    for model, datasets in results.items():
        print(f'  {model}:')
        for dataset, accuracies in datasets.items():
            dataset += ':' + ' '*(max(map(len, datasets.keys()))-len(dataset))
            mean = np.mean(accuracies)
            print(f'    {dataset}\t{100*mean:.3f}%\t\t[{", ".join(f"{a:.5f}" for a in accuracies)}]')
