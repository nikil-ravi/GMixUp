import torch
from torch_geometric.data import Data
import numpy as np

class GMixup(torch_geometric.datasets.graph_generator.GraphGenerator):
    def __init__(self, train_x, train_y):
        """
        Sets the training data and labels to generate synthetic data for
        """
        pass

    def __call__(self) -> Data:
        """
        Override of GraphGenerator's __call__ method for sampling graphs
        """
        pass

    def estimate_graphon():
        """
        Takes a class of graphs, returns a graphon representation for that class
        """
        pass

    def graphon_mixup():
        """
        Takes two graphons and an interpolation hyperparameter lambda and interpolates them to create a mixed graphon that can be utilized to generate new graphs with new labels
        """
        pass

    def label_mixup():
        """
        Takes two labels and an interpolation hyperparameter lambda and interpolates them to produce a new label.
        """
        pass

    def generate(aug_ratio, num_samples):
        """
        Uses the (typically mixed) graphon and samples from it to generate 
        synthetic graphs that can serve as augmentated data.
        """
        pass