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

    def align_nodes(self, graph, criterion="degree"):
        """
        Aligns nodes of a graph based on a criterion (e.g. degree, centrality, etc.)
        """
        if criterion == "degree":
            node_degrees = graph.degree()
            sorted_indices = np.argsort(node_degrees)
            aligned_matrix = graph.adjacency_matrix[sorted_indices, :][:, sorted_indices]
            return aligned_matrix
        else:
            raise NotImplementedError(f"Criterion {criterion} not implemented")


    def estimate_graphon(self, graphs, K):
        """
        Takes a set of graphs, returns an approximation of the graphon for that set of graphs.
        """
        
        # align adjacency matrices
        aligned_adjacency_matrices = []
        for graph in graphs:
            aligned_matrix = self.align_nodes(graph)
            aligned_adjacency_matrices.append(aligned_matrix)

        # initialize step function matrix
        step_function_matrix = np.zeros((K, K))

        # partition nodes into K intervals; each interval will have N/K nodes
        N = aligned_adjacency_matrices[0].shape[0] # assumes all graphs have the same number of nodes
        partition_size = N // K
        
        partitions = [list(range(i, min(i + partition_size, N))) for i in range(0, N, partition_size)]


        # for each partition pair, estimate edge probability, which is the average edge density between the two partitions
        for i in range(K):
            for j in range(K):
                total_edge_density = 0
                for adj_matrix in aligned_adjacency_matrices:
                    partition_i = partitions[i]
                    partition_j = partitions[j]
                    step_function_matrix[i, j] += np.mean(adj_matrix[partition_i, :][:, partition_j])
                step_function_matrix[i, j] = total_edge_density / len(aligned_adjacency_matrices)

        return step_function_matrix

    def graphon_mixup(self, graphon1: np.ndarray, graphon2: np.ndarray, interpolation_lambda: float) -> np.ndarray:
        """
        Takes two graphons (or approximations) and an interpolation hyperparameter lambda and interpolates them to create a mixed 
        graphon approximate that can be utilized to generate new graphs with new labels

        graphon1: numpy.ndarray
            The first graphon to be mixed
        graphon2: numpy.ndarray
            The second graphon to be mixed
        interpolation_lambda: float
            The interpolation hyperparameter lambda (between 0 and 1) that determines the weight of the first graphon in the mix
        """

        assert 0 <= interpolation_lambda <= 1, "lambda should be in the range [0, 1]"
        return interpolation_lambda * graphon1 + (1 - interpolation_lambda) * graphon2
        

    def label_mixup(self, label1: int, label2: int, interpolation_lambda: float) -> int:
        """
        Takes two labels and an interpolation hyperparameter lambda and interpolates them to create a mixed label

        label1: int
            The first label to be mixed
        label2: int
            The second label to be mixed
        interpolation_lambda: float
            The interpolation hyperparameter lambda (between 0 and 1) that determines the weight of the first label in the mix
        """

        assert 0 <= interpolation_lambda <= 1, "lambda should be in the range [0, 1]"
        return int(interpolation_lambda * label1 + (1 - interpolation_lambda) * label2)

    def generate(aug_ratio, num_samples):
        """
        Uses the (typically mixed) graphon and samples from it to generate 
        synthetic graphs that can serve as augmentated data.
        """
        pass