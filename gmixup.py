import torch
import torch_geometric
from torch_geometric.data import Data
import numpy as np

class GMixup(torch_geometric.datasets.graph_generator.GraphGenerator):
    def __init__(self, train_data):
        """
        Sets the training data to generate synthetic data from.
        
        Parameters:
        - train_data (torch_geometric.data.Dataset): Training dataset containing graphs and labels.
        """
        super().__init__()
        self.train_data = train_data
        self.train_data_x = [graph.x for graph in train_data]
        self.train_data_y = [graph.y for graph in train_data]


    def __call__(self, graph_indices=None, K=10, interpolation_lambda=None) -> Data:
        """
        Samples and returns a mixed synthetic graph using specified or random graphs from the training data.
        
        Parameters:
        - graph_indices (list of int, optional): Indices of two graphs from the training data to use for mixup.
        - K (int, optional): Number of partitions for the graphon estimation (default is 10).
        - interpolation_lambda (float, optional): The interpolation weight between graphon1 and graphon2 (between 0 and 1).
          If None, a random value between 0 and 1 will be generated.

        Returns:
        - Data: A PyG Data object containing a synthetic graph and its label.
        """
        # Choose two random graphs if graph_indices is not provided
        if graph_indices is None:
            idx1, idx2 = np.random.choice(len(self.train_data), size=2, replace=False)
        else:
            idx1, idx2 = graph_indices
        
        graph1, graph2 = self.train_data[idx1], self.train_data[idx2]

        # Estimate graphons for each graph
        graphon1 = self.estimate_graphon([graph1], K)
        graphon2 = self.estimate_graphon([graph2], K)
        
        # Set lambda for interpolation
        if interpolation_lambda is None:
            interpolation_lambda = np.random.rand()  # Random value between 0 and 1

        # Create a mixed graphon
        mixed_graphon = self.graphon_mixup(graphon1, graphon2, interpolation_lambda)
        
        # Generate a synthetic graph from the mixed graphon
        synthetic_graph = self.generate_from_graphon(mixed_graphon)

        # Mix labels
        label1 = graph1.y.item()
        label2 = graph2.y.item()
        synthetic_label = self.label_mixup(label1, label2, interpolation_lambda)
        
        # Create and return the synthetic Data object
        synthetic_data = Data(x=synthetic_graph.x, edge_index=synthetic_graph.edge_index, y=synthetic_label)
        return synthetic_data

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

    def generate(self, num_samples, K=10, interpolation_lambda=None):
        """
        Generates synthetic graphs for data augmentation.
        
        Parameters:
        - num_samples (int): Number of synthetic samples to generate.
        - K (int, optional): Number of partitions for graphon estimation (default is 10).
        - interpolation_lambda (float, optional): Lambda value for mixing; if None, random values will be generated.

        Returns:
        - list[Data]: List of generated synthetic data.
        """
        synthetic_data = [self(K=K, interpolation_lambda=interpolation_lambda) for _ in range(num_samples)]
        return synthetic_data
    
    def generate_from_graphon(self, graphon: np.ndarray) -> Data:
        """
        Generates a synthetic graph from a graphon matrix.
        """
        num_nodes = graphon.shape[0]
        x = torch.eye(num_nodes)  # Identity matrix as initial node features

        # Create an edge list based on probabilities in the graphon matrix
        edge_index = []
        for i in range(num_nodes):
            for j in range(i, num_nodes):  # upper triangular to avoid double counting
                if np.random.rand() < graphon[i, j]:  # Probability threshold
                    edge_index.append([i, j])
                    if i != j:  # Add reverse for undirected graph
                        edge_index.append([j, i])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)
