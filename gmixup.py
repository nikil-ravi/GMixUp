import torch
import torch_geometric
from torch_geometric.data import Data
import numpy as np
# from skimage.restoration import denoise_tv_chambolle
import torch.nn.functional as F


def mixup_cross_entropy_loss(input, target, size_average=True):
    """
    Custom cross-entropy loss for Mixup with soft labels.
    
    Arguments:
    input -- Predicted probabilities from the model (after softmax or log softmax).
    target -- Soft labels as the ground truth, resulting from Mixup interpolation.
    size_average -- Whether to return the average loss per sample.

    Returns:
    the mixup cross-entropy loss, averaging over all samples if size_average is True.
    """
    print(input.size(), target.size())
    assert input.size() == target.size()    
    loss = -torch.sum(input * target)
    return loss / input.size(0) if size_average else loss


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
    
    def __call__(self, aug_ratio=0.5, num_samples=5, interpolation_lambda=0.1):
        """
        Generates synthetic graphs using the specified parameters.
        
        Parameters:
        aug_ratio -- Proportion of augmented data relative to the original dataset.
        num_samples -- Number of synthetic graphs to generate per pair of classes.
        interpolation_lambda -- Interpolation factor between graphons and labels.

        Returns:
        synthetic_graphs -- List of synthetic graphs with features and labels.
        """
        return self.generate(aug_ratio, num_samples, interpolation_lambda)


    def align_nodes(self, graph, original_features, criterion="degree"):
        """
        Aligns nodes of a graph based on a criterion (e.g., degree) and aligns node features.
        
        Parameters:
        graph -- PyTorch Geometric Data object.
        original_features -- Original node features.
        criterion -- Criterion for node alignment (default is "degree").
        
        Returns:
        Aligned adjacency matrix and aligned node features.
        """
        if criterion == "degree":
            edge_index = graph.edge_index
            num_nodes = graph.num_nodes

            # Compute node degrees
            node_degrees = torch.bincount(edge_index[0], minlength=num_nodes)

            # Sort nodes by degree
            sorted_indices = torch.argsort(node_degrees, descending=True)

            # Create adjacency matrix
            adjacency_matrix = torch.zeros((num_nodes, num_nodes))
            adjacency_matrix[edge_index[0], edge_index[1]] = 1

            # Align adjacency matrix and features
            aligned_matrix = adjacency_matrix[sorted_indices][:, sorted_indices]
            aligned_features = original_features[sorted_indices]

            return aligned_matrix.numpy(), aligned_features
        else:
            raise NotImplementedError(f"Criterion {criterion} not implemented")


    def estimate_graphon(self, graphs, features_list, K, method="partition"):
        """
        Estimates a graphon and its features from aligned graphs and node features.
        """
        aligned_adjacency_matrices = []
        aligned_features_list_all = []

        for graph, features in zip(graphs, features_list):
            aligned_matrix, aligned_features = self.align_nodes(graph, features)
            aligned_adjacency_matrices.append(aligned_matrix)
            aligned_features_list_all.append(aligned_features)

        if method == "partition":
            # Ensure consistent shapes
            N = aligned_adjacency_matrices[0].shape[0]
            
            # Create K partitions (as evenly as possible)
            partitions = np.array_split(np.arange(N), K)
            
            # Compute graphon_matrix by averaging edge densities in each partition pair
            graphon_matrix = np.zeros((K, K))
            for i in range(K):
                for j in range(K):
                    edge_density = [
                        np.mean(matrix[np.ix_(partitions[i], partitions[j])])
                        for matrix in aligned_adjacency_matrices
                    ]
                    graphon_matrix[i, j] = np.mean(edge_density)

            # Average features across multiple graphs if present
            aligned_features_list_all = np.stack(aligned_features_list_all)  # [num_graphs, N, F]
            aligned_features_avg = np.mean(aligned_features_list_all, axis=0)  # [N, F]
            
            # Compute KxF graphon_features by averaging features in each partition
            F = aligned_features_avg.shape[1]
            graphon_features = np.zeros((K, F))
            for i, part in enumerate(partitions):
                graphon_features[i, :] = np.mean(aligned_features_avg[part, :], axis=0)

            return graphon_matrix, graphon_features

        # elif method == "usvt":
        #     from graphon_estimator import universal_svd
        #     aligned_tensors = torch.stack([torch.tensor(matrix) for matrix in aligned_adjacency_matrices])
        #     mean_adj = torch.mean(aligned_tensors, dim=0)
        #     graphon = universal_svd(mean_adj)
        #     return graphon

        # elif method == "sorted_smooth":
        #     from graphon_estimator import sorted_smooth
        #     return sorted_smooth(aligned_adjacency_matrices, h=5)
        # else:
        #     raise ValueError(f"Unknown graphon estimation method: {method}")


    def generate(self, aug_ratio, num_samples, interpolation_lambda, num_classes=None):
        """
        Generates synthetic graphs with aligned node features for data augmentation.

        Parameters:
        aug_ratio -- Proportion of augmented data relative to the original dataset.
        num_samples -- Number of synthetic graphs to generate per pair of classes.
        interpolation_lambda -- Interpolation factor between graphons and labels.

        Returns:
        synthetic_graphs -- List of synthetic `Data` objects with features.
        """
        synthetic_graphs = []
        num_graphs = len(self.train_data)
        num_augmented = int(num_graphs * aug_ratio)

        while len(synthetic_graphs) < num_augmented:
            idx1, idx2 = np.random.choice(len(self.train_data), size=2, replace=False)
            graph1, graph2 = self.train_data[idx1], self.train_data[idx2]

            graphon1, features1 = self.estimate_graphon([graph1], [graph1.x], K=10)
            graphon2, features2 = self.estimate_graphon([graph2], [graph2.x], K=10)

            max_nodes = max(features1.shape[0], features2.shape[0])
            features1_padded = self.pad_features(features1, max_nodes)
            features2_padded = self.pad_features(features2, max_nodes)

            for _ in range(num_samples):
                mixed_graphon = self.graphon_mixup(graphon1, graphon2, interpolation_lambda)
                mixed_features = interpolation_lambda * features1_padded + (1 - interpolation_lambda) * features2_padded
                mixed_label = self.label_mixup(graph1.y, graph2.y, interpolation_lambda)

                synthetic_graph = self.generate_from_graphon(mixed_graphon, mixed_features)
                synthetic_graph.y = mixed_label
                synthetic_graphs.append(synthetic_graph)

                if len(synthetic_graphs) >= num_augmented:
                    break

        return synthetic_graphs


    def pad_features(self, features, target_size):
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float)
        num_nodes, num_features = features.shape
        if num_nodes >= target_size:
            return features

        padded_features = torch.zeros((target_size, num_features), dtype=features.dtype)
        padded_features[:num_nodes, :] = features
        return padded_features

    
    def graphon_mixup(self, graphon1: np.ndarray, graphon2: np.ndarray, interpolation_lambda: float) -> np.ndarray:
        """
        Takes two graphons and an interpolation hyperparameter lambda and interpolates them 
        to create a mixed graphon approximation.

        Parameters:
        graphon1 -- The first graphon (numpy array).
        graphon2 -- The second graphon (numpy array).
        interpolation_lambda -- The interpolation factor between 0 and 1.

        Returns:
        mixed_graphon -- The interpolated graphon (numpy array).
        """
        assert 0 <= interpolation_lambda <= 1, "lambda should be in the range [0, 1]"
        return interpolation_lambda * graphon1 + (1 - interpolation_lambda) * graphon2
    
    def label_mixup(self, label1, label2, interpolation_lambda):
        mixed_label = interpolation_lambda * label1 + (1 - interpolation_lambda) * label2
        return mixed_label



    def generate_from_graphon(self, graphon, graphon_features):
        """
        Generates a synthetic graph from a graphon matrix and assigns node features.
        
        Parameters:
        graphon -- Step function matrix representing the graphon.
        graphon_features -- Node features derived from graphon alignment and pooling.
        
        Returns:
        Data -- A PyTorch Geometric graph with node features and edges.
        """
        num_nodes = graphon.shape[0]
        adjacency_matrix = (np.random.rand(num_nodes, num_nodes) < graphon).astype(float)
        adjacency_matrix = np.triu(adjacency_matrix) + np.triu(adjacency_matrix, k=1).T

        edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)
        if not isinstance(graphon_features, torch.Tensor):
            graphon_features = torch.tensor(graphon_features, dtype=torch.float)

        synthetic_graph = Data(edge_index=edge_index)
        synthetic_graph.num_nodes = num_nodes
        synthetic_graph.x = graphon_features.clone().detach()
        return synthetic_graph
