import torch
import torch_geometric
from torch_geometric.data import Data
import numpy as np

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
    assert input.size() == target.size()    
    loss = -torch.sum(input * target)
    
    # Returns the average loss if specified, otherwise returns total loss
    return loss / input.size()[0] if size_average else loss


def universal_svd(adj_matrix: torch.Tensor, threshold: float = 2.02) -> np.ndarray:
    """
    Estimate a graphon by universal singular value thresholding.
    """
    num_nodes = adj_matrix.size(0)
    u, s, v = torch.svd(adj_matrix)
    singular_threshold = threshold * (num_nodes ** 0.5)

    # Zero out singular values below threshold
    s[s < singular_threshold] = 0
    graphon = u @ torch.diag(s) @ v.t()

    # Clip to [0, 1]
    graphon = torch.clamp(graphon, min=0, max=1)
    return graphon.numpy()

def sorted_smooth(self, aligned_graphs: list, h: int) -> np.ndarray:
    """
    Implement the sorted_smooth method. This first averages the aligned graphs 
    and then applies a block-averaging via a convolutional operation.
    Finally, it applies total variation denoising to produce a smooth graphon estimate.
    """
    # Convert aligned graphs to a tensor
    aligned_graphs_t = self.graph_numpy2tensor(aligned_graphs)  # shape (B, N, N)
    num_graphs = aligned_graphs_t.size(0)

    if num_graphs > 1:
        # mean over all graphs, shape: (1,1,N,N) after unsqueeze
        sum_graph = torch.mean(aligned_graphs_t, dim=0, keepdim=True).unsqueeze(0)
    else:
        sum_graph = aligned_graphs_t.unsqueeze(0).unsqueeze(0)  # (1,1,N,N)

    # create a uniform kernel for block-averaging
    kernel = torch.ones(1, 1, h, h) / (h ** 2)
    
    # apply convolution with stride = h
    graphon = F.conv2d(sum_graph, kernel, padding=0, stride=h, bias=None)
    graphon = graphon[0, 0, :, :].numpy()
    
    # apply TV denoising (https://www.ipol.im/pub/art/2013/61/article.pdf)
    graphon = denoise_tv_chambolle(graphon, weight=h)
    
    # clip values to [0,1]
    graphon = np.clip(graphon, 0, 1)
    return graphon


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


    def __call__():
        """
        Samples and returns a mixed synthetic graph using specified or random graphs from the training data.
        """
        pass

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



    def estimate_graphon(self, graphs, features_list, K, method = "partition"):
        """
        Takes a set of graphs, returns an approximation of the graphon for that set of graphs.

        Uses one of two methods:
        - "partition" Partition nodes into K intervals and estimate edge probability between each pair of intervals
        - "usvt" SVD-based threhsolding method
        - "sorted_smooth" Sorted smooth method
        """
        
        # align adjacency matrices
        aligned_adjacency_matrices = []
        aligned_features_list = []

        for graph, features in zip(graphs, features_list):
            aligned_matrix, aligned_features = self.align_nodes(graph, features)
            aligned_adjacency_matrices.append(aligned_matrix)
            aligned_features_list.append(aligned_features)

        if method == "partition":
            # Compute step function graphon matrix
            N = aligned_adjacency_matrices[0].shape[0]
            partition_size = N // K
            partitions = [list(range(i, min(i + partition_size, N))) for i in range(0, N, partition_size)]

            graphon_matrix = np.zeros((K, K))
            for i in range(K):
                for j in range(K):
                    edge_density = [
                        np.mean(matrix[partitions[i], :][:, partitions[j]])
                        for matrix in aligned_adjacency_matrices
                    ]
                    graphon_matrix[i, j] = np.mean(edge_density)

            # Pool node features to compute graphon node features
            aligned_features_stacked = np.stack(aligned_features_list)
            graphon_features = np.mean(aligned_features_stacked, axis=0)  # Average pooling

            return graphon_matrix, graphon_features
        elif method == "usvt":

            aligned_tensors = self.graph_numpy2tensor(aligned_adjacency_matrices)
            if aligned_tensors.size(0) > 1:
                mean_adj = torch.mean(aligned_tensors, dim=0)
            else:
                mean_adj = aligned_tensors[0, :, :]
            
            # apply SVD-based thresholding
            graphon = self.universal_svd(mean_adj, threshold=2.02)
            return graphon

        elif method == "sorted_smooth":

            return sorted_smooth(aligned_adjacency_matrices, h=5)


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

    def generate(self, aug_ratio, num_samples, interpolation_lambda):
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
            # Randomly sample two graphs
            idx1, idx2 = np.random.choice(len(self.train_data), size=2, replace=False)
            graph1, graph2 = self.train_data[idx1], self.train_data[idx2]

            # Estimate graphons and node features
            graphon1, features1 = self.estimate_graphon([graph1], [graph1.x], K=10)
            graphon2, features2 = self.estimate_graphon([graph2], [graph2.x], K=10)

            # Align node features by padding
            max_nodes = max(features1.shape[0], features2.shape[0])
            features1_padded = self.pad_features(features1, max_nodes)
            features2_padded = self.pad_features(features2, max_nodes)

            for _ in range(num_samples):
                # Mix graphons and labels
                mixed_graphon = self.graphon_mixup(graphon1, graphon2, interpolation_lambda)
                mixed_features = interpolation_lambda * features1_padded + (1 - interpolation_lambda) * features2_padded
                mixed_label = self.label_mixup(graph1.y.item(), graph2.y.item(), interpolation_lambda)

                # Generate synthetic graph
                synthetic_graph = self.generate_from_graphon(mixed_graphon, mixed_features)
                synthetic_graph.y = torch.tensor([mixed_label], dtype=torch.float)

                synthetic_graphs.append(synthetic_graph)

                # Break if we reach the desired number of augmented graphs
                if len(synthetic_graphs) >= num_augmented:
                    break

        return synthetic_graphs
    
    def pad_features(self, features, target_size):
        """
        Pads a feature matrix to match the target size.

        Parameters:
        features -- Node feature matrix (num_nodes x num_features).
        target_size -- Target number of nodes.

        Returns:
        Padded feature matrix (target_size x num_features).
        """
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float)
        num_nodes, num_features = features.shape
        if num_nodes >= target_size:
            return features

        # Create a zero-padded matrix
        padded_features = torch.zeros((target_size, num_features), dtype=features.dtype)
        padded_features[:num_nodes, :] = features
        return padded_features


    
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

        # Generate adjacency matrix based on the graphon
        adjacency_matrix = (np.random.rand(num_nodes, num_nodes) < graphon).astype(float)
        adjacency_matrix = np.triu(adjacency_matrix) + np.triu(adjacency_matrix, k=1).T

        # Create edge index
        edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)

        # Create graph with node features
        synthetic_graph = Data(edge_index=edge_index)
        synthetic_graph.num_nodes = num_nodes  # Explicitly set num_nodes
        synthetic_graph.x = torch.tensor(graphon_features, dtype=torch.float)  # Assign graphon features to nodes

        return synthetic_graph

