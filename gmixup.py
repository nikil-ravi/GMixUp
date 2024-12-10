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

        Uses one of two methods:
        - "partition" Partition nodes into K intervals and estimate edge probability between each pair of intervals
        - "usvt" SVD-based threhsolding method
        - "sorted_smooth" Sorted smooth method
        """
        
        # align adjacency matrices
        aligned_adjacency_matrices = []
        for graph in graphs:
            aligned_matrix = self.align_nodes(graph)
            aligned_adjacency_matrices.append(aligned_matrix)

        if method == "partition":
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
        Generates synthetic graphs for data augmentation.
        
        Parameters:
        aug_ratio -- Proportion of augmented data relative to the original dataset.
        num_samples -- Number of synthetic samples to generate per pair of classes.
        interpolation_lambda -- Interpolation factor between graphons and labels.
        
        Returns:
        synthetic_graphs -- List of synthetic `Data` objects with labels in `y`.
        """
        synthetic_graphs = []
        num_graphs = len(self.train_data)
        num_augmented = int(num_graphs * aug_ratio)

        for _ in range(num_augmented):
            # Randomly sample two graphs
            idx1, idx2 = np.random.choice(len(self.train_data), size=2, replace=False)
            graph1, graph2 = self.train_data[idx1], self.train_data[idx2]

            # Estimate graphons
            graphon1 = self.estimate_graphon([graph1], K=10)
            graphon2 = self.estimate_graphon([graph2], K=10)

            # Mix graphons and labels
            mixed_graphon = self.graphon_mixup(graphon1, graphon2, interpolation_lambda)
            mixed_label = self.label_mixup(graph1.y.item(), graph2.y.item(), interpolation_lambda)

            # Generate synthetic graph
            synthetic_graph = self.generate_from_graphon(mixed_graphon)
            synthetic_graph.y = torch.tensor([mixed_label], dtype=torch.float)  # Ensure label is set

            synthetic_graphs.append(synthetic_graph)

        return synthetic_graphs
    
    def generate_from_graphon(self, graphon):
        """
        Generates a synthetic graph from a graphon matrix.
        
        Parameters:
        graphon -- Matrix representing the graphon approximation.
        """
        num_nodes = graphon.shape[0]
        adjacency_matrix = (np.random.rand(num_nodes, num_nodes) < graphon).astype(float)
        adjacency_matrix = np.triu(adjacency_matrix) + np.triu(adjacency_matrix, k=1).T

        edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)
        return Data(edge_index=edge_index)
