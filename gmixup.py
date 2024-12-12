import torch
import torch_geometric
from torch_geometric.data import Data
import numpy as np
from skimage.restoration import denoise_tv_chambolle
import torch.nn.functional as F


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

def sorted_smooth(aligned_graphs: list, h: int) -> np.ndarray:
    """
    Implement the sorted_smooth method. This first averages the aligned graphs 
    and then applies a block-averaging via a convolutional operation.
    Finally, it applies total variation denoising to produce a smooth graphon estimate.
    """
    # Convert aligned graphs to a tensor
    N = max(a.shape[0] for a in aligned_graphs)
    aligned_graphs = [GMixup.pad_adjacency(None, a, N) for a in aligned_graphs] # B list of shape (N, N)
    num_graphs = len(aligned_graphs)
    if num_graphs > 1:
        # mean over all graphs, shape: (1,1,N,N) after unsqueeze
        sum_graph = (sum(aligned_graphs) / num_graphs).unsqueeze(0).unsqueeze(0) # torch.mean(aligned_graphs, dim=0, keepdim=True).unsqueeze(0)
    else:
        sum_graph = aligned_graphs.unsqueeze(0).unsqueeze(0)  # (1,1,N,N)
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
        class_graphs = {}
        for graph in train_data:
            if graph.y.numel() == 1:
                raise RuntimeError(
                    f''
                )
            label = tuple(graph.y.squeeze().tolist())
            if label not in class_graphs:
                class_graphs[label] = []
            class_graphs[label].append(graph)

        self.class_graphons = {}
        self.class_features = {}
        for label, graphs in class_graphs.items():
            features_list = [graph.x for graph in graphs]
            graphon, features = self.estimate_graphon(graphs, features_list)
            self.class_graphons[label] = graphon
            self.class_features[label] = features

        # Pad graphons to the same size
        num_nodes = max(f.shape[0] for f in self.class_features.values())
        for label, graphon in self.class_graphons.items():
            self.class_graphons[label] = self.pad_adjacency(graphon, num_nodes)
        for label, features in self.class_features.items():
            self.class_features[label] = self.pad_features(features, num_nodes)

        # Create mapping of idx -> label for fast class sampling
        self.label_lookup = { i: l for i, l in enumerate(self.class_graphons.keys()) }

    
    def __call__(self, interpolation_range: list):
        """
        Generates synthetic graphs using the specified parameters.
        
        Parameters:
        aug_ratio -- Proportion of augmented data relative to the original dataset.
        num_samples -- Number of synthetic graphs to generate per pair of classes.
        interpolation_range -- Tuple of low and high values for interpolation

        Returns:
        synthetic_graphs -- List of synthetic graphs with features and labels.
        """
        interp_lambda = np.random.uniform(low=interpolation_range[0], high=interpolation_range[1])
        return self.sample(interp_lambda)


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


    def estimate_graphon(self, graphs, features_list, K = 10, method="sorted_smooth"):
        """
        Takes a set of graphs, returns an approximation of the graphon for that set of graphs.
        Uses one of two methods:
        - "partition" Partition nodes into K intervals and estimate edge probability between each pair of intervals
        - "usvt" SVD-based threhsolding method
        - "sorted_smooth" Sorted smooth method
        """

        aligned_adjacency_matrices = []
        aligned_features_list_all = []

        for graph, features in zip(graphs, features_list):
            aligned_matrix, aligned_features = self.align_nodes(graph, features)
            aligned_adjacency_matrices.append(aligned_matrix)
            aligned_features_list_all.append(aligned_features)

        # Pad features to the maximum size before stacking
        max_nodes = max(features.shape[0] for features in aligned_features_list_all)
        aligned_features_list_all = [self.pad_features(features, max_nodes) for features in aligned_features_list_all]
        aligned_features_list_all = np.stack(aligned_features_list_all)  # [num_graphs, N, F]
        graphon_features = np.mean(aligned_features_list_all, axis=0)  # [N, F]

        if method == "partition":
            # Ensure consistent shapes
            N = max(m.shape[0] for m in aligned_adjacency_matrices)
            
            # Create K partitions (as evenly as possible)
            partitions = np.array_split(np.arange(N), K)

            aligned_adjacency_matrices = [self.pad_adjacency(m, N) for m in aligned_adjacency_matrices]
            
            # Compute graphon by averaging edge densities in each partition pair
            graphon = np.zeros((K, K))
            for i in range(K):
                for j in range(K):
                    edge_density = [
                        np.mean(matrix[np.ix_(partitions[i], partitions[j])])
                        for matrix in aligned_adjacency_matrices
                    ]
                    graphon[i, j] = np.mean(edge_density)
            

            return graphon, graphon_features
        elif method == "usvt":
            N = max(m.shape[0] for m in aligned_adjacency_matrices)
            aligned_adjacency_matrices = [self.pad_adjacency(m, N) for m in aligned_adjacency_matrices]
            aligned_tensors = aligned_adjacency_matrices # torch.from_numpy(np.stack(aligned_adjacency_matrices))
            if len(aligned_tensors) > 1:
                mean_adj = sum(aligned_tensors) / len(aligned_tensors) # torch.mean(aligned_tensors, dim=0)
            else:
                mean_adj = aligned_tensors[0, :, :]
            
            # apply SVD-based thresholding
            graphon = universal_svd(mean_adj, threshold=2.02)
            #graphon = cv2.resize(graphon, interpolation=cv2.INTER_LINEAR)
            return graphon, graphon_features
        elif method == "sorted_smooth":
            graphon = sorted_smooth(aligned_adjacency_matrices, h=5)
            return graphon, graphon_features
        else:
            raise ValueError(f"Unknown graphon estimation method: {method}")


    def generate(self, num_samples, interpolation_range: tuple):
        """
        Generates synthetic graphs with aligned node features for data augmentation.

        Parameters:
        num_samples -- Number of synthetic graphs to generate.
        interpolation_range -- a low and high value for interpolation

        Returns:
        synthetic_graphs -- List of synthetic `Data` objects with features.
        """
        synthetic_graphs = []
        low = interpolation_range[0]
        high = interpolation_range[1]
        for _ in range(num_samples):
            interp_lambda = np.random.uniform(low=low, high=high)
            graph = self.sample(interp_lambda)
            synthetic_graphs.append(graph)
        return synthetic_graphs
    

    def sample(self, interpolation_lambda: float):
        """
        Generates one synthetic graph with aligned node features for data augmentation.

        Parameters:
        interpolation_lambda -- Interpolation factor between graphons and labels.

        Returns:
        synthetic_graph -- Synthetic `Data` objects with features.
        """
        class1, class2 = np.random.choice(len(self.class_graphons), size=2, replace=False)
        label1, label2 = self.label_lookup[class1], self.label_lookup[class2]
        graphon1, features1 = self.class_graphons[label1], self.class_features[label1]
        graphon2, features2 = self.class_graphons[label2], self.class_features[label2]

        mixed_graphon = self.graphon_mixup(graphon1, graphon2, interpolation_lambda)
        mixed_features = self.graphon_mixup(features1, features2, interpolation_lambda)
        label1 = torch.tensor(label1).to(mixed_graphon.device)
        label2 = torch.tensor(label2).to(mixed_graphon.device)
        mixed_label = self.label_mixup(label1, label2, interpolation_lambda)
        
        graph = self.generate_from_graphon(mixed_graphon, mixed_features)
        graph.y = mixed_label.unsqueeze(0)
        #graph.annotation = f'mixed labels {label1.tolist()} and {label2.tolist()} with lambda={interpolation_lambda:.2f}'
        return graph
    
    
    

    def pad_features(self, features, target_size):
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float)
        num_nodes, num_features = features.shape
        if num_nodes >= target_size:
            return features

        padded_features = torch.zeros((target_size, num_features), dtype=features.dtype)
        padded_features[:num_nodes, :] = features
        return padded_features
    
    def pad_adjacency(self, adj, target_size):
        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj, dtype=torch.float)
        num_nodes = adj.shape[-1]
        padded_adj = torch.zeros((target_size,target_size), dtype=adj.dtype)
        padded_adj[:num_nodes,:num_nodes] = adj
        return padded_adj


    
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
        adjacency_matrix = (torch.rand(num_nodes, num_nodes) < graphon).to(torch.float32)
        adjacency_matrix = torch.triu(adjacency_matrix)
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T - torch.diag(torch.diag(adjacency_matrix))

        
        edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long).T
        if not isinstance(graphon_features, torch.Tensor):
            graphon_features = torch.tensor(graphon_features, dtype=torch.float)

        synthetic_graph = Data(edge_index=edge_index)
        synthetic_graph.num_nodes = num_nodes
        synthetic_graph.x = graphon_features.clone().detach()
        
        return synthetic_graph
