from torch_geometric.transforms import AddLaplacianEigenvectorPE
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import LRGBDataset
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from torch_geometric.nn import GCN
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from torch.optim.lr_scheduler import LambdaLR
from typing import Callable, Optional
from torch import Tensor
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_batch, pool_edge, pool_pos
from torch_geometric.utils import scatter
import networkx as nx
from torch_geometric.utils import to_networkx
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture




# Load Dataset
dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct')
train_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct', split='train')
val_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct', split='val')
test_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct', split='test')

class Clustering:
    def __init__(self, clustering_type: str, n_clusters: int, random_state: int = 0):
        """
        Initializes the clustering class.

        Args:
            clustering_type (str): The type of clustering ('KMeans' or 'GMM').
            n_clusters (int): The number of clusters to create.
            random_state (int): Seed for reproducibility. Default: 0.
        """
        self.type = clustering_type
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None

        if clustering_type == 'KMeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        elif clustering_type == 'GMM':
            self.model = GaussianMixture(n_components=n_clusters, random_state=random_state)
        else:
            raise ValueError("Invalid clustering type. Choose 'KMeans' or 'GMM'.")

    def fit(self, features: torch.Tensor) -> torch.Tensor:
        """
        Fits the clustering model to the given features and returns cluster assignments.

        Args:
            features (torch.Tensor): Node features (shape: [num_nodes, num_features]).

        Returns:
            torch.Tensor: Cluster assignments for each node.
        """
        features_np = features.cpu().numpy()  # Convert to NumPy for sklearn
        if self.type == 'KMeans':
            clusters = self.model.fit_predict(features_np)
        elif self.type == 'GMM':
            clusters = self.model.fit(features_np).predict(features_np)
        return torch.tensor(clusters, dtype=torch.long)  # Convert back to PyTorch tensor



def _aggregate_features(
    cluster: Tensor,
    x: Tensor,
    size: Optional[int] = None,
    reduce: str = 'mean',
) -> Tensor:
    """
    Aggregates node features within clusters using the specified reduction operation.
    """
    return scatter(x, cluster, dim=0, dim_size=size, reduce=reduce)


def coarsen_graph(
    cluster: Tensor,
    data: Data,
    transform: Optional[Callable] = None,
    reduce: str = 'mean',
) -> Data:
    """
    Coarsens and pools a graph based on clustering information.

    Args:
        cluster (Tensor): Tensor assigning each node to a specific cluster.
        data (Data): Input graph data object.
        transform (Callable, optional): Transformation function to apply on the coarsened graph. Default: None.
        reduce (str): Aggregation method for node features ('mean', 'max', 'sum', etc.). Default: 'mean'.

    Returns:
        Data: Coarsened graph data object.
    """
    # Ensure cluster assignments are consecutive
    cluster, perm = consecutive_cluster(cluster)

    # Aggregate node features
    x = None if data.x is None else _aggregate_features(cluster, data.x, reduce=reduce)

    # Pool edges and edge attributes
    edge_index = cluster[data.edge_index.reshape(-1)].reshape(2, -1)
    edge_index, edge_attr = pool_edge(cluster, edge_index, data.edge_attr)

    # Pool batch information if present
    batch = None if data.batch is None else pool_batch(perm, data.batch)

    # Pool node positions if present
    pos = None if data.pos is None else pool_pos(cluster, data.pos)

    # Create a new coarsened Data object
    coarsened_data = Batch(batch=batch, x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

    # Apply optional transformation
    if transform is not None:
        coarsened_data = transform(coarsened_data)

    return coarsened_data



def visualize_graph(data, title="Graph", cluster_assignments=None):
    """
    Visualizes a graph using networkx and matplotlib.

    Args:
        data (torch_geometric.data.Data): Graph data object.
        title (str): Title of the plot.
        cluster_assignments (torch.Tensor, optional): Tensor of cluster assignments for nodes.
    """
    # Convert PyTorch Geometric Data to NetworkX Graph
    G = to_networkx(data, to_undirected=True)

    # Get node positions for better visualization
    pos = nx.spring_layout(G, seed=42)  # Seed for consistent layout

    # Draw nodes with cluster colors if provided
    if cluster_assignments is not None:
        color_map = [cluster_assignments[node] for node in range(data.num_nodes)]
        nx.draw(G, pos, with_labels=True, node_color=color_map, cmap=plt.cm.Set3, node_size=10)
    else:
        nx.draw(G, pos, with_labels=True, node_size=10)

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.8)

    plt.title(title)
    plt.show()

