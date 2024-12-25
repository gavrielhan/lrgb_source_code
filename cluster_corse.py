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
from IPython.display import clear_output
from torch_geometric.data import Data, Batch
from torch.optim.lr_scheduler import LambdaLR
from typing import Callable, Optional
from torch import Tensor
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_batch, pool_edge, pool_pos
from torch_geometric.utils import scatter




# Load Dataset
dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct')
train_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct', split='train')
val_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct', split='val')
test_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct', split='test')




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

