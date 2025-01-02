import torch
from torch_geometric.utils import to_scipy_sparse_matrix
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
#from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from IPython.display import clear_output
from fast_pytorch_kmeans import KMeans
import json
import os




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
            self.model = KMeans(n_clusters=n_clusters)
            #self.model = KMeans(n_clusters=n_clusters, random_state=random_state,n_init=1)
        elif clustering_type == 'GMM':
            self.model = GaussianMixture(n_components=n_clusters, random_state=random_state)
        else:
            raise ValueError("Invalid clustering type. Choose 'KMeans' or 'GMM'.")

    def fit(self, features: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
                Fits the clustering model to the given features and returns cluster assignments.

                Args:
                    features (torch.Tensor): Node features (shape: [num_nodes, num_features]).

                Returns:
                    torch.Tensor: Cluster assignments for each node.
        """
        features_np = features.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()

        clusters = np.zeros(features_np.shape[0], dtype=int)
        for b in np.unique(batch_np):
            mask = batch_np == b
            if self.type == 'KMeans':
                features_tensor = torch.tensor(features_np[mask], dtype=torch.float32)
                #clusters[mask] = self.model.fit_predict(features_np[mask]) + clusters.max() + 1
                clusters[mask] = self.model.fit_predict(features_tensor) + clusters.max() + 1
            elif self.type == 'GMM':
                clusters[mask] = self.model.fit(features_np[mask]).predict(features_np[mask]) + clusters.max() + 1

        return torch.tensor(clusters, dtype=torch.long,device=features.device)


def coarsen_graph(cluster: torch.Tensor, data: Data, reduce: str = 'mean') -> Data:

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

    cluster, perm = torch.unique(cluster, return_inverse=True)

    x = scatter(data.x, perm, dim=0, reduce=reduce)
    edge_index = perm[data.edge_index]
    edge_attr = data.edge_attr

    # Remove self-loops and duplicate edges
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    batch = scatter(data.batch, perm, dim=0, reduce='max')

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)



class MLPGraphHead(torch.nn.Module):
    """
    MLP prediction head for graph prediction tasks.

    Args:
        hidden_channels (int): Input dimension.
        out_channels (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        self.pooling_fun = global_mean_pool
        dropout = 0.1
        L = 3

        layers = []
        for _ in range(L - 1):
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Linear(hidden_channels, hidden_channels, bias=True))
            layers.append(torch.nn.GELU())

        # layers.append(torch.nn.BatchNorm1d(hidden_channels, track_running_stats=False))
        layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(hidden_channels, out_channels, bias=True))
        self.mlp = torch.nn.Sequential(*layers)

    # def _scale_and_shift(self, x):
    # return x

    def forward(self,x,batch):
        x = self.pooling_fun(x,batch)

        return self.mlp(x)


# Define GCNConv layers

# Updated GCN model with coarsening
class GCNWithCoarsening(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, clustering_type='KMeans', n_clusters=5):
        super(GCNWithCoarsening, self).__init__()

        self.gcn_conv_layers = torch.nn.ModuleList([
            GCNConv(in_channels if i == 0 else hidden_channels, hidden_channels)
            for i in range(2)
        ])
        self.clustering = Clustering(clustering_type=clustering_type, n_clusters=n_clusters)
        self.coarsen_projection = torch.nn.Linear(hidden_channels, hidden_channels)
        self.gcn_post_coarsen = GCNConv(hidden_channels, hidden_channels)
        self.head = MLPGraphHead(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()

        for gcn in self.gcn_conv_layers:
            x = torch.relu(gcn(x, edge_index=edge_index))

        cluster = self.clustering.fit(x, batch)
        coarsened_data = coarsen_graph(cluster, Data(x=x, edge_index=edge_index, batch=batch))

        coarsened_data.x = self.coarsen_projection(coarsened_data.x)
        x = torch.relu(self.gcn_post_coarsen(coarsened_data.x, coarsened_data.edge_index))

        return self.head(x, coarsened_data.batch)


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


def visualize_graph_with_clusters(data, cluster):
    """
    Visualizes the graph with nodes colored based on their cluster assignments.

    Args:
        data (torch_geometric.data.Data): The input graph data.
        cluster (torch.Tensor): A tensor assigning each node to a cluster.
    """
    # Convert the PyTorch Geometric data to a NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Map cluster assignments to node colors
    cluster_colors = cluster.tolist()
    unique_clusters = list(set(cluster_colors))
    num_clusters = len(unique_clusters)
    # Generate a color map for clusters
    cmap = plt.cm.get_cmap('tab10', num_clusters)
    node_colors = [cmap(c) for c in cluster_colors]

    # Plot the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)  # Spring layout for visualization
    nx.draw(G, pos, node_color=cluster_colors, with_labels=True, cmap=plt.cm.tab10, node_size=300)
    # Add a legend for cluster colors
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(c), markersize=10, label=f'Cluster {c}')
        for c in unique_clusters
    ]
    plt.legend(handles=legend_handles, title="Clusters", loc='upper left', bbox_to_anchor=(1, 1))
    plt.title("Graph Visualization with Cluster Coloring")
    plt.show()




# Initialize the model
model = GCNWithCoarsening(in_channels=9, hidden_channels=235, out_channels=11, n_clusters=20)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(model)


# Learning rate scheduler
# Define the warmup and cosine decay schedule
def cosine_with_warmup(epoch):
    if epoch < warmup_epochs:
        # Linear warmup
        return epoch / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))



print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the training loop
criterion = torch.nn.L1Loss()  # For MAE-based regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        target = data.y.float().to(device)  # Ensure target is on the same device
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# Evaluation Function
def test(loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0  # Track loss for the scheduler

    with torch.no_grad():
        for data in loader:
            # data = compute_laplacian_pe(data)
            data = data.to(device)
            data.x = data.x.float()
            out = model(data)
            loss = criterion(out, data.y.float())  # Compute loss
            total_loss += loss.item()
            pred = out.cpu().numpy()
            labels = data.y.cpu().numpy()  # Squeeze to remove single-dimensional entries
            all_preds.append(pred)
            all_labels.append(labels)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds, multioutput='uniform_average')  # Average across all tasks
    return mae, r2, total_loss / len(loader)


# Training loop with logging and saving results
def train_with_logging(model, seeds, epochs, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    second_sota_gcn_value = 0.2460  # SOTA GCN baseline value
    first_sota_gcn_value = 0.3496

    for seed in seeds:
        torch.manual_seed(seed)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = LambdaLR(optimizer, lr_lambda=cosine_with_warmup)

        logs = []

        for epoch in range(1, epochs + 1):
            loss = train()
            val_mae, val_r2, val_loss = test(val_loader)
            test_mae, test_r2, _ = test(test_loader)
            train_mae, train_r2, _ = test(train_loader)

            scheduler.step()

            log_entry = {
                'epoch': epoch,
                'loss': loss,
                'val_mae': val_mae,
                'val_r2': val_r2,
                'test_mae': test_mae,
                'test_r2': test_r2
            }
            logs.append(log_entry)

            print(f"Seed {seed}, Epoch {epoch:03d}, Loss: {loss:.4f}, Val MAE: {val_mae:.4f}, Val R2: {val_r2:.4f}, Test MAE: {test_mae:.4f}, Test R2: {test_r2:.4f}")

        # Save logs to a JSON file
        log_file_path = os.path.join(log_dir, f'seed_{seed}_logs.json')
        with open(log_file_path, 'w') as f:
            json.dump(logs, f, indent=4)

        # Save the final plot
        plot_file_path = os.path.join(log_dir, f'seed_{seed}_plot.png')
        plt.figure(figsize=(10, 6))
        plt.plot([log['epoch'] for log in logs], [log['val_mae'] for log in logs], label='Validation MAE')
        plt.plot([log['epoch'] for log in logs], [log['test_mae'] for log in logs], label='Test MAE')
        plt.axhline(y=second_sota_gcn_value, color='black', linestyle='--', label='2nd SOTA GCN')
        plt.axhline(y=first_sota_gcn_value, color='black', linestyle='--', label='1st SOTA GCN')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title(f'Seed {seed} Training Progress')
        plt.legend()
        plt.savefig(plot_file_path)
        plt.close()

# Example usage
warmup_epochs = 5
total_epochs = 250

seeds = [42, 123, 2025, 5, 7]
log_directory = './training_logs'

train_with_logging(model, seeds, total_epochs, log_directory)