import torch
import torch.nn.functional as F
from torch_geometric.datasets import LRGBDataset
from sklearn.metrics import r2_score
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from torch_geometric.nn import GCN
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.data import Data


class newGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels):
        super().__init__()
        self.gcn = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=hidden_channels,
                       num_layers=6, act='relu', dropout=0.1, norm='batch', norm_kwargs={'track_running_stats': False})
        # self.bn = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)
        # Add Laplacian eigenvector positional encoding transform
        self.linear = torch.nn.Linear(hidden_channels, out_channels)
        # Multi-layer prediction head
        self.prediction_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(hidden_channels, hidden_channels),
            # torch.nn.BatchNorm1d(hidden_channels, track_running_stats=False),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.2),

            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Preprocessing node features and edge attributes
        x = self.gcn(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = global_mean_pool(x, batch)
        # x = self.bn(x)
        # x = self.linear(x)
        x = self.prediction_head(x)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load Dataset
dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct')
train_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct', split='train')
val_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct', split='val')
test_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct', split='test')

# Initialize the GCN Model
model = newGCN(
    in_channels=dataset.num_node_features,
    hidden_channels=235,
    num_layers=6,
    out_channels=11  # Number of regression tasks
).to(device)
print(model)
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=20,
    min_lr=1e-5
)

torch.manual_seed(3)

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=200, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the training loop
criterion = torch.nn.L1Loss()  # For MAE-based regression


def compute_laplacian_pe(data, k_lap=8):
    transform = AddLaplacianEigenvectorPE(k=k_lap)
    return transform(data)


def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        # data = compute_laplacian_pe(data)
        data = data.to(device)
        optimizer.zero_grad()
        data.x = data.x.float()
        out = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        loss = criterion(out, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# Evaluation Function
def test(loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            # data = compute_laplacian_pe(data)
            data = data.to(device)
            data.x = data.x.float()
            out = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
            pred = out.cpu().numpy()
            labels = data.y.cpu().numpy()  # Squeeze to remove single-dimensional entries
            all_preds.append(pred)
            all_labels.append(labels)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds, multioutput='uniform_average')  # Average across all tasks
    return mae, r2


# Training
# In the main training loop
# Capture metrics during training


# Create figure and subplots before the training loop
plt.figure(figsize=(15, 10))

# SOTA GCN baseline value
sota_gcn_value = 0.2460

# Initialize empty lists for metrics
epochs = []
train_losses = []
train_maes = []
train_r2s = []
test_maes = []
test_r2s = []
val_maes = []
val_r2s = []

# Training loop with live plotting
for epoch in range(1, 251):
    # Clear the previous plots
    clear_output(wait=True)

    # Perform training and evaluation
    loss = train()
    val_mae, val_r2 = test(val_loader)
    test_mae, test_r2 = test(test_loader)
    train_mae, train_r2 = test(train_loader)

    # Step the scheduler
    scheduler.step(val_mae)

    # Print epoch information
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val MAE: {val_mae:.4f}, Val R2: {val_r2:.4f}, '
          f'Test MAE: {test_mae:.4f}, Test R2: {test_r2:.4f}')

    # Store metrics
    epochs.append(epoch)
    train_losses.append(loss)
    train_maes.append(train_mae)
    train_r2s.append(train_r2)
    test_maes.append(test_mae)
    test_r2s.append(test_r2)
    val_maes.append(val_mae)
    val_r2s.append(val_r2)

    # Create subplots
    plt.clf()  # Clear the entire current figure
    plt.figure(figsize=(15, 10))

    # Plot 1: Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, color='blue', label='Train Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot 2: MAE Comparison
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_maes, color='green', label='Train MAE')
    plt.plot(epochs, val_maes, color='yellow', label='Val MAE')
    plt.plot(epochs, test_maes, color='red', label='Test MAE')
    plt.axhline(y=sota_gcn_value, color='black', linestyle='--', label='SOTA TEST GCN')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    # Plot 3: R2 Score Comparison
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_r2s, color='purple', label='Train R2')
    plt.plot(epochs, val_r2s, color='cyan', label='Val R2')
    plt.plot(epochs, test_r2s, color='orange', label='Test R2')
    plt.title('R2 Score')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.legend()

    # Plot 4: Combined Metrics Normalized
    plt.subplot(2, 2, 4)
    # Normalize metrics to 0-1 range for comparison
    train_losses_norm = (np.array(train_losses) - np.min(train_losses)) / (np.max(train_losses) - np.min(train_losses))
    train_maes_norm = (np.array(train_maes) - np.min(train_maes)) / (np.max(train_maes) - np.min(train_maes))
    train_r2s_norm = (np.array(train_r2s) - np.min(train_r2s)) / (np.max(train_r2s) - np.min(train_r2s))

    plt.plot(epochs, train_losses_norm, color='blue', label='Normalized Train Loss')
    plt.plot(epochs, train_maes_norm, color='green', label='Normalized Train MAE')
    plt.plot(epochs, train_r2s_norm, color='purple', label='Normalized Train R2')
    plt.title('Normalized Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Value')
    plt.legend()

    plt.tight_layout()
    plt.pause(0.1)  # Small pause to update the plot

# Final plot after training completes
plt.show()