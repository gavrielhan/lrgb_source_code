import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import LRGBDataset
from sklearn.metrics import r2_score
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from torch_geometric.nn import GCN


class newGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels):
        super().__init__()
        self.gcn = GCN(in_channels=in_channels, hidden_channels=hidden_channels,out_channels=hidden_channels, num_layers=5)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Preprocessing node features and edge attributes
        x = self.gcn(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = global_mean_pool(x,batch)
        x = self.linear(x)
        return x


# Load Dataset
dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct')
train_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct', split='train')
val_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct', split='val')
test_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct', split='test')


# Initialize the GCN Model
model = newGCN(
    in_channels=dataset.num_node_features,
    hidden_channels=300,
    num_layers=5,
    out_channels=11  # Number of regression tasks
)
print(model)
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

torch.manual_seed(3)

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=200, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the training loop
criterion = torch.nn.L1Loss()  # For MAE-based regression


def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data.x = data.x.float()
        out = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        loss = criterion(out, data.y.float())
        loss.backward()
        optimizer.zero_grad()
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


# Training loop
for epoch in range(1, 101):
    loss = train()
    val_mae, val_r2 = test(val_loader)
    test_mae, test_r2 = test(test_loader)
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val MAE: {val_mae:.4f}, Val R2: {val_r2:.4f}, '
              f'Test MAE: {test_mae:.4f}, Test R2: {test_r2:.4f}')