# Import necessary libraries
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import LRGBDataset
from sklearn.metrics import r2_score
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn.models import GCN

# Define the GCN Model
"""class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout=0.5):
        super(GCN, self).__init__()
        # Define GCN layers
        self.convs = torch.nn.ModuleList()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden_channels))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Final layer
        self.fc = torch.nn.Linear(hidden_channels, out_channels)  # Output for regression
        self.dropout = dropout


    def forward(self, x, edge_index, batch):
        h = self.model_(x, edge_index)
        out = h.relu()  # self.activation(h)
        out = global_mean_pool(out, batch)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.out_layer(out)
        return out"""



# Load Dataset
train_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct', split='train')
val_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct', split='val')
test_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct', split='test')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the GCN Model
model = GCN(
    in_channels=-1,  # Feature size
    hidden_channels=32,
    num_layers=5,
    out_channels=11,  # Number of regression tasks
    dropout=0.5
)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

torch.manual_seed(3)  # 0, 1, 2, 3, 4

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')


train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the training loop
criterion = torch.nn.L2Loss()  # For MAE-based regression
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data.x = data.x.float()  # Convert to float for integer features
        # data = data.to(model.device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
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
            #data = data.to(model.device)
            out = model(data.x, data.edge_index, data.edge_attr, batch=data.batch)
            pred = out.cpu().numpy()
            labels = data.y.cpu().numpy()
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
    if epoch % 10 == 0:
        val_mae, val_r2 = test(val_loader)
        test_mae, test_r2 = test(test_loader)
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val MAE: {val_mae:.4f}, Val R2: {val_r2:.4f}, '
              f'Test MAE: {test_mae:.4f}, Test R2: {test_r2:.4f}')