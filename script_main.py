# imports just wrote this for frun
from math import sqrt

import torch
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn.models import AttentiveFP

from src.features.build_features import Gen_Afp_Features
from torch_geometric.loader import DataLoader

# loading, and construction of featurized graphs
property_tag = "ESOL"  # define the dataset
feature_type = "AFP"  # define the type featurizer
dataset = MoleculeNet("data", property_tag, pre_transform=Gen_Afp_Features())
author_name = "arnaou"

# %% Splitting and wrapping the data in dataloader

split_seed = 420
split_frac = [70, 15]
batch_size = 64
N = len(dataset)
n_train = N // 100 * split_frac[0]
n_val = N // 100 * split_frac[1]
n_test = N - n_train - n_val
dataset = dataset.shuffle()

test_dataset = dataset[:n_test]
val_dataset = dataset[n_test: n_test + n_val]
train_dataset = dataset[n_test + n_val:]

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
# %% Model definition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# hyper_parameters for the afp model
in_channel = 39
hidden_channels = 200
out_channel = 1
edge_dim = 10
num_layers = 2
num_timesteps = 2
dropout = 0.0

model = AttentiveFP(
    in_channels=in_channel,
    hidden_channels=hidden_channels,
    out_channels=out_channel,
    edge_dim=edge_dim,
    num_layers=num_layers,
    num_timesteps=num_timesteps,
    dropout=dropout,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=10 ** -2.5, weight_decay=10 ** -5)


def train(train_loader):
    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_examples += data.num_graphs
    return sqrt(total_loss / total_examples)


@torch.no_grad()
def test(loader):
    mse = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        mse.append(F.mse_loss(out, data.y, reduction="none").cpu())
    return float(torch.cat(mse, dim=0).mean().sqrt())


for epoch in range(1, 201):
    train_rmse = train(train_loader)
    val_rmse = test(val_loader)
    test_rmse = test(test_loader)
    print(
        f"Epoch: {epoch:03d}, Loss: {train_rmse:.4f} Val: {val_rmse:.4f} "
        f"Test: {test_rmse:.4f}"
    )

optimizer = torch.optim.Adam(model.parameters(), lr=10 ** -2.5, weight_decay=10 ** -5)
