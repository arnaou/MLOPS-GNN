import torch
from torch_geometric.nn.models import AttentiveFP




class GNNModel(torch.nn.Module):
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
