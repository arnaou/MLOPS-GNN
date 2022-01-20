import torch
from torch_geometric.nn.models import AttentiveFP


class GNNModel:
    def __init__(self, in_channels=39, hidden_channels=150, out_channels=1, edge_dim=10,
                 num_layers=2, num_timesteps=2, dropout=0.0):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        self.device = self.get_device()

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model(self):
        return AttentiveFP(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            edge_dim=self.edge_dim,
            num_layers=self.num_layers,
            num_timesteps=self.num_timesteps,
            dropout=self.dropout,
        ).to(self.device)
