# define path for the model
path = r'C:\Users\arnaou\OneDrive - Danmarks Tekniske Universitet\02 My courses\MLOPS-GNN\models\checkpoint.pth'


import torch

from src.models.model import GNNModel

model = GNNModel.model
model.load_state_dict(torch.load(path))

from src.features.build_features import process_smiles

data = process_smiles('CCCCCC')
out = model(data.x, data.edge_index, data.edge_attr, torch.tensor([0]))
