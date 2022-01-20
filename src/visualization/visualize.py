import pdb
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import wandb
from src.models.model import GNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, model_config_path):
#    model_path = "./models/checkpoint.pt"
#    model_config_path = "./models/model_config.yml"
    with open(model_config_path) as file:
        model_config = yaml.load(file, Loader=yaml.FullLoader)
    in_channels = model_config['in_channels']['value']
    hidden_channels = model_config['hidden_channels']['value']
    out_channels = model_config['out_channels']['value']
    edge_dim = model_config['edge_dim']['value']
    num_layers = model_config['num_layers']['value']
    num_timesteps = model_config['num_timesteps']['value']
    dropout = model_config['dropout']['value']

    model = GNNModel(in_channels=in_channels, hidden_channels=hidden_channels,
                     out_channels=out_channels, edge_dim=edge_dim, num_layers=num_layers,
                     num_timesteps=num_timesteps, dropout=dropout).model()
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    return model


def predict_with_model(data_loader, model):
    pred_list, target_list = [], []
    for data in data_loader:
        data = data.to(device)
        preds = model(data.x, data.edge_index, data.edge_attr, data.batch)
        targets = data.y
        pred_list.append(preds.detach().numpy())
        target_list.append(targets.detach().numpy())

    pred_list = np.vstack(pred_list)
    target_list = np.vstack(target_list)
    return pred_list, target_list


def plot_parity(train_targets, train_preds, val_targets, val_preds, plot_path):
    plt.plot(np.array(train_targets), np.array(train_preds), 'bo', label='training data')
    plt.plot(np.array(val_targets), np.array(val_preds), 'ro', label='validation data')
    plt.title("Parity Plot")
    plt.xlabel("experimental log(solubility [mol/L])")
    plt.ylabel("Predicted log(solubility [mol/L])")
    plt.savefig(plot_path)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument("--config", default="config_default.yaml")

    args = parser.parse_args()

    config_path = "models/model_config.yml"

    wandb.init(project="MLOPS-GNN", config=config_path)
    config = wandb.config

    epochs = config.epochs
    lr = config.learning_rate
    trainpath = config.trainpath
    valpath = config.valpath
    checkpoint = config.checkpoint
    plot_path = config.visualize_path

    in_channels = config.in_channels
    hidden_channels = config.hidden_channels
    out_channels = config.out_channels
    edge_dim = config.edge_dim
    num_layers = config.num_layers
    num_timesteps = config.num_timesteps
    dropout = config.dropout

    model = GNNModel(in_channels=in_channels, hidden_channels=hidden_channels,
                     out_channels=out_channels, edge_dim=edge_dim, num_layers=num_layers,
                     num_timesteps=num_timesteps, dropout=dropout).model()

    wandb.watch(model)

    train_loader = torch.load(trainpath)
    val_loader = torch.load(valpath)

    train_preds, train_targets = predict_with_model(train_loader, model)
    val_preds, val_targets = predict_with_model(val_loader, model)


    plot_parity(train_targets, train_preds, val_targets, val_preds, plot_path)



