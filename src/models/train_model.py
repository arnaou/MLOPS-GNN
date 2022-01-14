import torch
import argparse
import logging
from torch import nn, optim
from src.models.model import GNNModel
import torch.nn.functional as F
import wandb
from math import sqrt

log = logging.getLogger(__name__)
model = GNNModel.model
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=10 ** -2.5, weight_decay=10 ** -5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
def test(test_loader):
    mse = []
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        mse.append(F.mse_loss(out, data.y, reduction='none').cpu())
    return float(torch.cat(mse, dim=0).mean().sqrt())


def train_loop():    
 
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--config', default = 'config_default.yaml')
    
    args = parser.parse_args()
    
    config_path = 'src/configs/' + str(args.config)
    wandb.init(project = 'MLOPS-GNN', config = config_path)
    config = wandb.config
    wandb.watch(model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    epochs      = config.epochs
    lr          = config.learning_rate
    trainpath   = config.trainpath
    valpath     = config.valpath
    checkpoint  = config.checkpoint       

    log.info("Training day and night")
    train_loader    = torch.load(trainpath)
    val_loader      = torch.load(valpath)

    # Implement training loop here


    for epoch in range(1, epochs+1):
        train_rmse = train(train_loader)
        val_rmse = test(val_loader)
        
        wandb.log({"train_rmse": train_rmse})
        wandb.log({"val_rmse": val_rmse})
        print(f'Epoch: {epoch:03d}, Train Loss: {train_rmse:.4f} Val Loss: {val_rmse:.4f} ')

    torch.save(model.state_dict(),checkpoint)
    wandb.finish()

if __name__ == "__main__":
    train_loop()






