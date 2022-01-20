import argparse
import logging
from math import sqrt

import torch
import torch.nn.functional as F
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    tensorboard_trace_handler,
)

import wandb
from src.models.model import GNNModel

log = logging.getLogger(__name__)
model = GNNModel.model
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=10 ** -2.5, weight_decay=10 ** -5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        mse.append(F.mse_loss(out, data.y, reduction="none").cpu())
    return float(torch.cat(mse, dim=0).mean().sqrt())


def train_loop():

    with profile(
        schedule=torch.profiler.schedule(
            wait=5,  # during this phase profiler is not active
            warmup=2,  # during this phase profiler starts tracing, but the results are discarded
            active=6,  # during this phase profiler traces and records data
            repeat=2,
        ),
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        on_trace_ready=tensorboard_trace_handler("./log/train/"),
    ) as prof:
        with record_function("model_inference"):

            for epoch in range(1, epochs + 1):
                train_rmse = train(train_loader)
                val_rmse = test(val_loader)

                wandb.log({"train_rmse": train_rmse})
                wandb.log({"val_rmse": val_rmse})
                print(
                    f"Epoch: {epoch:03d}, Train Loss: {train_rmse:.4f} Val Loss: {val_rmse:.4f} "
                )
                prof.step()

            torch.save(model.state_dict(), checkpoint)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument("--config", default="config_default.yaml")

    args = parser.parse_args()

    config_path = "src/configs/" + str(args.config)
    wandb.init(project="MLOPS-GNN", config=config_path)
    config = wandb.config
    wandb.watch(model)

    epochs = config.epochs
    lr = config.learning_rate
    trainpath = config.trainpath
    valpath = config.valpath
    checkpoint = config.checkpoint

    log.info("Training day and night")
    train_loader = torch.load(trainpath)
    val_loader = torch.load(valpath)

    train_loop()
    wandb.finish()
