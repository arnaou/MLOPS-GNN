import torch
import argparse
import logging
from torch import nn, optim
from src.models.model import GNNModel
import torch.nn.functional as F
import wandb
from math import sqrt
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

log = logging.getLogger(__name__)
model = GNNModel.model
criterion = torch.nn.NLLLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def predict(data):
    data = data.to(device)
    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
    mse = F.mse_loss(out, data.y, reduction='none').cpu()
    return float(mse.mean().sqrt())


def predict_loop():  
    with profile(schedule=torch.profiler.schedule(
        wait=5, # during this phase profiler is not active
        warmup=2, # during this phase profiler starts tracing, but the results are discarded
        active=6, # during this phase profiler traces and records data
        repeat=2),
        activities=[ProfilerActivity.CPU], 
        record_shapes=True, 
        on_trace_ready=tensorboard_trace_handler('./log/predict/')
        ) as prof:
        with record_function("model_inference"):  
 
            parser = argparse.ArgumentParser(description='Test Model')
            parser.add_argument('--config', default = 'config_default.yaml')
            
            args = parser.parse_args()
            
            config_path = 'src/configs/' + str(args.config)
            wandb.init(project = 'MLOPS-GNN', config = config_path)
            config = wandb.config
            wandb.watch(model)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            testpath    = config.testpath
            checkpoint  = config.checkpoint  
            test_loader = torch.load(testpath)
            
            model.load_state_dict(torch.load(checkpoint))     
            
            for data_idx, data in enumerate(test_loader):
                test_rmse = predict(data)
                
                wandb.log({"test_rmse": test_rmse})
                print(f'Data-set: {data_idx+1}',f'Test Error: {test_rmse:.4f}')
                prof.step()
                
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True, on_trace_ready=tensorboard_trace_handler('./log')) as prof:
                with record_function("model_inference"):
                    model(data.x, data.edge_index, data.edge_attr, data.batch)
                            
            wandb.finish()

if __name__ == "__main__":
    predict_loop()