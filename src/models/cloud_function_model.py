import torch
import torchvision
from model import GNNModel
import torch.nn.functional as F


def predict(data, use_model):
    if use_model == 'scripted':
        out = scripted_model(data.x, data.edge_index, data.edge_attr, data.batch)
    else:
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
    mse = F.mse_loss(out, data.y, reduction="none").cpu()
    return float(mse.mean().sqrt())

model = GNNModel()
model.load_state_dict(torch.load('models/checkpoint.pth'))
scripted_model = torch.jit.script(model)
scripted_model.save('models/cloud_function_model.pt')

testloader = torch.load('data/esol/processed/test_loader.pt')

data = next(iter(testloader))
test_rmse = predict(data, 'unscripted')
test_rmse_scripted = predict(data, 'scripted')
print(test_rmse,test_rmse_scripted)
assert torch.allclose(test_rmse,test_rmse_scripted)