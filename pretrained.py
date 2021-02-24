import torch
import torchvision

# change to your model
net = torchvision.models.resnet18(pretrained=True)

state_dict = {k: v.data if isinstance(v, torch.nn.Parameter) else v for k, v in net.state_dict().items()}
torch.save(state_dict, "net.pth")
