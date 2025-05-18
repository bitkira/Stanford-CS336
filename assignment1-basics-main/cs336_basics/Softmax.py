import torch
import torch.nn as nn
from einops import reduce

def softmax(x, i): 
    return torch.exp(x - torch.max(x, dim=i, keepdim=True)[0]) / torch.sum(torch.exp(x - torch.max(x, dim=i, keepdim=True)[0]), dim=1, keepdim=True)