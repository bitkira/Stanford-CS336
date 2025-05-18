import torch
import torch.nn as nn
import sys
sys.path.append("/Users/bitkira/Documents/GitHub/Stanford-CS336/assignment1-basics-main/")
from cs336_basics.Linear import linear

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = linear(d_model, d_ff)
        self.w2 = linear(d_ff, d_model)
        self.w3 = linear(d_model, d_ff)
    def forward(self, x):
        return self.w2((self.w1(x) * torch.sigmoid(self.w1(x))) * self.w3(x))

        