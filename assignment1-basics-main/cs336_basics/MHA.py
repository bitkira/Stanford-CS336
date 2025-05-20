import torch
import torch.nn as nn
import math
from einops import reduce
from einops import einsum

import sys
sys.path.append("/Users/bitkira/Documents/GitHub/Stanford-CS336/assignment1-basics-main/")
from cs336_basics.Linear import linear
from cs336_basics.Softmax import softmax
from cs336_basics.ScaledDotProductAttention import scaled_dot_product_attention

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.Q = linear(d_model, d_model)
        self.K = linear(d_model, d_model)
        self.V = linear(d_model, d_model)
    def forward(self, x):
        scaled_dot_product_attention(self.Q(x), self.K(x), self.V(x),mask=)
