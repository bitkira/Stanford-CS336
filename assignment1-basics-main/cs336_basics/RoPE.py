import torch
import torch.nn as nn
import math
from einops import reduce

class rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        R = torch.empty(d_k, d_k)
        R[0::2, 0::2] = math.pow(theta, )
        R[1::2] = 
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor: