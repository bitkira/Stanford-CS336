import torch
import torch.nn as nn
import math
from einops import reduce

class rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        THETA = [math.pow(theta, (2*k)/d_k) for k in range(d_k/2)]
        Index = []
        for i in range(max_seq_len):
            Index.append(i / THETA)
        
        R_list = []
        for j in Index:
            R = torch.empty(d_k, d_k)
            R[0::2, 0::2] = torch.diag(torch.cos(j/THETA))  
            R[0::2, 1::2] = torch.diag(-torch.sin(j/THETA))   
            R[1::2, 0::2] = torch.diag(torch.sin(j/THETA)) 
            R[1::2, 1::2] = torch.diag(torch.cos(j/THETA))
            R_list.append(R)
        self.register_buffer("RoPE" ,torch.tensor(R),persistent=False)
        torch.tensor(R).register_buffer(persistent=False)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor: