import torch
import torch.nn as nn
import math
from einops import reduce
from einops import einsum
import numpy as np
class rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        THETA = torch.tensor([math.pow(theta, (2*k)/d_k) for k in range(int(d_k/2))])

        R_list = []
        for j in range(max_seq_len):
            R = np.zeros((d_k, d_k), dtype=np.float32)
            R[0::2, 0::2] = np.diag(torch.cos(j/THETA))  
            R[0::2, 1::2] = np.diag(-torch.sin(j/THETA))   
            R[1::2, 0::2] = np.diag(torch.sin(j/THETA)) 
            R[1::2, 1::2] = np.diag(torch.cos(j/THETA))
            R_list.append(R)
        self.register_buffer("RoPE" ,torch.tensor(R_list, dtype=torch.float32, device=device),persistent=False)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        self.PosRoPE = self.RoPE[0:x.shape[-2], :, :]
        return einsum(x, self.PosRoPE[token_positions], "... sequence_length d_k, ... sequence_length dk d_k-> ... sequence_length dk")