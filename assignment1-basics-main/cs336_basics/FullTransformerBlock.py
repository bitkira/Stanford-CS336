import torch
import torch.nn as nn
import sys
sys.path.append("/Users/bitkira/Documents/GitHub/Stanford-CS336/assignment1-basics-main/")
from cs336_basics.RMSnorm import RMSnorm
from cs336_basics.SwiGLU import SwiGLU
from cs336_basics.MHA import MultiheadSelfAttention
import torch
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, token_pos, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        pos = torch.tensor([i for i in range(token_pos)], dtype=int)
        self.norm1 = RMSnorm(d_model)
        self.norm2 = RMSnorm(d_model)
        self.MHA_layer = MultiheadSelfAttention(d_model, num_heads, device, max_seq_len,theta, token_positions=pos)
        self.SwiGLU = SwiGLU(d_model, d_ff)
    
    def forward(self, x):
        sub_result1 = x + self.MHA_layer(self.norm1(x))
        sub_result2 = sub_result1 + self.SwiGLU(self.norm2(sub_result1))
        return sub_result2
