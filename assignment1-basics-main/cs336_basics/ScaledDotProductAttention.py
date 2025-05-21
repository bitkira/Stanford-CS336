import torch
import torch.nn as nn
from einops import rearrange
import sys
sys.path.append("/Users/bitkira/Documents/GitHub/Stanford-CS336/assignment1-basics-main/")
from cs336_basics.Softmax import softmax


def scaled_dot_product_attention(q, k, v, mask=None):
    k = rearrange(k, "... seq_len d_k -> ... d_k seq_len")
    out = torch.matmul(q, k) / torch.sqrt(torch.tensor(q.shape[-1], dtype=q.dtype))
    if mask is not None:
        out.masked_fill_(~mask,float('-inf'))
    return torch.matmul(softmax(out, i=-1), v)
  