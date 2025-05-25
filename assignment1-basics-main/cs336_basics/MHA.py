import torch
import torch.nn as nn
import math
from einops import rearrange
from einops import einsum

import sys
sys.path.append("/Users/bitkira/Documents/GitHub/Stanford-CS336/assignment1-basics-main/")
from cs336_basics.Linear import linear
from cs336_basics.Softmax import softmax
from cs336_basics.ScaledDotProductAttention import scaled_dot_product_attention
from cs336_basics.RoPE import rope

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, device, max_seq_len=None, theta=None, token_positions=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.device = device
        self.num_heads = num_heads
        self.d_model = d_model
        self.Q = linear(d_model, d_model)
        self.K = linear(d_model, d_model)
        self.V = linear(d_model, d_model)
        self.O = linear(d_model, d_model)
        if theta is not None:
            self.rope = rope(theta, d_model//num_heads, max_seq_len, device=device)
            self.token_pos = token_positions
        else:
            self.rope = None
    def forward(self, x):
        mask = torch.tril(torch.ones(x.shape[1], x.shape[1], dtype=int,  device = self.device)).bool()
        Q = rearrange(self.Q(x), "batch_size seq_len (h dk) -> batch_size h seq_len dk", h=self.num_heads, dk=self.d_model//self.num_heads)
        K = rearrange(self.K(x), "batch_size seq_len (h dk) -> batch_size h seq_len dk", h=self.num_heads, dk=self.d_model//self.num_heads)
        if self.rope is not None:
            Q = self.rope(Q, self.token_pos)
            K = self.rope(K, self.token_pos)
        V = rearrange(self.V(x), "batch_size seq_len (h dv) -> batch_size h seq_len dv", h=self.num_heads, dv=self.d_model//self.num_heads)

        attention_score = scaled_dot_product_attention(Q, K, V, mask=mask)
        attention_score = rearrange(attention_score, "batch_size h seq_len dv -> batch_size seq_len (h dv)")
        return self.O(attention_score)
