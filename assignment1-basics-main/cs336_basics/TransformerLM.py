import torch
import torch.nn as nn
import sys
sys.path.append("/Users/bitkira/Documents/GitHub/Stanford-CS336/assignment1-basics-main/")
from cs336_basics.RMSnorm import RMSnorm
from cs336_basics.Linear import linear
from cs336_basics.Softmax import softmax
from cs336_basics.FullTransformerBlock import TransformerBlock
from cs336_basics.Embedding import embedding

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, token_pos, vocab_size, context_length, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        embedding_layer = embedding(vocab_size, d_model)
        self.layers.append(embedding_layer)
        for i in range(num_layers):
            Block = TransformerBlock(d_model, num_heads, d_ff, context_length, theta, token_pos)
            self.layers.append(Block)
        self.norm_layer = RMSnorm(d_model)
        self.linear_layer = linear(d_model, vocab_size)
        self.layers.append(self.norm_layer)
        self.layers.append(self.linear_layer)
    def forward(self, x):
        for i in self.layers:
            x = i(x)
        return x