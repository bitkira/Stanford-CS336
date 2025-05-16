import torch.nn as nn
import torch
class embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.EmbeddingLayer = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(self.EmbeddingLayer, 0 , 1)
        self.EmbeddingLayer = nn.Parameter(self.EmbeddingLayer)
    def forward(self, token_ids):
        return self.EmbeddingLayer[token_ids]