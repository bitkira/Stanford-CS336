import torch
import torch.nn as nn
from einops import reduce
class RMSnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        return (x * (1/(torch.sqrt(reduce(x*x, "... d_model -> ... 1", "mean") + self.eps))) * self.g).to(in_dtype)