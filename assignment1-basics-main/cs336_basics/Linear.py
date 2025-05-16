import torch
import torch.nn as nn
from einops import einsum
class linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """Construct a linear transformation module.

        This function should accept the following parameters:
            in_features: int - final dimension of the input
            out_features: int - final dimension of the output
            device: torch.device | None = None - Device to store the parameters on
            dtype: torch.dtype | None = None - Data type of the parameters
        """
        super().__init__()
        self.W = torch.empty(out_features, in_features, dtype=dtype, device=device)
        torch.nn.init.trunc_normal_(self.W, 0 , 2/(out_features + in_features))
        self.W = nn.Parameter(self.W)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... infeature, outfeature infeature -> ... outfeature")