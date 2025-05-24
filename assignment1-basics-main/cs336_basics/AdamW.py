import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        default = {"lr":lr, "beta1":betas[0], "beta2":betas[1], "eps":eps, "lamada":weight_decay}
        super().__init__(params, default)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            lamada = group["lamada"]
            for p in group["params"]:
                if p.grad is None: 
                    continue
                state = self.state[p] 
                t = state.get("t", 1)
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p))
                state = self.state[p]
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                state["m"] = m
                v = beta2 * v + (1 - beta2) * torch.pow(grad,2)
                state["v"] = v
                original_lr = lr
                lr = lr * (math.sqrt(1-math.pow(beta2, t))/(1 - math.pow(beta1, t)))
                p.data = p.data - lr*(m/(torch.sqrt(v)+eps))
                p.data = p.data - p.data*lamada*original_lr
                state["t"] = t+1
        return loss


               