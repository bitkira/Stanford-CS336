import math
import torch
def GradientClipping(parms, max):
    parasum = 0
    for p in parms: 
        if p.grad is None: # <--- 重要检查！
            continue
        parasum = parasum + torch.pow(p.grad, 2).sum()
    parasum = torch.sqrt(parasum)
    if parasum < max:
        pass
    else:
        with torch.no_grad():
            for p in parms:
                if p.grad is None: # <--- 重要检查！
                    continue
                p.grad.mul_(max/(parasum + math.pow(10,-6)))