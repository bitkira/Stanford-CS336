from einops import rearrange, reduce
import torch
import sys
import torch.nn.functional as F

sys.path.append("/Users/bitkira/Documents/GitHub/Stanford-CS336/assignment1-basics-main/")
from cs336_basics.Softmax import softmax

def CrossEntropy(logits ,targts):
    num_classes = logits.shape[-1]
    targts = F.one_hot(targts, num_classes).bool()
    expsum = reduce(torch.exp(logits - ), "logits -> 1", "mean")
    logits = logits[targts]
    logits = logits - 
    return reduce(logits, "logits -> 1", "mean")