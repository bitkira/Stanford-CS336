import torch
import torch
import numpy as np
def DataLoader(x, batch_size, context_length, device_string):
    inputtuple = []
    labeltuple = []
    for i in range(len(x)-context_length):
        inputtuple.append(x[i:i+context_length+1][0:context_length])
        labeltuple.append(x[i:i+context_length+1][1:context_length+1])
    batchinput = []
    batchlabel = []

    index = np.random.choice(
        len(inputtuple),
        size=batch_size,
        replace=False
    )
    for id in index:
        batchinput.append(inputtuple[id])
        batchlabel.append(labeltuple[id])
    return (torch.tensor(batchinput, device=device_string), torch.tensor(batchlabel, device=device_string))