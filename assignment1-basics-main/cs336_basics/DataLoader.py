import torch
import torch
import numpy as np
def DataLoader(x, batch_size, context_length, device_string):
    num_possible_start_indices = len(x) - context_length

    index = np.random.choice(
        num_possible_start_indices,
        size=batch_size,
        replace=False
    )
    batchinput = []
    batchlabel = []

    for id in index:
        inputtuple = x[id : id + context_length]
        labeltuple = x[id + 1 : id + context_length + 1]
        batchinput.append(inputtuple)
        batchlabel.append(labeltuple)
    batchinput = np.array(batchinput)
    batchlabel = np.array(batchlabel)
    return (torch.tensor(batchinput, device=device_string, dtype=torch.long), torch.tensor(batchlabel, device=device_string, dtype=torch.long))