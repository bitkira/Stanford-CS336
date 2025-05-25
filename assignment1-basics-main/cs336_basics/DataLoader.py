import torch
import torch
def DataLoader(x, batch_size, context_length, device_string):
    inputtuple = []
    labeltuple = []
    for i in range(len(x)-context_length-1):
        inputtuple.append(x[i:i+context_length+1][0:context_length])
        labeltuple.append(x[i:i+context_length+1][1:context_length+1])
    batchinput = []
    batchlabel = []
    for j in range(batch_size):
        batchinput.append(inputtuple[j])
        batchlabel.append(labeltuple[j])
    return (torch.tensor(batchinput, device=device_string), torch.tensor(batchlabel, device=device_string))