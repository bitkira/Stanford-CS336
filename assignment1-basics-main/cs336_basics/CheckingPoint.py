import torch
import os
import typing
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    train_dict={
    "model_dict":model.state_dict(),
    "optimizer_dict":optimizer.state_dict(),
    "iteration":iteration}
    torch.save(train_dict, out)
    
def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    train_dict = torch.load(src)
    model.load_state_dict(train_dict["model_dict"])
    optimizer.load_state_dict(train_dict["optimizer_dict"])
    return train_dict["iteration"]