# src/utils/checkpoint.py

import torch

def load_checkpoint(filepath, model, optimizer=None, device='cuda'):
    """
    Loads model and optimizer state from a checkpoint file.

    Args:
    - filepath (str): Path to the checkpoint file.
    - model (torch.nn.Module): The model to load the state into.
    - optimizer (torch.optim.Optimizer, optional): The optimizer to load the state into.
    - device (str): Device to map the model and optimizer to.

    Returns:
    - model: Model with loaded state.
    - optimizer: Optimizer with loaded state (if provided).
    - epoch: The epoch at which the checkpoint was saved.
    - loss: The loss value at which the checkpoint was saved.
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss
