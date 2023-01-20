# Imports required libraries
import torch
import torch.nn as nn

class BCELoss(nn.Module):
    """
    Binary Cross Entropy Loss
    """
    def __init__(self):
        # Calls the constructor of the parent class
        super(BCELoss, self).__init__()

    # Defines the forward pass of the loss
    @staticmethod
    def forward(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        # Checks if the given arguments are tensors
        assert isinstance(y_pred, torch.Tensor), "y_pred must be a torch.Tensor"
        assert isinstance(y_true, torch.Tensor), "y_true must be a torch.Tensor"

        # Checks if the input tensors are on the same device
        assert y_pred.device == y_true.device, "y_pred and y_true must be on the same device"

        # Checks if the input tensors have the same shape
        assert y_pred.shape == y_true.shape, "y_pred and y_true must have the same shape"

        # Regulates the predicted values to be between 1E-7 and 1-1E-7
        y_pred = torch.clamp(y_pred, 1E-7, 1-1E-7)

        # Calculates the binary cross entropy loss
        loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

        # Returns the mean of the loss
        return loss.mean()

if __name__ == "__main__":
    # Defines the predicted and true values
    y_pred = torch.rand(4)
    y_true = torch.rand(4)

    # Calculates the mean absolute error
    bce = BCELoss.forward(y_pred, y_true)

    # Prints the mean absolute error
    print(f"Mean Absolute Error: {bce.item():.4f}")
