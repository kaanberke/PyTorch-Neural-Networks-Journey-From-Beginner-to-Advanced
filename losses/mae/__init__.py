# Imports required libraries
import torch
import torch.nn as nn

# Defines the Mean Absolute Error loss
class MAELoss(nn.Module):
    def __init__(self):
        # Calls the constructor of the parent class
        super(MAELoss, self).__init__()

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

        # Calculates the absolute difference between the predicted and true values
        loss = (y_pred - y_true).abs()

        # Returns the mean of the loss
        return loss.mean()

if __name__ == "__main__":
    # Defines the predicted and true values
    y_pred = torch.rand(4)
    y_true = torch.rand(4)

    # Calculates the mean absolute error
    mae = MAELoss.forward(y_pred, y_true)

    # Prints the mean absolute error
    print(f"Mean Absolute Error: {mae.item():.4f}")
