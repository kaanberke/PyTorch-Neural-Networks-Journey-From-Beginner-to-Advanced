import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    @staticmethod
    def forward(y_pred, y_true):
        loss = (y_pred - y_true) ** 2
        return loss.mean()
