import torch
import torch.nn as nn

class AnalyticalLinear(nn.Linear):
    """
    A linear layer that supports analytical weight updates via least squares
    and back-calculation of target inputs using pseudo-inverse.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        
    def get_input_target(self, y_target):
        """
        Back-calculates the target input for this layer given a target output.
        x_target = (y_target - b) @ pinv(W^T)
        """
        y_centered = y_target
        if self.bias is not None:
            y_centered = y_target - self.bias.data
            
        W_pinv = torch.linalg.pinv(self.weight.data.T)
        x_target = y_centered @ W_pinv
        return x_target
        
    def solve_and_update(self, x_actual, y_target, lr=1.0):
        """
        Solves for the optimal weights to map x_actual to y_target and applies 
        the update with the given learning rate.
        """
        if self.bias is not None:
            ones = torch.ones(x_actual.size(0), 1, dtype=x_actual.dtype, device=x_actual.device)
            x_aug = torch.cat([x_actual, ones], dim=1)
            
            W_aug_T = torch.linalg.lstsq(x_aug, y_target).solution
            W_opt = W_aug_T.T
            
            dW = W_opt[:, :-1] - self.weight.data
            db = W_opt[:, -1] - self.bias.data
            
            self.weight.data += lr * dW
            self.bias.data += lr * db
        else:
            W_T = torch.linalg.lstsq(x_actual, y_target).solution
            W_opt = W_T.T
            
            dW = W_opt - self.weight.data
            self.weight.data += lr * dW
