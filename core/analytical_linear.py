import torch
import torch.nn as nn

class AnalyticalLinear(nn.Linear):
    """
    A linear layer that supports analytical weight updates via least squares
    and back-calculation of target inputs using pseudo-inverse.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.cached_input = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        # Cache the input for the analytical update calculation
        self.cached_input = x.clone().detach()
        return super().forward(x)
        
    def calculate_updates_and_target(self, y_target):
        """
        Given the target output for this layer, calculates dW, db 
        and back-calculates the target input for the previous layer.
        
        Args:
            y_target: (batch_size, out_features)
            
        Returns:
            x_target: (batch_size, in_features)
        """
        if self.cached_input is None:
            raise RuntimeError("Must run forward pass before calculating updates.")
            
        x = self.cached_input  # Shape: (batch_size, in_features)
        y = y_target           # Shape: (batch_size, out_features)
        
        # 1. Solve for optimal weights: x @ W^T + b = y
        if self.bias is not None:
            # Augment x with 1s to solve for bias simultaneously
            ones = torch.ones(x.size(0), 1, dtype=x.dtype, device=x.device)
            x_aug = torch.cat([x, ones], dim=1)  # (batch_size, in_features + 1)
            
            # Solve x_aug @ W_aug^T = y
            # Using lstsq: AX = B -> x_aug @ W_aug^T = y -> W_aug^T = lstsq(x_aug, y)
            W_aug_T = torch.linalg.lstsq(x_aug, y).solution
            W_opt = W_aug_T.T  # (out_features, in_features + 1)
            
            W_new = W_opt[:, :-1]
            b_new = W_opt[:, -1]
            
            self.dW = W_new - self.weight.data
            self.db = b_new - self.bias.data
        else:
            # Solve x @ W^T = y
            W_T = torch.linalg.lstsq(x, y).solution
            W_opt = W_T.T
            
            self.dW = W_opt - self.weight.data
            self.db = None
            
        # 2. Back-calculate target input for the previous layer
        # x_target @ W^T + b = y_target => x_target @ W^T = y_target - b
        # x_target = (y_target - b) @ pinv(W^T)
        
        y_centered = y_target
        if self.bias is not None:
            y_centered = y_target - self.bias.data
            
        # W^T shape: (in_features, out_features)
        # pinv(W^T) shape: (out_features, in_features)
        W_pinv = torch.linalg.pinv(self.weight.data.T)
        
        x_target = y_centered @ W_pinv  # (batch_size, in_features)
        
        return x_target
        
    def apply_updates(self, lr=1.0):
        """
        Applies the calculated analytical update with a learning rate.
        """
        if self.dW is not None:
            self.weight.data += lr * self.dW
            self.dW = None
        if self.db is not None and self.bias is not None:
            self.bias.data += lr * self.db
            self.db = None
        self.cached_input = None
