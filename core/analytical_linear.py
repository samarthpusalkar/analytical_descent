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
            
        try:
            # rcond helps ignore tiny singular values that cause instability
            W_pinv = torch.linalg.pinv(self.weight.data.T, rcond=1e-5)
        except RuntimeError:
            # Fallback if SVD fails to converge: W^+ = (W^T W + eps I)^-1 W^T
            W_T = self.weight.data.T
            eps = 1e-4
            I = torch.eye(W_T.size(1), device=W_T.device, dtype=W_T.dtype)
            W_pinv = torch.linalg.inv(W_T.T @ W_T + eps * I) @ W_T.T
            
        x_target = y_centered @ W_pinv
        return x_target
        
    def _solve_ridge(self, A, B, lam=1e-3):
        """
        Solves (A^T A + \lambda I) X = A^T B for X.
        This Tikhonov regularization guarantees invertibility and bounds weights.
        """
        I = torch.eye(A.size(1), device=A.device, dtype=A.dtype)
        return torch.linalg.solve(A.T @ A + lam * I, A.T @ B)
        
    def solve_and_update(self, x_actual, y_target, lr=1.0):
        """
        Solves for the optimal weights to map x_actual to y_target and applies 
        the update with the given learning rate.
        """
        if self.bias is not None:
            ones = torch.ones(x_actual.size(0), 1, dtype=x_actual.dtype, device=x_actual.device)
            x_aug = torch.cat([x_actual, ones], dim=1)
            
            W_aug_T = self._solve_ridge(x_aug, y_target)
            W_opt = W_aug_T.T
            
            dW = W_opt[:, :-1] - self.weight.data
            db = W_opt[:, -1] - self.bias.data
            
            self.weight.data += lr * dW
            self.bias.data += lr * db
        else:
            W_T = self._solve_ridge(x_actual, y_target)
            W_opt = W_T.T
            
            dW = W_opt - self.weight.data
            self.weight.data += lr * dW
