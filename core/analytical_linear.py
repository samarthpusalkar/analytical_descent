import torch
import torch.nn as nn

class AnalyticalLinear(nn.Linear):
    """
    A linear layer that supports analytical weight updates via least squares
    and back-calculation of target inputs using pseudo-inverse.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        
    def propagate_error(self, output_error):
        """
        Propagates the error backward through the layer using the transpose of the weights.
        This preserves the null space of the forward representation.
        """
        return output_error @ self.weight.data
        
    def _solve_ridge(self, A, B, lam=1e-3):
        """
        Solves (A^T A + \lambda I) X = A^T B for X.
        This Tikhonov regularization guarantees invertibility and bounds weights.
        """
        I = torch.eye(A.size(1), device=A.device, dtype=A.dtype)
        return torch.linalg.solve(A.T @ A + lam * I, A.T @ B)
        
    def solve_and_update(self, x_actual, y_target, lr=1.0):
        """
        Calculates the optimal weight updates using Delta-Regularized Ridge Regression.
        This minimizes ||X_aug (W + dW)^T - Y_target||^2 + lam ||dW||^2.
        dW^T = (X_aug^T X_aug + lam I)^-1 X_aug^T (Y_target - X_aug W^T)
        """
        # We want to map x_actual -> y_target.
        
        if self.bias is not None:
            # Augment x with ones for bias trick
            ones = torch.ones(x_actual.size(0), 1, dtype=x_actual.dtype, device=x_actual.device)
            x_aug = torch.cat([x_actual, ones], dim=1)
            W_aug = torch.cat([self.weight.data, self.bias.data.unsqueeze(1)], dim=1)
            
            # Calculate current prediction and error
            y_pred = x_aug @ W_aug.T
            error = y_target - y_pred
            
            # Solve for delta
            dW_aug_T = self._solve_ridge(x_aug, error)
            dW_aug = dW_aug_T.T
            
            # Extract weights and bias deltas
            dW = dW_aug[:, :-1]
            db = dW_aug[:, -1]
            
            # Clip gradients to prevent massive steps from conditioning noise
            max_norm = 1.0
            norm_w = torch.norm(dW)
            if norm_w > max_norm:
                dW = dW * (max_norm / norm_w)
                
            norm_b = torch.norm(db)
            if norm_b > max_norm:
                db = db * (max_norm / norm_b)
            
            self.weight.data += lr * dW
            self.bias.data += lr * db
        else:
            # Calculate current prediction and error
            y_pred = x_actual @ self.weight.data.T
            error = y_target - y_pred
            
            # Solve for delta
            dW_T = self._solve_ridge(x_actual, error)
            dW = dW_T.T
            
            # Clip gradients to prevent massive steps from conditioning noise
            max_norm = 1.0
            norm = torch.norm(dW)
            if norm > max_norm:
                dW = dW * (max_norm / norm)
            
            self.weight.data += lr * dW
