import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from core.solver import compute_analytical_delta

class AnalyticalConv2d(nn.Conv2d):
    """
    A Convolutional layer that supports analytical filter updates.
    It works by unfolding the spatial patches into a flat matrix, solving the exact
    least-squares target, and folding the optimal deltas back into the spatial filters.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        
    def propagate_error(self, output_error):
        """
        Propagates error back through the transpose convolution.
        output_error: [N, OutC, H_out, W_out]
        """
        return F.conv_transpose2d(
            output_error, self.weight.data, 
            stride=self.stride, padding=self.padding, 
            dilation=self.dilation, groups=self.groups
        )
        
    def solve_and_update(self, x_actual, y_target, lr=1.0):
        """
        Calculates the optimal filter updates analytically using Woodbury Ridge Regression.
        """
        N, C, H_in, W_in = x_actual.shape
        
        # 1. Unfold the input into sliding window patches
        # Output shape: [N, C * K * K, L] where L is the number of spatial patches
        x_unfolded = F.unfold(
            x_actual, kernel_size=self.kernel_size, 
            dilation=self.dilation, padding=self.padding, stride=self.stride
        )
        
        N, D_patch, L = x_unfolded.shape
        
        # 2. Reshape into a flat matrix for the linear solver
        # [N * L, C * K * K]
        x_flat = x_unfolded.transpose(1, 2).reshape(N * L, D_patch)
        
        if self.bias is not None:
            # Augment with ones
            ones = torch.ones(x_flat.size(0), 1, dtype=x_flat.dtype, device=x_flat.device)
            x_aug = torch.cat([x_flat, ones], dim=1)
            
            # Reshape weights [OutC, C * K * K] and bias
            W_flat = self.weight.data.view(self.out_channels, -1)
            W_aug = torch.cat([W_flat, self.bias.data.unsqueeze(1)], dim=1)
            
            # Predict flat outputs: [N * L, OutC]
            y_pred_flat = x_aug @ W_aug.T
            
            # Reshape target to [N * L, OutC]
            # y_target: [N, OutC, H_out, W_out] -> [N, OutC, L] -> [N, L, OutC] -> [N*L, OutC]
            y_target_flat = y_target.view(N, self.out_channels, -1).transpose(1, 2).reshape(N * L, self.out_channels)
            
            error = y_target_flat - y_pred_flat
            
            # Solve analytically
            # (Solver automatically sketches down to 1024 samples for massive N)
            dW_aug = compute_analytical_delta(x_aug, error, lam=1e-3, max_norm=1.0)
            
            # Extract and reshape updates
            dW_flat = dW_aug[:, :-1]
            db = dW_aug[:, -1]
            
            dW = dW_flat.view(self.weight.data.shape)
            
            self.weight.data += lr * dW
            self.bias.data += lr * db
            
        else:
            W_flat = self.weight.data.view(self.out_channels, -1)
            
            y_pred_flat = x_flat @ W_flat.T
            y_target_flat = y_target.view(N, self.out_channels, -1).transpose(1, 2).reshape(N * L, self.out_channels)
            
            error = y_target_flat - y_pred_flat
            
            dW_flat = compute_analytical_delta(x_flat, error, lam=1e-3, max_norm=1.0)
            dW = dW_flat.view(self.weight.data.shape)
            
            self.weight.data += lr * dW
