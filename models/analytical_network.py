import torch
import torch.nn as nn
from core.analytical_linear import AnalyticalLinear

class AnalyticalSequential(nn.Module):
    """
    A Sequential container for analytical layers that supports 
    target back-propagation for training.
    """
    def __init__(self, *modules):
        super().__init__()
        self.layers = nn.ModuleList(modules)
        self.layer_inputs = []
        self.network_output = None
        
    def forward(self, x):
        self.layer_inputs = [x.clone().detach()]
        for layer in self.layers:
            x = layer(x)
            self.layer_inputs.append(x.clone().detach())
        self.network_output = x.clone().detach()
        return x
        
    def backward_target(self, final_target, lr=1.0, lr_decay=1.0):
        """
        Implements Forward-First Gradient-Nudged Target Propagation.
        """
        # Phase 1: Forward pass is already done, actual representations are cached.
        
        # Calculate output error
        current_error = final_target - self.network_output
        current_lr = lr
        
        # Phase 2: Traverse backwards, update weights mapping actual_in -> actual_out + error
        for idx in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[idx]
            h_in = self.layer_inputs[idx]
            h_out_actual = self.layer_inputs[idx + 1]
            
            if hasattr(layer, 'propagate_error'):
                if isinstance(layer, AnalyticalLinear):
                    # Target is the actual representation nudged by the error
                    h_target = h_out_actual + current_error
                    
                    # Update weights to map actual input to the nudged target
                    layer.solve_and_update(h_in, h_target, lr=current_lr)
                    
                    # Propagate error backwards through the linear layer
                    current_error = layer.propagate_error(current_error)
                    current_lr *= lr_decay
                else:
                    # Activation layer (AnalyticalLeakyReLU)
                    current_error = layer.propagate_error(current_error, h_in)
