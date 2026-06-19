import torch
import torch.nn as nn
from layers.analytical_linear import AnalyticalLinear

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
        # Calculate output error
        # We use the gradient of the Huber Loss (clamped MSE) to prevent
        # massive gradient explosions from large confident logits while preserving
        # the stabilizing 'spring' effect of MSE for bounded regression.
        raw_error = final_target - self.network_output
        current_error = torch.clamp(raw_error, min=-1.0, max=1.0)
            
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
                    
                    # 1. Propagate error backwards through the linear layer using OLD weights
                    next_error = layer.propagate_error(current_error)
                    
                    # 2. Update weights safely to map actual input to the nudged target
                    layer.solve_and_update(h_in, h_target, lr=current_lr)
                    
                    # 3. Advance to next layer
                    current_error = next_error
                    current_lr *= lr_decay
                else:
                    # Activation layer (AnalyticalLeakyReLU)
                    current_error = layer.propagate_error(current_error, h_in)
