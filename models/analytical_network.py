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
        self.network_input = None
        
    def forward(self, x):
        self.network_input = x.clone().detach()
        for layer in self.layers:
            x = layer(x)
        return x
        
    def backward_target(self, final_target, lr=1.0, lr_decay=1.0):
        """
        Implements Two-Phase Forward-Update for Analytical Target Propagation.
        """
        # Phase 1: Back-calculate targets
        layer_targets = {}
        layer_lrs = {}
        current_target = final_target
        current_lr = lr
        
        # Traverse in reverse to compute targets
        for idx in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[idx]
            if hasattr(layer, 'inverse_func'):
                current_target = layer.inverse_func(current_target)
                layer_targets[idx] = current_target
            elif isinstance(layer, AnalyticalLinear):
                layer_targets[idx] = current_target
                layer_lrs[idx] = current_lr
                # Calculate input target for previous layer using pseudo-inverse
                current_target = layer.get_input_target(current_target)
                current_lr *= lr_decay
                
        # Phase 2: Forward update
        x = self.network_input
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, AnalyticalLinear):
                target = layer_targets[idx]
                layer_lr = layer_lrs[idx]
                
                # Solve least squares to map CURRENT 'x' to 'target' and apply update
                layer.solve_and_update(x, target, lr=layer_lr)
                
            # Pass x through the newly updated layer to get exact input for next layer
            x = layer(x)
