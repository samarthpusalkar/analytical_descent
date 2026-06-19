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
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def backward_target(self, final_target, lr=1.0, lr_decay=1.0):
        """
        Backpropagates the target through the network and calculates updates.
        
        Args:
            final_target: The desired output for the final layer.
            lr: Initial learning rate for the final layer update.
            lr_decay: Multiplier for learning rate at each subsequent lower layer.
                      As we back-calculate correct values, divergence can grow,
                      so slightly decreasing the learning rate is recommended.
        """
        current_target = final_target
        current_lr = lr
        
        # Traverse layers in reverse order
        for layer in reversed(self.layers):
            if hasattr(layer, 'inverse_func'):
                # Apply inverse activation to get pre-activation target
                current_target = layer.inverse_func(current_target)
            elif isinstance(layer, AnalyticalLinear):
                # Calculate analytical update and get input target for previous layer
                current_target = layer.calculate_updates_and_target(current_target)
                
                # Apply the weight updates
                layer.apply_updates(lr=current_lr)
                
                # Decay learning rate for the preceding layers
                current_lr *= lr_decay
            else:
                # Fallback for standard linear layers if we want to mix and match (though they won't update)
                pass
