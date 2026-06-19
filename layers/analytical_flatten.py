import torch
import torch.nn as nn

class AnalyticalFlatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        
    def forward(self, x):
        self.input_shape = x.shape
        return x.view(x.size(0), -1)
        
    def propagate_error(self, output_error):
        return output_error.view(self.input_shape)
        
    def solve_and_update(self, x_actual, y_target, lr=1.0):
        pass # No weights to update
