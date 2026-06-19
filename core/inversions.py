import torch
import torch.nn as nn

class InverseLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope
        
    def forward(self, y):
        # Inverse of LeakyReLU
        # f(x) = x if x > 0 else x * slope
        # f^-1(y) = y if y > 0 else y / slope
        mask = (y > 0).float()
        return y * mask + (y / self.negative_slope) * (1 - mask)


class InverseReLU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y):
        # Pseudo-inverse of ReLU
        # Negative information is lost, so we just return max(0, y) or identity
        return torch.relu(y)


class AnalyticalLeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__(negative_slope, inplace)
        self.inverse_module = InverseLeakyReLU(negative_slope)
        
    def inverse_func(self, y):
        return self.inverse_module(y)
        
    def propagate_error(self, output_error, h_in):
        mask = (h_in > 0).float()
        derivative = mask + self.negative_slope * (1 - mask)
        return output_error * derivative
