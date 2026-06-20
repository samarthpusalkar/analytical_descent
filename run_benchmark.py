import sys
import os
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from layers.analytical_conv import AnalyticalConv2d
from layers.analytical_flatten import AnalyticalFlatten
from layers.analytical_linear import AnalyticalLinear
from core.inversions import AnalyticalLeakyReLU
from models.analytical_network import AnalyticalSequential

def run_bench(method_name, max_samples=None, approx_method=None):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    
    import core.solver
    original_compute = core.solver.compute_analytical_delta
    def patched_compute(x_in, error, lam=1e-3, max_norm=1.0, **kwargs):
        return original_compute(x_in, error, lam, max_norm, max_samples=max_samples, approx_method=approx_method)
    
    import layers.analytical_conv
    import layers.analytical_linear
    layers.analytical_conv.compute_analytical_delta = patched_compute
    layers.analytical_linear.compute_analytical_delta = patched_compute
    
    model = AnalyticalSequential(
        AnalyticalConv2d(1, 16, kernel_size=4, stride=2, padding=1),
        AnalyticalLeakyReLU(0.01),
        AnalyticalFlatten(),
        AnalyticalLinear(16 * 14 * 14, 128),
        AnalyticalLeakyReLU(0.01),
        AnalyticalLinear(128, 10)
    )
    
    print(f"\n--- Running: {method_name} (Batch Size: 64) ---")
    
    start_t = time.time()
    total_samples = 0
    for x, y in train_loader:
        Y_onehot = torch.zeros(64, 10, device=x.device)
        Y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
        
        # Forward pass to populate self.network_output
        model(x)
        model.backward_target(Y_onehot, lr=0.07, lr_decay=0.70)
        
        total_samples += 64
        if total_samples >= 64 * 50: break # Run 50 batches to benchmark speed
        
    end_t = time.time()
    print(f"Time for 50 batches: {end_t - start_t:.3f}s")
    
    layers.analytical_conv.compute_analytical_delta = original_compute
    layers.analytical_linear.compute_analytical_delta = original_compute

run_bench("Exact (Cholesky)", max_samples=None, approx_method=None)
run_bench("Sketching (Subsample 1024)", max_samples=1024, approx_method=None)
run_bench("Diagonal (Jacobi)", max_samples=None, approx_method="diagonal")

