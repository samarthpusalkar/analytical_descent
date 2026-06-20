import sys
import os
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.analytical_network import AnalyticalSequential
from layers.analytical_conv import AnalyticalConv2d
from layers.analytical_flatten import AnalyticalFlatten
from layers.analytical_linear import AnalyticalLinear
from core.inversions import AnalyticalLeakyReLU

def evaluate(model, dataloader, criterion):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            correct += (pred.argmax(1) == y).sum().item()
            total += x.size(0)
    model.train()
    return correct / total

def run_bench(method_name, max_samples=None, approx_method=None):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
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
    
    print(f"\nTraining: {method_name}")
    for epoch in range(1, 3):
        for x, y in train_loader:
            Y_onehot = torch.zeros(128, 10, device=x.device)
            Y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
            model(x)
            model.backward_target(Y_onehot, lr=0.07, lr_decay=0.70)
        
        acc = evaluate(model, test_loader, nn.CrossEntropyLoss())
        print(f"  Epoch {epoch} Test Acc: {acc:.4f}")
        
    layers.analytical_conv.compute_analytical_delta = original_compute
    layers.analytical_linear.compute_analytical_delta = original_compute

run_bench("Exact Cholesky", max_samples=None, approx_method=None)
run_bench("Sketching (1024)", max_samples=1024, approx_method=None)
run_bench("Diagonal Approximation", max_samples=None, approx_method="diagonal")

# Baseline Adam
def run_baseline():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.01),
        nn.Flatten(),
        nn.Linear(16 * 14 * 14, 128), nn.LeakyReLU(0.01),
        nn.Linear(128, 10)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining: GD (Adam)")
    for epoch in range(1, 3):
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                correct += (model(x).argmax(1) == y).sum().item()
                total += x.size(0)
        model.train()
        print(f"  Epoch {epoch} Test Acc: {correct/total:.4f}")
        
run_baseline()
