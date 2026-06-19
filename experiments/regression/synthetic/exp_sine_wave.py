import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from models.analytical_network import AnalyticalSequential
from layers.analytical_linear import AnalyticalLinear
from core.inversions import AnalyticalLeakyReLU

def generate_data(num_samples=1000):
    # Generate non-linear complex synthetic data
    x = torch.linspace(-3, 3, num_samples).unsqueeze(1)
    # y = sin(3x) + cos(5x) + noise
    y = torch.sin(3 * x) + torch.cos(5 * x) + torch.randn_like(x) * 0.1
    return x, y

def run_regression_experiment(epochs=100, batch_size=32):
    print("================ Analytical Regression Benchmark ================")
    
    # 1. Dataset
    x_train, y_train = generate_data(2000)
    x_test, y_test = generate_data(500)
    
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Models
    # 1 -> 64 -> 64 -> 1
    ana_model = AnalyticalSequential(
        AnalyticalLinear(1, 64), AnalyticalLeakyReLU(0.01),
        AnalyticalLinear(64, 64), AnalyticalLeakyReLU(0.01),
        AnalyticalLinear(64, 1)
    )
    
    # Baseline model
    import copy
    gd_model = copy.deepcopy(ana_model)
    optimizer = optim.Adam(gd_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    print(f"{'Epoch':<5} | {'Ana MSE Loss':<15} | {'GD MSE Loss':<15}")
    print("-" * 40)
    
    for epoch in range(1, epochs + 1):
        ana_loss_total = 0.0
        gd_loss_total = 0.0
        
        for x_b, y_b in dataloader:
            # Analytical Update
            # In regression, the target is directly the continuous y_b values!
            ana_pred = ana_model(x_b)
            ana_loss = criterion(ana_pred, y_b)
            ana_loss_total += ana_loss.item()
            
            # Forward-first backward error update
            ana_model.backward_target(y_b, lr=0.01, lr_decay=0.99)
            
            # GD Update
            optimizer.zero_grad()
            gd_pred = gd_model(x_b)
            gd_loss = criterion(gd_pred, y_b)
            gd_loss.backward()
            optimizer.step()
            
            gd_loss_total += gd_loss.item()
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:<5} | {ana_loss_total/len(dataloader):<15.4f} | {gd_loss_total/len(dataloader):<15.4f}")
            
    # 4. Evaluation & Plotting
    ana_model.eval()
    gd_model.eval()
    with torch.no_grad():
        x_plot = torch.linspace(-3, 3, 500).unsqueeze(1)
        y_true = torch.sin(3 * x_plot) + torch.cos(5 * x_plot)
        y_ana = ana_model(x_plot)
        y_gd = gd_model(x_plot)
        
    plt.figure(figsize=(10, 6))
    plt.scatter(x_test.numpy(), y_test.numpy(), alpha=0.3, label='Test Data', color='gray')
    plt.plot(x_plot.numpy(), y_true.numpy(), 'k--', label='True Function', linewidth=2)
    plt.plot(x_plot.numpy(), y_ana.numpy(), 'r-', label='Analytical Model', linewidth=2)
    plt.plot(x_plot.numpy(), y_gd.numpy(), 'b-', label='Gradient Descent', linewidth=2)
    plt.legend()
    plt.title("Regression: Analytical Pseudo-Inverse vs Adam")
    plt.savefig("../../../results/regression/synthetic/regression_plot.png")
    print("Saved plot to regression_plot.png")

if __name__ == '__main__':
    run_regression_experiment()
