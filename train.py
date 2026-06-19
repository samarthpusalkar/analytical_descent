import torch
import torch.nn as nn
from core.analytical_linear import AnalyticalLinear
from core.inversions import AnalyticalLeakyReLU
from models.analytical_network import AnalyticalSequential

def test_1_layer():
    print("--- Testing 1-Layer Analytical Model ---")
    torch.manual_seed(42)
    
    # Generate some toy data
    X = torch.randn(100, 5)
    true_W = torch.randn(5, 2)
    true_b = torch.randn(2)
    Y_target = X @ true_W + true_b
    
    model = AnalyticalSequential(
        AnalyticalLinear(5, 2)
    )
    
    # Forward pass
    Y_pred = model(X)
    initial_loss = nn.MSELoss()(Y_pred, Y_target).item()
    print(f"Initial MSE: {initial_loss:.6f}")
    
    # Backward target pass
    # For a 1-layer model, the analytical solution should solve it perfectly in 1 step.
    model.backward_target(Y_target, lr=1.0)
    
    # Verify
    Y_pred_new = model(X)
    new_loss = nn.MSELoss()(Y_pred_new, Y_target).item()
    print(f"Post-update MSE: {new_loss:.6f}\n")


def test_2_layer():
    print("--- Testing 2-Layer Analytical Model ---")
    torch.manual_seed(42)
    
    # Generate some toy data
    X = torch.randn(200, 5)
    
    # Generate targets using a non-linear target model
    target_model = nn.Sequential(
        nn.Linear(5, 10),
        nn.LeakyReLU(0.01),
        nn.Linear(10, 2)
    )
    with torch.no_grad():
        Y_target = target_model(X)
        
    model = AnalyticalSequential(
        AnalyticalLinear(5, 10),
        AnalyticalLeakyReLU(0.01),
        AnalyticalLinear(10, 2)
    )
    
    # Train for a few epochs
    print(" Epoch | MSE Loss")
    print("------------------")
    for epoch in range(1, 11):
        Y_pred = model(X)
        loss = nn.MSELoss()(Y_pred, Y_target).item()
        print(f" {epoch:5d} | {loss:.6f}")
        
        # Use a smaller learning rate for stability in deep models
        model.backward_target(Y_target, lr=0.01, lr_decay=0.5)
        
    # Final evaluation
    Y_pred = model(X)
    loss = nn.MSELoss()(Y_pred, Y_target).item()
    print(f" Final | {loss:.6f}")

if __name__ == "__main__":
    test_1_layer()
    test_2_layer()
