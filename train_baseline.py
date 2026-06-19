import torch
import torch.nn as nn
import torch.optim as optim

def test_1_layer():
    print("--- Testing 1-Layer Baseline Model (Gradient Descent) ---")
    torch.manual_seed(42)
    
    # Generate identical toy data as in train.py
    X = torch.randn(100, 5)
    true_W = torch.randn(5, 2)
    true_b = torch.randn(2)
    Y_target = X @ true_W + true_b
    
    model = nn.Sequential(
        nn.Linear(5, 2)
    )
    
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    print(" Epoch | MSE Loss")
    print("------------------")
    for epoch in range(1, 11):
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = criterion(Y_pred, Y_target)
        
        # Using 1 epoch print just to keep it concise, though GD takes longer to converge to 0
        print(f" {epoch:5d} | {loss.item():.6f}")
        
        loss.backward()
        optimizer.step()
        
    # Final evaluation
    Y_pred = model(X)
    loss = criterion(Y_pred, Y_target)
    print(f" Final | {loss.item():.6f}\n")


def test_2_layer():
    print("--- Testing 2-Layer Baseline Model (Gradient Descent) ---")
    torch.manual_seed(42)
    
    # Generate identical toy data as in train.py
    X = torch.randn(200, 5)
    
    # Generate targets using a non-linear target model
    target_model = nn.Sequential(
        nn.Linear(5, 10),
        nn.LeakyReLU(0.01),
        nn.Linear(10, 2)
    )
    with torch.no_grad():
        Y_target = target_model(X)
        
    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.LeakyReLU(0.01),
        nn.Linear(10, 2)
    )
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print(" Epoch | MSE Loss")
    print("------------------")
    for epoch in range(1, 11):
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = criterion(Y_pred, Y_target)
        
        print(f" {epoch:5d} | {loss.item():.6f}")
        
        loss.backward()
        optimizer.step()
        
    # Final evaluation
    Y_pred = model(X)
    loss = criterion(Y_pred, Y_target)
    print(f" Final | {loss.item():.6f}")

if __name__ == "__main__":
    test_1_layer()
    test_2_layer()
