import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import copy

from models.analytical_network import AnalyticalSequential
from layers.analytical_linear import AnalyticalLinear
from core.inversions import AnalyticalLeakyReLU

def run_california_experiment(epochs=30, batch_size=128):
    print("================ Analytical California Housing Benchmark ================")
    
    # 1. Dataset
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    # Normalize features and targets for stable regression
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X = scaler_x.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 2. Models
    # 8 features -> 64 -> 64 -> 1
    ana_model = AnalyticalSequential(
        AnalyticalLinear(8, 64), AnalyticalLeakyReLU(0.01),
        AnalyticalLinear(64, 64), AnalyticalLeakyReLU(0.01),
        AnalyticalLinear(64, 1)
    )
    
    # Baseline model
    gd_model = copy.deepcopy(ana_model)
    optimizer = optim.Adam(gd_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    print(f"{'Epoch':<5} | {'Ana Train MSE':<15} | {'GD Train MSE':<15} | {'Ana Test MSE':<15} | {'GD Test MSE':<15}")
    print("-" * 80)
    
    for epoch in range(1, epochs + 1):
        ana_loss_total = 0.0
        gd_loss_total = 0.0
        
        for x_b, y_b in train_loader:
            # --- Analytical Update ---
            ana_pred = ana_model(x_b)
            ana_loss = criterion(ana_pred, y_b)
            ana_loss_total += ana_loss.item()
            
            ana_model.backward_target(y_b, lr=0.05, lr_decay=0.9)
            
            # --- Gradient Descent Update ---
            optimizer.zero_grad()
            gd_pred = gd_model(x_b)
            gd_loss = criterion(gd_pred, y_b)
            gd_loss.backward()
            optimizer.step()
            
            gd_loss_total += gd_loss.item()
            
        # Evaluation
        if epoch % 5 == 0 or epoch == 1:
            ana_model.eval()
            gd_model.eval()
            with torch.no_grad():
                ana_test_pred = ana_model(X_test_t)
                gd_test_pred = gd_model(X_test_t)
                
                ana_test_mse = criterion(ana_test_pred, y_test_t).item()
                gd_test_mse = criterion(gd_test_pred, y_test_t).item()
                
            ana_train_mse = ana_loss_total / len(train_loader)
            gd_train_mse = gd_loss_total / len(train_loader)
            
            print(f"{epoch:<5} | {ana_train_mse:<15.4f} | {gd_train_mse:<15.4f} | {ana_test_mse:<15.4f} | {gd_test_mse:<15.4f}")
            ana_model.train()
            gd_model.train()

if __name__ == '__main__':
    run_california_experiment()
