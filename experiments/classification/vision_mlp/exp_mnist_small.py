import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy

from layers.analytical_linear import AnalyticalLinear
from core.inversions import AnalyticalLeakyReLU
from models.analytical_network import AnalyticalSequential
from utils.data import get_digits_dataset
from utils.metrics import calculate_classification_metrics
from utils.logger import BenchmarkLogger

def create_model(num_layers, in_features=64, hidden_features=32, out_features=10):
    """
    Creates an AnalyticalSequential model with the specified number of layers.
    """
    layers = []
    if num_layers == 1:
        layers.append(AnalyticalLinear(in_features, out_features))
    elif num_layers == 2:
        layers.append(AnalyticalLinear(in_features, hidden_features))
        layers.append(AnalyticalLeakyReLU(0.01))
        layers.append(AnalyticalLinear(hidden_features, out_features))
    elif num_layers == 3:
        layers.append(AnalyticalLinear(in_features, hidden_features))
        layers.append(AnalyticalLeakyReLU(0.01))
        layers.append(AnalyticalLinear(hidden_features, hidden_features))
        layers.append(AnalyticalLeakyReLU(0.01))
        layers.append(AnalyticalLinear(hidden_features, out_features))
    else:
        raise ValueError("Only 1, 2, or 3 layers supported for this benchmark.")
        
    return AnalyticalSequential(*layers)


def run_experiment(num_layers, X_train, y_train, Y_train_onehot, 
                   X_test, y_test, epochs=150, eval_steps=10, batch_size=64,
                   base_lr=0.1, ana_lr_base=0.01, ana_decay_ratio=0.5, 
                   criterion=nn.CrossEntropyLoss()):
    logger = BenchmarkLogger(f"{num_layers}-Layer Digits Classification")
    logger.print_header()
    
    # 1. Initialize base model and duplicate it so weights are perfectly identical
    torch.manual_seed(42)
    base_model = create_model(num_layers)
    ana_model = copy.deepcopy(base_model)
    gd_model = copy.deepcopy(base_model)
    
    # Optimizer for Gradient Descent Baseline
    optimizer = optim.SGD(gd_model.parameters(), lr=base_lr)
    
    # Create DataLoader for batch-wise updates
    dataset = TensorDataset(X_train, y_train, Y_train_onehot)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(1, epochs + 1):
        for x_b, y_b, Y_onehot_b in dataloader:
            # --- Analytical Step ---
            ana_pred_b = ana_model(x_b)
            
            # Calculate dynamic learning rate for analytical model based on depth
            ana_lr = ana_lr_base if num_layers > 1 else ana_lr_base
            ana_decay = ana_decay_ratio if num_layers > 1 else 1.0
            
            ana_model.backward_target(Y_onehot_b, lr=ana_lr, lr_decay=ana_decay)
            
            # --- Baseline Step ---
            optimizer.zero_grad()
            gd_pred_b = gd_model(x_b)
            
            if isinstance(criterion, nn.MSELoss):
                loss = criterion(gd_pred_b, Y_onehot_b)
            else:
                loss = criterion(gd_pred_b, y_b)
                
            loss.backward()
            optimizer.step()
            
        # Calculate overall training metrics at the end of the epoch
        with torch.no_grad():
            ana_pred = ana_model(X_train)
            ana_metrics = calculate_classification_metrics(ana_pred, y_train)
            
            gd_pred = gd_model(X_train)
            gd_metrics = calculate_classification_metrics(gd_pred, y_train)
        
        # --- Evaluation Step ---
        ana_test_metrics = None
        gd_test_metrics = None
        
        if epoch % eval_steps == 0 or epoch == epochs:
            with torch.no_grad():
                ana_test_pred = ana_model(X_test)
                ana_test_metrics = calculate_classification_metrics(ana_test_pred, y_test)
                
                gd_test_pred = gd_model(X_test)
                gd_test_metrics = calculate_classification_metrics(gd_test_pred, y_test)
        
        # Log
        logger.log_step(epoch, ana_metrics, gd_metrics, ana_test_metrics, gd_test_metrics)
        
    logger.save_to_csv("../../../results/classification/vision_mlp/benchmark_results.csv")
    print("-" * 75)


if __name__ == "__main__":
    print("Loading Dataset...")
    X_train, X_test, y_train, y_test, Y_train_onehot, Y_test_onehot = get_digits_dataset()
    print(f"Loaded Scikit-Learn Digits: Train Size={X_train.shape[0]}, Features={X_train.shape[1]}")
    
    # Run Benchmark for varying depths
    run_experiment(num_layers=1, X_train=X_train, y_train=y_train, Y_train_onehot=Y_train_onehot, 
                   X_test=X_test, y_test=y_test, epochs=150)
    run_experiment(num_layers=2, X_train=X_train, y_train=y_train, Y_train_onehot=Y_train_onehot, 
                   X_test=X_test, y_test=y_test, epochs=150)
    run_experiment(num_layers=3, X_train=X_train, y_train=y_train, Y_train_onehot=Y_train_onehot, 
                   X_test=X_test, y_test=y_test, epochs=150)
    
    print("\nBenchmark complete. Results saved to benchmark_results.csv")
