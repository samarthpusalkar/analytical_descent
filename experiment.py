import torch
import torch.nn as nn
import torch.optim as optim
import copy

from core.analytical_linear import AnalyticalLinear
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


def run_experiment(num_layers, X_train, y_train, Y_train_onehot, epochs=15):
    logger = BenchmarkLogger(f"{num_layers}-Layer Digits Classification")
    logger.print_header()
    
    # 1. Initialize base model and duplicate it so weights are perfectly identical
    torch.manual_seed(42)
    base_model = create_model(num_layers)
    ana_model = copy.deepcopy(base_model)
    gd_model = copy.deepcopy(base_model)
    
    # Optimizer for Gradient Descent Baseline
    optimizer = optim.SGD(gd_model.parameters(), lr=0.01)
    
    for epoch in range(1, epochs + 1):
        # --- Analytical Step ---
        ana_pred = ana_model(X_train)
        ana_metrics = calculate_classification_metrics(ana_pred.detach(), y_train)
        
        # We need lr_decay and smaller base lr for deeper models
        # For 1 layer, lr=1.0 works perfectly. For deeper, we need small steps.
        ana_lr = 1.0 if num_layers == 1 else (0.1 if num_layers == 2 else 0.05)
        ana_decay = 1.0 if num_layers == 1 else (0.5 if num_layers == 2 else 0.5)
        
        ana_model.backward_target(Y_train_onehot, lr=ana_lr, lr_decay=ana_decay)
        
        # --- Baseline Step ---
        optimizer.zero_grad()
        gd_pred = gd_model(X_train)
        
        # MSE against one-hot for fair apples-to-apples comparison
        # (Though cross-entropy is standard for classification, we want the exact same objective landscape)
        loss = nn.MSELoss()(gd_pred, Y_train_onehot)
        loss.backward()
        optimizer.step()
        
        gd_metrics = calculate_classification_metrics(gd_pred.detach(), y_train)
        
        # Log
        logger.log_step(epoch, ana_metrics, gd_metrics)
        
    logger.save_to_csv()
    print("-" * 75)


if __name__ == "__main__":
    print("Loading Dataset...")
    X_train, X_test, y_train, y_test, Y_train_onehot, Y_test_onehot = get_digits_dataset()
    print(f"Loaded Scikit-Learn Digits: Train Size={X_train.shape[0]}, Features={X_train.shape[1]}")
    
    # Run Benchmark for varying depths
    run_experiment(num_layers=1, X_train=X_train, y_train=y_train, Y_train_onehot=Y_train_onehot, epochs=15)
    run_experiment(num_layers=2, X_train=X_train, y_train=y_train, Y_train_onehot=Y_train_onehot, epochs=15)
    run_experiment(num_layers=3, X_train=X_train, y_train=y_train, Y_train_onehot=Y_train_onehot, epochs=15)
    
    print("\nBenchmark complete. Results saved to benchmark_results.csv")
