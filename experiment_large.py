import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import pandas as pd
import numpy as np
import copy

from models.analytical_network import AnalyticalSequential
from layers.analytical_linear import AnalyticalLinear
from core.inversions import AnalyticalLeakyReLU

class BenchmarkLogger:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.results = []
        
    def print_header(self):
        print(f"\n{'='*20} {self.experiment_name} {'='*20}")
        print(f"{'Epoch':<4} | {'Ana Loss':<10} | {'GD Loss':<10} | {'Ana TestL':<10} | {'GD TestL':<10} | {'Ana Acc':<9} | {'GD Acc':<9} | {'Ana TestA':<9} | {'GD TestA':<9}")
        print("-" * 115)
        
    def log_step(self, epoch, ana_loss, gd_loss, ana_test_loss, gd_test_loss, 
                 ana_acc, gd_acc, ana_test_acc, gd_test_acc):
        print(f"{epoch:<4} | {ana_loss:<10.4f} | {gd_loss:<10.4f} | {ana_test_loss:<10.4f} | {gd_test_loss:<10.4f} | {ana_acc:<9.4f} | {gd_acc:<9.4f} | {ana_test_acc:<9.4f} | {gd_test_acc:<9.4f}")
        self.results.append({
            'Epoch': epoch,
            'Ana_Loss': ana_loss, 'GD_Loss': gd_loss,
            'Ana_Test_Loss': ana_test_loss, 'GD_Test_Loss': gd_test_loss,
            'Ana_Train_Acc': ana_acc, 'GD_Train_Acc': gd_acc,
            'Ana_Test_Acc': ana_test_acc, 'GD_Test_Acc': gd_test_acc
        })

def evaluate(model, dataloader, criterion, is_analytical=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            if is_analytical:
                pred = model(x)
            else:
                pred = model(x)
                
            loss = criterion(pred, y)
            total_loss += loss.item() * x.size(0)
            
            _, predicted = torch.max(pred.data, 1)
            correct += (predicted == y).sum().item()
            total += x.size(0)
            
    model.train()
    return total_loss / total, correct / total

def run_experiment(epochs=10, eval_steps=1, batch_size=16,
                   base_lr=0.01, ana_lr_base=0.07, ana_decay_ratio=0.70):
    logger = BenchmarkLogger("4-Layer MNIST Classification")
    
    # 1. Load MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    
    # 2. Build 4-Layer Models (784 -> 256 -> 128 -> 64 -> 10)
    # Baseline Model (Gradient Descent)
    # gd_model = nn.Sequential(
    #     nn.Linear(784, 256), nn.LeakyReLU(0.01),
    #     nn.Linear(256, 128), nn.LeakyReLU(0.01),
    #     nn.Linear(128, 64), nn.LeakyReLU(0.01),
    #     nn.Linear(64, 10)
    # )
    # optimizer = optim.Adam(gd_model.parameters(), lr=base_lr)
    
    # Analytical Model
    ana_model = AnalyticalSequential(
        AnalyticalLinear(784, 256), AnalyticalLeakyReLU(0.01),
        AnalyticalLinear(256, 128), AnalyticalLeakyReLU(0.01),
        AnalyticalLinear(128, 64), AnalyticalLeakyReLU(0.01),
        AnalyticalLinear(64, 10)
    )

    gd_model = copy.deepcopy(ana_model)
    optimizer = optim.Adam(gd_model.parameters(), lr=base_lr)
    logger.print_header()
    
    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        start_t = time.time()
        gd_model.train()
        
        ana_running_loss = 0.0
        gd_running_loss = 0.0
        ana_correct = 0
        gd_correct = 0
        total_samples = 0
        
        for x_b, y_b in train_loader:
            batch_sz = x_b.size(0)
            
            # Create one-hot targets for analytical model
            Y_onehot_b = torch.zeros(batch_sz, 10, device=x_b.device)
            Y_onehot_b.scatter_(1, y_b.unsqueeze(1), 1.0)
            
            # --- Analytical Step ---
            ana_pred_b = ana_model(x_b)
            ana_loss_val = criterion(ana_pred_b, y_b)
            ana_running_loss += ana_loss_val.item() * batch_sz
            
            _, ana_pred_labels = torch.max(ana_pred_b.data, 1)
            ana_correct += (ana_pred_labels == y_b).sum().item()
            
            # We use Forward-First Error-Nudged update
            ana_model.backward_target(Y_onehot_b, lr=ana_lr_base, lr_decay=ana_decay_ratio)
            
            # --- Baseline Step ---
            optimizer.zero_grad()
            gd_pred_b = gd_model(x_b)
            loss = criterion(gd_pred_b, y_b)
            loss.backward()
            optimizer.step()
            
            gd_running_loss += loss.item() * batch_sz
            _, gd_pred_labels = torch.max(gd_pred_b.data, 1)
            gd_correct += (gd_pred_labels == y_b).sum().item()
            
            total_samples += batch_sz
            
        # Calculate epoch metrics
        ana_train_loss = ana_running_loss / total_samples
        gd_train_loss = gd_running_loss / total_samples
        ana_train_acc = ana_correct / total_samples
        gd_train_acc = gd_correct / total_samples
        
        # Test evaluation
        ana_test_loss, gd_test_loss = 0.0, 0.0
        ana_test_acc, gd_test_acc = 0.0, 0.0
        
        if epoch % eval_steps == 0 or epoch == epochs:
            ana_test_loss, ana_test_acc = evaluate(ana_model, test_loader, criterion, True)
            gd_test_loss, gd_test_acc = evaluate(gd_model, test_loader, criterion, False)
            
        logger.log_step(epoch, ana_train_loss, gd_train_loss, 
                        ana_test_loss, gd_test_loss,
                        ana_train_acc, gd_train_acc,
                        ana_test_acc, gd_test_acc)
        print(f"  --> Epoch {epoch} finished in {time.time()-start_t:.1f}s")
                        
    print("-" * 115)
    print("\nBenchmark complete. Results saved to benchmark_mnist_results.csv")
    df = pd.DataFrame(logger.results)
    df.to_csv("benchmark_mnist_results.csv", index=False)

if __name__ == "__main__":
    run_experiment()
