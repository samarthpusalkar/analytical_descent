import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import copy

from models.analytical_network import AnalyticalSequential
from layers.analytical_linear import AnalyticalLinear
from layers.analytical_conv import AnalyticalConv2d
from layers.analytical_flatten import AnalyticalFlatten
from core.inversions import AnalyticalLeakyReLU
from utils.logger import BenchmarkLogger

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item() * x.size(0)
            
            _, predicted = torch.max(pred.data, 1)
            correct += (predicted == y).sum().item()
            total += x.size(0)
            
    model.train()
    return total_loss / total, correct / total

def run_cnn_experiment(epochs=5, batch_size=64):
    logger = BenchmarkLogger("Analytical CNN on FashionMNIST")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Using FashionMNIST (more complex than MNIST, fast download)
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    
    # 2. Build CNN Model
    # Input: [N, 1, 28, 28]
    ana_model = AnalyticalSequential(
        AnalyticalConv2d(1, 16, kernel_size=4, stride=2, padding=1), # Output: [N, 16, 14, 14]
        AnalyticalLeakyReLU(0.01),
        AnalyticalFlatten(),
        AnalyticalLinear(16 * 14 * 14, 10)
    )

    gd_model = copy.deepcopy(ana_model)
    optimizer = optim.Adam(gd_model.parameters(), lr=0.005)
    
    print(f"{'Step':<4} | {'Ana TrLoss':<10} | {'Bas TrLoss':<10} | {'Ana TeLoss':<10} | {'Bas TeLoss':<10} | {'Ana TrAcc':<9} | {'Bas TrAcc':<9} | {'Ana TeAcc':<9} | {'Bas TeAcc':<9}")
    print("-" * 115)
    
    for epoch in range(1, epochs + 1):
        start_t = time.time()
        
        ana_running_loss = 0.0
        gd_running_loss = 0.0
        ana_correct = 0
        gd_correct = 0
        total_samples = 0
        
        for x_b, y_b in train_loader:
            batch_sz = x_b.size(0)
            
            Y_onehot_b = torch.zeros(batch_sz, 10, device=x_b.device)
            Y_onehot_b.scatter_(1, y_b.unsqueeze(1), 1.0)
            
            # --- Analytical Step ---
            ana_pred_b = ana_model(x_b)
            ana_loss_val = criterion(ana_pred_b, y_b)
            ana_running_loss += ana_loss_val.item() * batch_sz
            
            _, ana_pred_labels = torch.max(ana_pred_b.data, 1)
            ana_correct += (ana_pred_labels == y_b).sum().item()
            
            ana_model.backward_target(Y_onehot_b, lr=0.1, lr_decay=0.5)
            
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
            
        ana_train_loss = ana_running_loss / total_samples
        gd_train_loss = gd_running_loss / total_samples
        ana_train_acc = ana_correct / total_samples
        gd_train_acc = gd_correct / total_samples
        
        ana_test_loss, ana_test_acc = evaluate(ana_model, test_loader, criterion)
        gd_test_loss, gd_test_acc = evaluate(gd_model, test_loader, criterion)
            
        print(f"{epoch:<4} | {ana_train_loss:<10.4f} | {gd_train_loss:<10.4f} | {ana_test_loss:<10.4f} | {gd_test_loss:<10.4f} | {ana_train_acc:<9.4f} | {gd_train_acc:<9.4f} | {ana_test_acc:<9.4f} | {gd_test_acc:<9.4f}")
        print(f"  --> Epoch {epoch} finished in {time.time()-start_t:.1f}s")
                        
    print("-" * 115)

if __name__ == "__main__":
    run_cnn_experiment()
