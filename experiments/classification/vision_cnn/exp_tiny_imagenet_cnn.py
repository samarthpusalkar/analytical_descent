import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy

from models.analytical_network import AnalyticalSequential
from layers.analytical_conv import AnalyticalConv2d
from layers.analytical_linear import AnalyticalLinear
from layers.analytical_flatten import AnalyticalFlatten
from core.inversions import AnalyticalLeakyReLU

class BenchmarkLogger:
    def __init__(self, title):
        self.title = title
        print(f"\n==================== {title} ====================")
        print(f"{'Epoch':<5} | {'Ana TrLoss':<10} | {'Bas TrLoss':<10} | {'Ana TeLoss':<10} | {'Bas TeLoss':<10} | {'Ana TrAcc':<9} | {'Bas TrAcc':<9} | {'Ana TeAcc':<9} | {'Bas TeAcc':<9}")
        print("-" * 115)

    def log_epoch(self, epoch, ana_metrics, bas_metrics, epoch_time):
        print(f"{epoch:<5} | {ana_metrics['tr_loss']:<10.4f} | {bas_metrics['tr_loss']:<10.4f} | "
              f"{ana_metrics['te_loss']:<10.4f} | {bas_metrics['te_loss']:<10.4f} | "
              f"{ana_metrics['tr_acc']:<9.4f} | {bas_metrics['tr_acc']:<9.4f} | "
              f"{ana_metrics['te_acc']:<9.4f} | {bas_metrics['te_acc']:<9.4f}")
        print(f"  --> Epoch {epoch} finished in {epoch_time:.1f}s")


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            pred = model(x_b)
            loss = criterion(pred, y_b)
            
            total_loss += loss.item() * x_b.size(0)
            _, predicted = torch.max(pred.data, 1)
            total += y_b.size(0)
            correct += (predicted == y_b).sum().item()
            
    model.train()
    return total_loss / total, correct / total

def run_tiny_imagenet_experiment(epochs=10, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = BenchmarkLogger("Analytical CNN on Tiny-ImageNet (64x64)")
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    data_dir = os.path.join("data", "tiny-imagenet-200")
    if not os.path.exists(data_dir):
        print(f"Dataset not found at {data_dir}. Please run data/setup_tiny_imagenet.py first.")
        return
        
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform_train)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Dataset Loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    
    # Tiny-ImageNet input: 3x64x64
    # Conv1: 3->32 (k3, s2) -> 32x32x32
    # Conv2: 32->64 (k3, s2) -> 64x16x16
    # Conv3: 64->128 (k3, s2) -> 128x8x8
    # Flatten -> 8192
    ana_model = AnalyticalSequential(
        AnalyticalConv2d(3, 32, kernel_size=3, stride=2, padding=1), AnalyticalLeakyReLU(0.01),
        AnalyticalConv2d(32, 64, kernel_size=3, stride=2, padding=1), AnalyticalLeakyReLU(0.01),
        AnalyticalConv2d(64, 128, kernel_size=3, stride=2, padding=1), AnalyticalLeakyReLU(0.01),
        AnalyticalFlatten(),
        AnalyticalLinear(128 * 8 * 8, 512), AnalyticalLeakyReLU(0.01),
        AnalyticalLinear(512, 200)
    ).to(device)
    
    gd_model = copy.deepcopy(ana_model).to(device)
    optimizer = optim.Adam(gd_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training Loop
    num_classes = 200
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        ana_running_loss = 0.0
        gd_running_loss = 0.0
        ana_correct = 0
        gd_correct = 0
        total = 0
        
        for batch_idx, (x_b, y_b) in enumerate(train_loader):
            x_b, y_b = x_b.to(device), y_b.to(device)
            batch_sz = x_b.size(0)
            
            Y_onehot_b = torch.zeros(batch_sz, num_classes, device=device)
            Y_onehot_b.scatter_(1, y_b.unsqueeze(1), 1.0)
            
            # --- Analytical Step ---
            ana_pred_b = ana_model(x_b)
            ana_loss_val = criterion(ana_pred_b, y_b)
            ana_running_loss += ana_loss_val.item() * batch_sz
            
            _, ana_pred_labels = torch.max(ana_pred_b.data, 1)
            ana_correct += (ana_pred_labels == y_b).sum().item()
            
            # Deep target backprop
            # lr=0.2 provides stable EMA updates
            ana_model.backward_target(Y_onehot_b, lr=2.0*(1-0.8*(epoch/epochs)), lr_decay=0.9, use_cross_entropy=True, max_norm=1.0, momentum=0.9)
            
            # --- Baseline Step ---
            optimizer.zero_grad()
            gd_pred_b = gd_model(x_b)
            loss = criterion(gd_pred_b, y_b)
            loss.backward()
            optimizer.step()
            
            gd_running_loss += loss.item() * batch_sz
            _, gd_pred_labels = torch.max(gd_pred_b.data, 1)
            gd_correct += (gd_pred_labels == y_b).sum().item()
            
            total += batch_sz
            
            if batch_idx % 100 == 0:
                print(f"  [Batch {batch_idx}/{len(train_loader)}] AnaLoss: {ana_loss_val.item():.4f} | GDLoss: {loss.item():.4f}", end='\r')
        
        # Epoch metrics
        ana_train_loss = ana_running_loss / total
        gd_train_loss = gd_running_loss / total
        ana_train_acc = ana_correct / total
        gd_train_acc = gd_correct / total
        
        ana_test_loss, ana_test_acc = evaluate(ana_model, test_loader, criterion, device)
        gd_test_loss, gd_test_acc = evaluate(gd_model, test_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        ana_metrics = {'tr_loss': ana_train_loss, 'te_loss': ana_test_loss, 'tr_acc': ana_train_acc, 'te_acc': ana_test_acc}
        bas_metrics = {'tr_loss': gd_train_loss, 'te_loss': gd_test_loss, 'tr_acc': gd_train_acc, 'te_acc': gd_test_acc}
        
        print(" " * 100, end='\r') # clear line
        logger.log_epoch(epoch, ana_metrics, bas_metrics, epoch_time)

if __name__ == "__main__":
    run_tiny_imagenet_experiment(epochs=10, batch_size=64)
