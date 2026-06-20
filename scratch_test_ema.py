import sys
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from models.analytical_network import AnalyticalSequential
from layers.analytical_conv import AnalyticalConv2d
from layers.analytical_linear import AnalyticalLinear
from layers.analytical_flatten import AnalyticalFlatten
from core.inversions import AnalyticalLeakyReLU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
data_dir = os.path.join("data", "tiny-imagenet-200")
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

ana_model = AnalyticalSequential(
    AnalyticalConv2d(3, 32, kernel_size=3, stride=2, padding=1), AnalyticalLeakyReLU(0.01),
    AnalyticalConv2d(32, 64, kernel_size=3, stride=2, padding=1), AnalyticalLeakyReLU(0.01),
    AnalyticalConv2d(64, 128, kernel_size=3, stride=2, padding=1), AnalyticalLeakyReLU(0.01),
    AnalyticalFlatten(),
    AnalyticalLinear(128 * 8 * 8, 512), AnalyticalLeakyReLU(0.01),
    AnalyticalLinear(512, 200)
).to(device)

# Monkey-patch EMA into Linear
original_linear_update = AnalyticalLinear.solve_and_update
def ema_linear_update(self, x_actual, y_target, lr=1.0, max_norm=1.0):
    # Hijack lr to just compute dW but not apply it
    original_weight = self.weight.data.clone()
    original_linear_update(self, x_actual, y_target, lr=1.0, max_norm=max_norm)
    dW = self.weight.data - original_weight
    self.weight.data = original_weight
    
    if not hasattr(self, 'dW_ema'):
        self.dW_ema = torch.zeros_like(dW)
        if self.bias is not None:
            self.db_ema = torch.zeros_like(self.bias.data)
            
    self.dW_ema = 0.9 * self.dW_ema + 0.1 * dW
    self.weight.data += lr * self.dW_ema
AnalyticalLinear.solve_and_update = ema_linear_update

criterion = nn.CrossEntropyLoss()

print("Testing Analytical with EMA Momentum")
for batch_idx, (x_b, y_b) in enumerate(train_loader):
    x_b, y_b = x_b.to(device), y_b.to(device)
    Y_onehot_b = torch.zeros(x_b.size(0), 200, device=device)
    Y_onehot_b.scatter_(1, y_b.unsqueeze(1), 1.0)
    
    ana_pred = ana_model(x_b)
    loss = criterion(ana_pred, y_b)
    
    ana_model.backward_target(Y_onehot_b, lr=5.0, lr_decay=1.0, use_cross_entropy=True, max_norm=1.0)
    
    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
    if batch_idx == 50:
        break
