# Integration Guide: Analytical Target Propagation

This guide explains how to integrate the Analytical Target Propagation (solver-based loss calculator) into your existing PyTorch machine learning pipeline without breaking the rest of your codebase.

## The Recommended Approach: `@analytical_solver` Decorator

The easiest and most robust way to integrate Analytical Target Propagation is to use the `@analytical_solver` decorator. This approach is completely "plug-and-play" and works with **any architecture**—including CNNs, LSTMs, and even Hugging Face Transformers.

It automatically intercepts standard PyTorch gradients and solves for optimal weight updates behind the scenes.

**Before:**
```python
import torch.nn as nn
from transformers import BertModel # Example complex model

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, 2)
```

**After:**
```python
import torch.nn as nn
from transformers import BertModel
from core.analytical_decorator import analytical_solver

# 1. Simply add the decorator!
@analytical_solver(lr=1.0, lr_decay=0.9, momentum=0.5, max_norm=10.0)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, 2)
```

### How it Works (Hybrid Training)
When you call `loss.backward()` and `optimizer.step()`:
1. The analytical solver dynamically intercepts PyTorch's gradients at every `nn.Linear` and `nn.Conv2d` layer.
2. It solves and updates those specific layers perfectly using Woodbury Ridge Regression.
3. It hides those gradients from your standard optimizer (like `Adam`).
4. Complex, non-linear components (like LayerNorms, Embeddings, or custom activations) that cannot be solved analytically are simply left alone. Your `Adam` optimizer will update them normally!

---

## Alternative Approach: Manual Layer Swapping

If you prefer explicit control over the network definition instead of using the decorator, you can manually swap PyTorch layers for their analytical equivalents.

To integrate this, you only need to modify **two parts** of your codebase:
1. The model definition (swapping standard PyTorch layers for Analytical layers).
2. The training step (replacing standard backprop with `backward_target`).

Everything else—DataLoaders, logging, evaluation loops, and even standard loss calculation for tracking—remains exactly the same.

---

## Step 1: Updating the Model Definition

You need to replace your standard PyTorch `nn.Sequential` and related layers with their `Analytical` equivalents. The Analytical layers act as drop-in replacements for standard layers.

**Before (Standard PyTorch):**
```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 10)
        )

    def forward(self, x):
        return self.net(x)
```

**After (Analytical Integration):**
```python
import torch.nn as nn
from models.analytical_network import AnalyticalSequential
from layers.analytical_linear import AnalyticalLinear
from layers.analytical_conv import AnalyticalConv2d
from layers.analytical_flatten import AnalyticalFlatten
from core.inversions import AnalyticalLeakyReLU

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Swap nn.Sequential for AnalyticalSequential
        self.net = AnalyticalSequential(
            AnalyticalConv2d(1, 16, kernel_size=4, stride=2, padding=1),
            AnalyticalLeakyReLU(0.01),
            AnalyticalFlatten(),
            AnalyticalLinear(16 * 14 * 14, 10)
        )

    def forward(self, x):
        return self.net(x)
```

> [!TIP]
> The Analytical layers share the exact same initialization parameters as standard PyTorch layers (e.g., `in_channels`, `out_channels`, `kernel_size`), making the transition seamless.

---

## Step 2: Modifying the Training Loop

In your training loop, you will replace the traditional Backpropagation (`loss.backward()`) and Optimizer Step (`optimizer.step()`) with the Analytical Network's built-in target solver.

You will also need to convert your target labels into **one-hot encoded targets** so the network can compute the layer-wise error properly.

**Before (Standard PyTorch):**
```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for x_b, y_b in train_loader:
    # 1. Forward Pass
    predictions = model(x_b)
    loss = criterion(predictions, y_b)
    
    # 2. Backward & Update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**After (Analytical Integration):**
```python
criterion = nn.CrossEntropyLoss()
# No optimizer needed! The model updates itself analytically.

for x_b, y_b in train_loader:
    # 1. Forward Pass
    predictions = model(x_b)
    
    # Optional: You can still calculate standard loss for tracking/logging!
    loss = criterion(predictions, y_b) 
    
    # 2. Prepare Target Data (Must be one-hot encoded for classification)
    batch_size = x_b.size(0)
    num_classes = 10
    Y_onehot_b = torch.zeros(batch_size, num_classes, device=x_b.device)
    Y_onehot_b.scatter_(1, y_b.unsqueeze(1), 1.0)
    
    # 3. Analytical Backward & Update
    model.net.backward_target(
        Y_onehot_b, 
        lr=1.0,               # Analytical learning rate (often higher than GD, e.g., 0.5 to 2.0)
        lr_decay=0.5,         # Layer-wise LR decay to stabilize deeper layers
        use_cross_entropy=True, # Set to True for classification, False for MSE
        momentum=0.5,         # Regularization momentum
        max_norm=10.0         # Gradient clipping norm
    )
```

> [!NOTE]
> Because `backward_target` calculates weight updates and applies them immediately within the layer, you do **not** need a PyTorch `Optimizer` (like `Adam` or `SGD`).

---

## Step 3: Evaluation (No Changes Required)

The best part about this integration is that your evaluation logic does not need to change at all. `model.eval()` works exactly the same, and forward passes return standard logits just like a standard PyTorch model.

```python
# Unchanged Evaluation Loop!
model.eval()
correct = 0
with torch.no_grad():
    for x_b, y_b in test_loader:
        predictions = model(x_b)
        _, pred_labels = torch.max(predictions.data, 1)
        correct += (pred_labels == y_b).sum().item()
```

---

## Summary of Parameters in `backward_target`

When calling `backward_target(final_target, ...)`, tune the following hyperparameters:

- **`lr`**: The base analytical learning rate. Often `1.0` or higher, unlike Adam which uses `0.001`.
- **`lr_decay`**: Scaling factor for the learning rate as error propagates backward through the network (prevents quadratic error compounding in deep layers). Recommended: `0.5` to `0.9`.
- **`use_cross_entropy`**: `True` uses Cross-Entropy derivative w.r.t logits (`target - softmax`). `False` uses clamped MSE (Huber loss derivative).
- **`momentum`**: Controls the EMA of the covariance matrix for Ridge Regression.
- **`max_norm`**: Clips the maximum norm of the analytically computed weight updates.

By keeping your dataloaders, metrics, and tracking the same, this setup allows for rapid A/B testing against your standard Gradient Descent baselines!
