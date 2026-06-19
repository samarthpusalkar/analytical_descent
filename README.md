# Deep Analytical Learning: Forward-First Neural Networks

This repository explores an alternative approach to training deep neural networks: **Forward-First Analytical Learning**. Instead of using standard End-to-End Backpropagation (Gradient Descent) to update weights iteratively, this framework trains layers sequentially by treating each layer's input and target as a localized Least-Squares problem and solving for exact weight updates using **Delta-Regularized Ridge Regression**.

## The Premise

Standard Gradient Descent calculates a gradient (a slope) and takes a small step in that direction. The Analytical approach calculates the exact mathematical distance to the local minimum and leaps directly there. 

By propagating "target values" backward through inverse activation functions (Target Propagation) and solving for weight updates analytically at each layer, the model can theoretically achieve incredibly fast convergence. This repository contains the custom PyTorch layers, mathematical solvers, and benchmarking experiments used to stabilize and evaluate this concept against traditional Gradient Descent baselines.

## Key Breakthroughs & Findings

Achieving parity with Gradient Descent required solving several mathematical instabilities inherent in analytical updates:
- **Catastrophic Weight Erasure:** Solved via *Delta-Regularized Ridge Regression*, minimizing the change in weights (Random-Walk Prior) rather than their absolute value (Zero-Mean Prior).
- **Quadratic Error Compounding:** Solved by strictly adhering to the Chain Rule and using pre-update weights to back-calculate targets for previous layers.
- **Null-Space Memory Preservation:** Discovered that smaller batch sizes actively protect previously learned global features. The Delta-Regularization forces updates in the undefined dimensions (null-space) to be zero, preventing the network from aggressively overwriting its memory to fit localized batch noise.

*Read the full mathematical breakdown and evolution in [experimentation_analysis.md](./experimentation_analysis.md).*

## Repository Structure

```text
.
├── core/
│   ├── inversions.py         # Inverse activation functions (LeakyReLU, etc.)
│   └── solver.py             # Delta-Regularized Ridge Regression solver
├── layers/
│   ├── analytical_conv.py    # Analytical Convolutional layer implementation
│   ├── analytical_flatten.py # Forward/Backward flattening logic
│   └── analytical_linear.py  # Custom nn.Linear with analytical update logic
├── models/
│   └── analytical_network.py # Custom Sequential model handling target propagation
├── experiments/
│   ├── classification/       # CNN & MLP experiments (e.g., MNIST, CIFAR10, FashionMNIST)
│   └── regression/           # Tabular and synthetic regression experiments
├── utils/
│   ├── data.py               # Data loading utilities
│   ├── logger.py             # Benchmarking loggers
│   └── metrics.py            # Custom metric functions
├── plot_results.py           # Utility to generate performance comparison plots
└── experimentation_analysis.md # Detailed write-up of mathematical fixes and results
```

## Setup & Installation

This project is built on standard deep learning and data science tools.
1. Clone the repository
2. Install the necessary dependencies:
   ```bash
   pip install torch torchvision pandas matplotlib seaborn scikit-learn
   ```

## Running Experiments

You can run the various benchmark scripts to compare the Analytical Network directly against a Gradient Descent baseline (using identical architectures).

**Classification Experiments (MLP & CNN):**
```bash
python experiments/classification/vision_mlp/exp_mnist_small.py
python experiments/classification/vision_mlp/exp_mnist_large.py
python experiments/classification/vision_mlp/exp_cifar10_mlp.py
python experiments/classification/vision_cnn/exp_fashion_mnist_cnn.py
```

**Regression Experiments:**
```bash
python experiments/regression/synthetic/exp_sine_wave.py
python experiments/regression/tabular/exp_california_housing.py
```

Training metrics (Loss, Accuracy, F1-Score) are automatically saved to CSV files in the root directory. You can visualize the results by running:
```bash
python plot_results.py
```
This will read the CSV outputs and generate performance comparison graphs in the `plots/` directory.

## Results Summary

When properly tuned—using Delta-Regularization, Huber-bounded errors, and null-space preservation via batch sizing—the Deep Analytical Model successfully matches the performance of standard Gradient Descent (Adam/SGD). It reaches comparable accuracies (e.g., 97% on MNIST) with fewer required epoch steps, though it incurs a higher computational cost per step due to the pseudo-inverse matrix calculations.

---
