import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from core.analytical_decorator import analytical_solver

@analytical_solver(lr=1.0, lr_decay=0.8, lam=1e-3, momentum=0.5)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def main():
    print("Initializing CNN model...")
    model = SimpleCNN()
    
    print("Checking if hooks are registered...")
    assert hasattr(model, '_analytical_state'), "Decorator failed to attach state!"
    
    # Create fake dataset (e.g. MNIST 28x28)
    X = torch.randn(64, 1, 28, 28)
    Y = torch.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(X, Y), batch_size=16)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting CNN training loop...")
    for epoch in range(2):
        total_loss = 0
        for x_b, y_b in loader:
            pred = model(x_b)
            loss = criterion(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")
        
    print("CNN Training loop completed successfully!")

if __name__ == "__main__":
    main()
