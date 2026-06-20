import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from core.analytical_decorator import analytical_solver

@analytical_solver(lr=1.0, lr_decay=0.9, lam=1e-3, momentum=0.5)
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    print("Initializing model...")
    model = SimpleNet()
    
    print("Checking if hooks are registered...")
    assert hasattr(model, '_analytical_state'), "Decorator failed to attach state!"
    
    # Create fake dataset
    X = torch.randn(100, 10)
    Y = torch.randint(0, 2, (100,))
    loader = DataLoader(TensorDataset(X, Y), batch_size=16)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training loop...")
    for epoch in range(3):
        total_loss = 0
        for x_b, y_b in loader:
            # 1. Forward Pass
            pred = model(x_b)
            
            # Create one-hot target
            # Note: For our target propagation, cross entropy error should be (Softmax(Pred) - Target)
            # The standard PyTorch CrossEntropyLoss backward does exactly this!
            # So if we just use standard CrossEntropyLoss and loss.backward(),
            # the grad_output of the last layer is exactly (Softmax(Pred) - Target) / batch_size
            loss = criterion(pred, y_b)
            
            # 2. Backward
            optimizer.zero_grad()
            loss.backward()
            
            # 3. Step
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")
        
    print("Training loop completed successfully!")

if __name__ == "__main__":
    main()
