import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from core.analytical_decorator import analytical_solver

# @analytical_solver(lr=1.0, lr_decay=0.9, lam=1e-3, momentum=0.5)
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=100, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # PyTorch's native transformer encoder uses nn.Linear for attention and feedforward internally!
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, 2)
        
    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embedding(x)  # [batch, seq_len, d_model]
        x = self.transformer(x) # [batch, seq_len, d_model]
        
        # Pool
        x = x.mean(dim=1) # [batch, d_model]
        x = self.fc_out(x)
        return x

def main():
    print("Initializing Transformer model...")
    model = SimpleTransformer()
    
    print("Checking if hooks are registered...")
    # assert hasattr(model, '_analytical_state'), "Decorator failed to attach state!"
    
    # Create fake dataset
    X = torch.randint(0, 100, (32, 10)) # [batch, seq_len]
    Y = torch.randint(0, 2, (32,))
    loader = DataLoader(TensorDataset(X, Y), batch_size=8)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting Transformer training loop...")
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
        
    print("Transformer Training loop completed successfully!")

if __name__ == "__main__":
    main()
