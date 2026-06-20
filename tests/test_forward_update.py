import torch
import torch.nn as nn
from utils.data import get_digits_dataset
from core.inversions import InverseLeakyReLU

X_train, X_test, y_train, y_test, Y_train_onehot, Y_test_onehot = get_digits_dataset()

class CustomLayer:
    def __init__(self, in_f, out_f):
        self.W = torch.randn(out_f, in_f) * 0.1
        self.b = torch.randn(out_f) * 0.1
        
    def forward(self, x):
        return x @ self.W.T + self.b
        
    def get_input_target(self, y_target):
        y_centered = y_target - self.b
        W_pinv = torch.linalg.pinv(self.W.T)
        return y_centered @ W_pinv
        
    def update(self, x, y_target, lr):
        ones = torch.ones(x.size(0), 1)
        x_aug = torch.cat([x, ones], dim=1)
        W_aug_T = torch.linalg.lstsq(x_aug, y_target).solution
        W_opt = W_aug_T.T
        dW = W_opt[:, :-1] - self.W
        db = W_opt[:, -1] - self.b
        self.W += lr * dW
        self.b += lr * db

l1 = CustomLayer(64, 32)
act1_inv = InverseLeakyReLU(0.01)
l2 = CustomLayer(32, 32)
act2_inv = InverseLeakyReLU(0.01)
l3 = CustomLayer(32, 10)

def act(x):
    return torch.nn.functional.leaky_relu(x, 0.01)

for epoch in range(15):
    # Forward pass just to get initial loss
    h1 = act(l1.forward(X_train))
    h2 = act(l2.forward(h1))
    out = l3.forward(h2)
    loss = nn.MSELoss()(out, Y_train_onehot).item()
    print(f"Epoch {epoch}: Loss {loss:.4f}")
    
    # Phase 1: Backward target propagation
    t3 = Y_train_onehot
    t2_post_act = l3.get_input_target(t3)
    t2_pre_act = act2_inv(t2_post_act)
    t1_post_act = l2.get_input_target(t2_pre_act)
    t1_pre_act = act1_inv(t1_post_act)
    
    # Phase 2: Forward update
    # Update L1
    l1.update(X_train, t1_pre_act, lr=0.1)
    new_h1 = act(l1.forward(X_train))
    
    # Update L2
    l2.update(new_h1, t2_pre_act, lr=0.1)
    new_h2 = act(l2.forward(new_h1))
    
    # Update L3
    l3.update(new_h2, t3, lr=0.1)
