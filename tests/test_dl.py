import torch
import torch.nn as nn
import torch.optim as optim
from utils.data import get_digits_dataset
from utils.metrics import calculate_classification_metrics

X_train, X_test, y_train, y_test, Y_train_onehot, Y_test_onehot = get_digits_dataset()

model = nn.Sequential(nn.Linear(64, 10))
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

for epoch in range(101):
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        _, acc, _ = calculate_classification_metrics(pred.detach(), y_train)
        print(f"Epoch {epoch}: Acc {acc:.4f}")

_, acc, _ = calculate_classification_metrics(model(X_test).detach(), y_test)
print(f"Final Test Acc: {acc:.4f}")
