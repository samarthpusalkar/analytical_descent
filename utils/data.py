import torch
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

def get_digits_dataset():
    """
    Loads the scikit-learn digits dataset.
    Returns:
        X_train, X_test, y_train, y_test, Y_train_onehot, Y_test_onehot
    """
    data = load_digits()
    X, y = data.data, data.target
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Create one-hot targets for analytical MSE training
    # For a stable analytical target, we can scale one-hot to [0.1, 0.9] or just use [0, 1]
    Y_train_onehot = F.one_hot(y_train, num_classes=10).float()
    Y_test_onehot = F.one_hot(y_test, num_classes=10).float()
    
    return X_train, X_test, y_train, y_test, Y_train_onehot, Y_test_onehot
