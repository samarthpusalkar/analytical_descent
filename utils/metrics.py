import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

def calculate_classification_metrics(y_pred_logits, y_true):
    """
    Calculates Loss (MSE against one-hot), Accuracy, and F1 Score.
    
    Args:
        y_pred_logits: Tensor of shape (batch_size, num_classes)
        y_true: Tensor of shape (batch_size,) containing class indices
    """
    # For fair comparison with analytical (which uses MSE on one-hot),
    # we compute the MSE loss on the one-hot representations.
    num_classes = y_pred_logits.shape[1]
    y_true_onehot = F.one_hot(y_true, num_classes=num_classes).float()
    
    loss = F.mse_loss(y_pred_logits, y_true_onehot).item()
    
    # Calculate predictions (argmax)
    predictions = torch.argmax(y_pred_logits, dim=1).numpy()
    targets = y_true.numpy()
    
    acc = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='macro')
    
    return loss, acc, f1
