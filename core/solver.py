import torch

def compute_analytical_delta(x_in, error, lam=1e-3, max_norm=1.0):
    """
    Computes the optimal weight update delta_W analytically.
    
    Args:
        x_in (torch.Tensor): Flattened input tensor of shape [N, D]
        error (torch.Tensor): Flattened error tensor of shape [N, Out] (target - current_output)
        lam (float): Tikhonov regularization lambda (Ridge parameter)
        max_norm (float): Maximum norm for gradient clipping
        
    Returns:
        torch.Tensor: dW of shape [Out, D]
    """
    N, D = x_in.shape
    device = x_in.device
    
    # Woodbury Matrix Identity for N < D (Small Batch, Large Feature Space)
    # This mathematically guarantees the same pseudo-inverse but reduces 
    # complexity from O(D^3) to O(N^3), preserving the exact null-space dimensions.
    if N < D:
        I_N = torch.eye(N, device=device)
        dW_T = x_in.T @ torch.linalg.solve(x_in @ x_in.T + lam * I_N, error)
    else:
        I_D = torch.eye(D, device=device)
        dW_T = torch.linalg.solve(x_in.T @ x_in + lam * I_D, x_in.T @ error)
        
    dW = dW_T.T
    
    # Optional max-norm clipping to prevent destructive updates from noisy batches
    if max_norm is not None:
        norm = torch.norm(dW)
        if norm > max_norm:
            dW = dW * (max_norm / norm)
            
    return dW
