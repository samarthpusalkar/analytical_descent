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
    # complexity from O(D^3) to O(N^3).
    # OPTIMIZATION: We use Cholesky Decomposition instead of standard LU solve. 
    # Since A @ A^T + lam * I is symmetric positive definite (SPD), Cholesky is 
    # exactly 2x faster (1/3 N^3 vs 2/3 N^3) and numerically superior.
    if N < D:
        I_N = torch.eye(N, device=device)
        M = x_in @ x_in.T + lam * I_N
        L = torch.linalg.cholesky(M)
        dW_T = x_in.T @ torch.cholesky_solve(error, L)
    else:
        I_D = torch.eye(D, device=device)
        M = x_in.T @ x_in + lam * I_D
        L = torch.linalg.cholesky(M)
        dW_T = torch.cholesky_solve(x_in.T @ error, L)
        
    dW = dW_T.T
    
    # Optional max-norm clipping to prevent destructive updates from noisy batches
    if max_norm is not None:
        norm = torch.norm(dW)
        if norm > max_norm:
            dW = dW * (max_norm / norm)
            
    return dW
