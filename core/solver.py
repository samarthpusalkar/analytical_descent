import torch

def compute_analytical_delta(x_in, error, lam=1e-3, max_norm=1.0, max_samples=None):
    """
    Computes the optimal weight update delta_W analytically.
    
    Args:
        x_in (torch.Tensor): Flattened input tensor of shape [N, D]
        error (torch.Tensor): Flattened error tensor of shape [N, Out] (target - current_output)
        lam (float): Tikhonov regularization lambda (Ridge parameter)
        max_norm (float): Maximum norm for gradient clipping
        max_samples (int): Max rows to randomly subsample (Sketching) to speed up Gram matrix
        
    Returns:
        torch.Tensor: dW of shape [Out, D]
    """
    N, D = x_in.shape
    device = x_in.device
    
    # Auto-Optimization: If the user didn't specify, but BOTH N and D are massively huge 
    # (e.g., massive LLM layers where N=4096 and D=4096), computing the exact covariance 
    # is mathematically impossible in real-time. We automatically apply sketching.
    # CRITICAL: We DO NOT sketch if D is small (like in CNNs where N=12500, D=16).
    # If D=16, X^T X is a tiny 16x16 matrix, which takes 0.001ms to invert using 100% of the data!
    if max_samples is None and min(N, D) > 1024:
        max_samples = 1024
    
    # Ensemble Optimization (Batched GPU Averaging)
    # Instead of a Python for loop, we reshape the tensor into batches and 
    # use PyTorch's native batched matrix multiplication (bmm) and batched 
    # Cholesky decomposition. This computes the entire ensemble in parallel 
    # on the GPU with zero iteration overhead!
    if max_samples is not None and N > max_samples:
        K = N // max_samples
        if K == 0: K = 1
        
        # Truncate to make perfectly divisible chunks
        N_trunc = K * max_samples
        x_c = x_in[:N_trunc].view(K, max_samples, D)
        e_c = error[:N_trunc].view(K, max_samples, -1)
        
        # We know max_samples <= min(N, D), so we use the Woodbury (N < D) form
        # M will be [K, max_samples, max_samples]
        I_batch = torch.eye(max_samples, device=device).unsqueeze(0)
        M = x_c @ x_c.transpose(1, 2) + lam * I_batch
        
        # Batched Cholesky
        L = torch.linalg.cholesky(M)
        
        # Batched Solve
        term = torch.cholesky_solve(e_c, L)
        
        # Batched Multiply
        dW_T_chunked = x_c.transpose(1, 2) @ term
        dW_chunked = dW_T_chunked.transpose(1, 2)
        
        # Average across chunks
        dW = dW_chunked.mean(dim=0)
        
        if max_norm is not None:
            norm = torch.norm(dW)
            if norm > max_norm:
                dW = dW * (max_norm / norm)
        return dW
    
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
