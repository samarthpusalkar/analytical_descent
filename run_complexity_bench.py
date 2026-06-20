import torch
import time
from core.solver import compute_analytical_delta

def benchmark_solver(method_name, N, D, approx_method=None, max_samples=None, force_exact=False, runs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_in = torch.randn(N, D, device=device)
    error = torch.randn(N, 10, device=device)
    
    start = time.time()
    for _ in range(runs):
        if force_exact:
            lam = 1e-3
            I_D = torch.eye(D, device=device)
            M = x_in.T @ x_in + lam * I_D
            L = torch.linalg.cholesky(M)
            dW_T = torch.cholesky_solve(x_in.T @ error, L)
        else:
            _ = compute_analytical_delta(x_in, error, max_samples=max_samples)
    end = time.time()
    
    avg_time = (end - start) / runs
    print(f"{method_name:<40} | N={N:<5} D={D:<4} | {avg_time*1000:>8.2f} ms")

print("\n--- Massive N == D (Simulating Massive LLM Layer) ---")
N, D = 4096, 4096
benchmark_solver("Exact Cholesky O(N^3)", N, D, force_exact=True)
benchmark_solver("Batched Tensor Ensemble (Chunk=1024)", N, D, max_samples=1024)

