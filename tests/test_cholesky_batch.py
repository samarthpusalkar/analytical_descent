import torch

K, max_samples, D = 4, 1024, 16
x_c = torch.randn(K, max_samples, D) * 1000.0

M_raw = x_c @ x_c.transpose(1, 2)
scale = M_raw.diagonal(dim1=-2, dim2=-1).mean(dim=-1).view(K, 1, 1).clamp(min=1.0)

I_batch = torch.eye(max_samples).unsqueeze(0)
M = M_raw + (1e-3 * scale) * I_batch

try:
    L = torch.linalg.cholesky(M)
    print("Batched Scale-Invariant Cholesky: SUCCESS")
except Exception as e:
    print("Batched Scale-Invariant Cholesky: FAILED -", str(e))
