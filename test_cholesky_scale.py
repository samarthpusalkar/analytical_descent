import torch

# Simulate massive x_in resulting from weight drift
x_in = torch.randn(128, 64) * 1000.0  # variance 1,000,000
M = x_in @ x_in.T  # [128, 128]

# Standard lam
lam_standard = 1e-3
M_std = M + lam_standard * torch.eye(128)

try:
    L1 = torch.linalg.cholesky(M_std)
    print("Standard Cholesky: SUCCESS")
except Exception as e:
    print("Standard Cholesky: FAILED -", str(e))

# Scale-invariant lam
scale = M.diagonal().mean()
lam_scaled = 1e-3 * scale
M_scaled = M + lam_scaled * torch.eye(128)

try:
    L2 = torch.linalg.cholesky(M_scaled)
    print("Scale-Invariant Cholesky: SUCCESS")
except Exception as e:
    print("Scale-Invariant Cholesky: FAILED -", str(e))

