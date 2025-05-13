import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader
from sklearn.neighbors import KernelDensity

def KDE_custom(x, centers, bandwidth):
    B, d = x.shape
    n = centers.shape[0]
    x_exp = x.unsqueeze(1)
    X_exp = centers.unsqueeze(0)
    sq_dist = ((x_exp - X_exp) ** 2).sum(dim=2)
    log_kernel = -0.5 * sq_dist / bandwidth**2
    log_gauss_norm = -(d / 2) * torch.log(
        torch.tensor(2 * np.pi * bandwidth**2, dtype=x.dtype, device=x.device)
    )
    log_kernel = log_kernel + log_gauss_norm
    log_terms = log_kernel
    log_density = torch.logsumexp(log_terms, dim=1) - np.log(n)
    return log_density
    
def KDE(x, centers, bandwidth):
    kde = KernelDensity(kernel='gaussian', bandwidth= bandwidth)
    kde.fit(centers)
    log_density = kde.score_samples(x)
    return log_density

# Create test data
np.random.seed(0)
torch.manual_seed(0)

centers_np = np.random.randn(100, 3)
x_np = np.random.randn(10, 3)
bandwidth = 0.5

# Compute using sklearn
log_density_sklearn = KDE(x_np, centers_np, bandwidth)

# Compute using PyTorch
x_torch = torch.tensor(x_np, dtype=torch.float32)
centers_torch = torch.tensor(centers_np, dtype=torch.float32)
log_density_torch = KDE_custom(x_torch, centers_torch, bandwidth).numpy()

# Compare
print("Are outputs close?", np.allclose(log_density_sklearn, log_density_torch, atol=1e-4))
print("\nSklearn log-density:\n", log_density_sklearn)
print("\nPyTorch log-density:\n", log_density_torch)