"""
Module: geometric_harmonics.py

Provides geometric harmonics tools for extending functions (e.g., diffusion coordinates)
from a reference dataset to new data points based on the method from
Coifman & Lafon (2006), https://doi.org/10.1109/TPAMI.2006.223.

Author: Christopher Chaney
Email: christopher.chaney@utsouthwestern.edu

License: MIT License

Dependencies:
- torch
- tqdm
"""

from dataclasses import dataclass
from typing import Optional

import torch


@torch.no_grad()
def _pairwise_sqdist(X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
    if Y is None:
        Y = X
    # (N,d)·(d,M) ? (N,M) distances^2 without large intermediates
    x2 = (X * X).sum(dim=1, keepdim=True)  # (N,1)
    y2 = (Y * Y).sum(dim=1, keepdim=True).t()  # (1,M)
    xy = X @ Y.t()  # (N,M)
    return (x2 + y2 - 2 * xy).clamp_min_(0.0)


@torch.no_grad()
def rbf_kernel(
    X: torch.Tensor, Y: Optional[torch.Tensor], sigma: float
) -> torch.Tensor:
    d2 = _pairwise_sqdist(X, Y)
    return torch.exp(-d2 / (2.0 * (sigma**2)))


@dataclass
class GHModel:
    X_train: torch.Tensor  # (N,d), frozen
    sigma: float  # scalar bandwidth
    tau: float  # ridge
    A: torch.Tensor  # (N,m) coefficients solving (K+tauI)A = Y
    # (optional knobs you might add later: top_k, metric, device)


def pick_sigma_median(X: torch.Tensor, max_samples: int = 5000) -> float:
    """Heuristic ?: sqrt( median(pairwise d^2) / 2 )."""
    N = X.shape[0]
    idx = torch.randperm(N)[: min(N, max_samples)]
    Xs = X[idx]
    d2 = _pairwise_sqdist(Xs)
    tri = torch.triu_indices(d2.shape[0], d2.shape[1], offset=1)
    med = torch.median(d2[tri[0], tri[1]])
    # For exp(-d2 / (2 ?^2)), set 2?^2 ? median(d2) ? ? = sqrt(median/2)
    return float(torch.sqrt(med / 2.0).item())


def fit_geometric_harmonics(
    X_train: torch.Tensor,  # (N,d) WT PCs
    Y_train: torch.Tensor,  # (N,m) WT diffusion coords (already sign-fixed / ?^t scaled)
    sigma: Optional[float] = None,
    tau: float = 1e-3,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> GHModel:
    """
    Solve (K + tau I) A = Y_train once; store A.
    K_ij = exp(-||x_i - x_j||^2 / (2 sigma^2))
    """
    if device is None:
        device = X_train.device
    X = X_train.to(device=device, dtype=dtype, copy=False)
    Y = Y_train.to(device=device, dtype=dtype, copy=False)

    if sigma is None:
        sigma = pick_sigma_median(X)

    K = rbf_kernel(X, None, sigma)  # (N,N)
    # Regularize and Cholesky solve with multiple RHS
    N = K.shape[0]
    K.view(-1)[:: N + 1] += tau  # add tau to diagonal (in-place)
    # Cholesky (symmetric p.d.)
    L = torch.linalg.cholesky(K)  # K = L L^T
    # Solve K A = Y  ?  L Z = Y,  L^T A = Z
    Z = torch.cholesky_solve(Y, L)  # directly: solves K * A = Y
    A = Z  # (N,m)
    return GHModel(
        X_train=X.detach().clone(), sigma=sigma, tau=tau, A=A.detach().clone()
    )


@torch.no_grad()
def apply_geometric_harmonics(
    model: GHModel, X_query: torch.Tensor, batch_size: int = 0
) -> torch.Tensor:
    """
    Evaluate extension: Y_ext = K(X_query, X_train) @ A
    """
    Xq = X_query.to(model.X_train.dtype).to(model.A.device)
    Xt = model.X_train.to(Xq.device)
    A = model.A.to(Xq.device)

    if batch_size and Xq.shape[0] > batch_size:
        out = []
        for i in range(0, Xq.shape[0], batch_size):
            j = min(i + batch_size, Xq.shape[0])
            Kqx = rbf_kernel(Xq[i:j], Xt, model.sigma)  # (b,N)
            out.append(Kqx @ A)  # (b,m)
        return torch.cat(out, dim=0)
    else:
        Kqx = rbf_kernel(Xq, Xt, model.sigma)  # (M,N)
        return Kqx @ A  # (M,m)
