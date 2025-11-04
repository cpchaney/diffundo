import numpy as np
import scipy.sparse as sp
import torch
from diffundo.utils.kernel import (get_anisotropic_kernel, get_distance_matrix,
                                   get_sigma)
from diffundo.utils.sparse import keep_topk_elements
from scipy.sparse.linalg import eigsh
from torch import Tensor


# --- small helper: diffusion map embedding from symmetric op eigpairs ---
def _diffusion_map_from_A(evals, evecs, d_alpha_sqrt, t: int, m: int):
    """
    evals: (r,) eigenvalues of A sorted descending (?0?1 ? ?1 ? ...).
    evecs: (N,r) eigenvectors of A (orthonormal).
    d_alpha_sqrt: (N,) = sqrt(d_alpha) used to map back to right eigenvectors of P_alpha
    Returns Y: (N, m) diffusion coordinates using components 1..m scaled by ?^t.
    """
    # Right eigenvectors of P_alpha: ?_j = D_alpha^{-1/2} ?_j  (?_j are columns of evecs)
    phi = evecs / d_alpha_sqrt[:, None]  # (N,r)
    lambdas = evals
    # drop trivial component j=0
    lambdas = lambdas[1 : m + 1]
    phi = phi[:, 1 : m + 1]
    Y = phi * (lambdas[None, :] ** t)
    return Y, lambdas, phi  # Y used for downstream, phi are right eigenvectors at t=0


def construct_diffusion_map(
    features: torch.Tensor,
    k: int = 8,  # bandwidth estimator if your utils use it
    epsilon: float = 1.0,  # RBF bandwidth (global)
    top_k: int = 64,  # kNN sparsification
    alpha: float = 1.0,  # 1.0 = Fokker?Planck; 0.5 = Coifman?Lafon common choice
    t: int = 1,  # diffusion time
    n_eigs: int = 51,  # how many eigenpairs to compute (first is trivial)
):
    """
    Returns:
      P_alpha: row-stochastic Markov matrix (scipy.sparse CSR)
      Y: diffusion coordinates (numpy array, N x m) with m=n_eigs-1
      evals: eigenvalues of A (numpy, length n_eigs)
      evecs: eigenvectors of A (numpy, N x n_eigs)
      d_alpha: degree vector after alpha-normalization (numpy, N)
    """
    # ---- Torch -> Numpy once ----
    X = features.detach().cpu().numpy().astype(np.float32)  # (N,d)
    N = X.shape[0]

    # === Build kernel with your utilities ===
    # If your get_distance_matrix etc. expect torch, call them first then convert to scipy.sparse
    D_torch = get_distance_matrix(torch.from_numpy(X))  # (N,N) torch (dense or sparse)
    sigma = get_sigma(D_torch, k)  # (N,) torch
    K_torch = get_anisotropic_kernel(
        D_torch, sigma, epsilon
    )  # kernel K (torch, likely dense)
    K_torch = keep_topk_elements(K_torch, top_k)  # sparsify (torch sparse)
    # Convert to scipy.sparse CSR
    if K_torch.is_sparse:
        K = sp.csr_matrix(K_torch.to_dense().cpu().numpy())
    else:
        K = sp.csr_matrix(K_torch.cpu().numpy())
    # Ensure symmetry after kNN truncation (take max to symmetrize)
    K = K.maximum(K.T)

    # === Alpha normalization: K_alpha = D^{-alpha} K D^{-alpha} ===
    d = np.asarray(K.sum(axis=1)).ravel() + 1e-12
    Dm_alpha_left = sp.diags(d ** (-alpha))
    K_alpha = Dm_alpha_left @ K @ Dm_alpha_left

    # Row sums after alpha norm
    d_alpha = np.asarray(K_alpha.sum(axis=1)).ravel() + 1e-12
    D_alpha_inv = sp.diags(1.0 / d_alpha)
    # Markov operator P_alpha = D_alpha^{-1} K_alpha  (row-stochastic)
    P_alpha = D_alpha_inv @ K_alpha

    # === Symmetric operator A = D_alpha^{1/2} P_alpha D_alpha^{-1/2} = D_alpha^{-1/2} K_alpha D_alpha^{-1/2}
    D_alpha_mh = sp.diags(1.0 / np.sqrt(d_alpha))
    A = D_alpha_mh @ K_alpha @ D_alpha_mh  # symmetric PSD

    # === Eigen-decomposition of A (largest eigenvalues) ===
    r = min(n_eigs, N)
    evals, evecs = eigsh(
        A, k=r, which="LA"
    )  # returns ascending? eigsh with 'LA' returns largest algebraic
    # Sort descending to have ?0?1 first
    ord_desc = np.argsort(evals)[::-1]
    evals = evals[ord_desc]
    evecs = evecs[:, ord_desc]

    # === Diffusion map coordinates (skip trivial comp) ===
    d_alpha_sqrt = np.sqrt(d_alpha)
    m = r - 1
    Y, lambdas, phi = _diffusion_map_from_A(evals, evecs, d_alpha_sqrt, t=t, m=m)

    return (
        P_alpha.tocsr(),
        Y.astype(np.float32),
        evals.astype(np.float32),
        evecs.astype(np.float32),
        d_alpha,
    )


def get_diffusion_map(weights: Tensor, features: Tensor, t: int = 1) -> Tensor:
    diffusion_map = (1 - weights) ** t
    return (torch.diag(diffusion_map[1:]) @ features[:, 1:].T).T


def diffuse_spectrum(
    transition_matrix: Tensor,
    raw_spectrum: Tensor,
    max_iter: int = 8,
    min_corr: float = 0.95,
) -> Tensor:
    """
    Apply diffusion-based smoothing to a feature matrix using a transition matrix.
    Automatically stops when correlation stabilizes or max_iter is reached.

    Args:
        transition_matrix (Tensor): Shape (M, N) ? can be square or rectangular.
        raw_spectrum (Tensor): Shape (N, D) ? features to diffuse.
        max_iter (int): Max number of diffusion steps.
        min_corr (float): Minimum correlation coefficient for convergence.

    Returns:
        Tensor: Smoothed feature matrix of shape (M, D)
    """
    current = transition_matrix @ raw_spectrum  # Shape: (M, D)
    previous = current.clone()

    for _ in range(1, max_iter):
        current = transition_matrix @ raw_spectrum  # Always diffuse from raw
        # Compute correlation between current and previous
        corr_num = torch.sum((previous - previous.mean()) * (current - current.mean()))
        corr_den = torch.sqrt(
            torch.sum((previous - previous.mean()) ** 2)
        ) * torch.sqrt(torch.sum((current - current.mean()) ** 2))
        corr = corr_num / (corr_den + 1e-10)

        if corr**2 >= min_corr:
            break

        previous = current

    # Normalize to match original scale
    raw_quantile = torch.quantile(raw_spectrum, q=0.99, dim=0)
    diff_max = current.max(dim=0)[0]
    safe_diff_max = torch.where(
        diff_max == 0, torch.tensor(1e-10, device=diff_max.device), diff_max
    )
    norm_factor = raw_quantile / safe_diff_max

    return current * norm_factor[None, :]
