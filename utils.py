"""
utils.py

This module provides utilities for computing distance-based kernels,
normalizing them, and applying spectral diffusion techniques on feature data.

It includes:

- Pairwise distance computation
- Local scaling using nearest neighbors
- Interference and anisotropic kernel constructions
- Kernel normalization and asymmetric transition matrix calculation
- Diffusion of feature spectra with convergence based on correlation

All functions use PyTorch tensors for efficient GPU/CPU compatibility.

Typical workflow:
1. Compute distance matrix from feature points
2. Estimate local sigma using k-th nearest neighbor
3. Construct a kernel (e.g., interference or anisotropic)
4. Normalize the kernel and create a transition matrix
5. Use the transition matrix to diffuse feature spectra

Dependencies:
- torch (PyTorch)

Author: Christopher Chaney
"""

import torch


def keep_topk_elements(tensor, k):
    _, topk_indices = torch.topk(tensor, k=k, dim=1)
    mask = torch.zeros_like(tensor, dtype=torch.bool)
    mask.scatter_(1, topk_indices, True)
    return tensor * mask


def get_distance_matrix(input_tensor):
    """
    Computes the pairwise Euclidean distance matrix for a set of input points.

    Args:
        input_tensor (torch.Tensor): Tensor of shape [num_points, num_dimensions],
            where each row represents a point in space.

    Returns:
        torch.Tensor: Distance matrix of shape [num_points, num_points],
            where entry (i, j) is the Euclidean distance between point i and j.
    """
    return torch.cdist(input_tensor, input_tensor)


def get_sigma(distance_matrix, k=8):
    """
    Estimates local bandwidth (sigma) for each point based on k-th nearest neighbor distance.

    Args:
        distance_matrix (torch.Tensor): Pairwise distance matrix of shape [N, N].
        k (int): Index of the neighbor to use for sigma estimation (1-based). Default is 8.

    Returns:
        torch.Tensor: Sigma values of shape [N], where each value corresponds to the distance
            to the k-th nearest neighbor of a point.
    """
    return torch.kthvalue(distance_matrix, k + 1).values


def get_interference_kernel(distance_matrix, sigma):
    """
    Computes the interference kernel based on local scaling (sigma).

    Args:
        distance_matrix (torch.Tensor): Distance matrix of shape [N, N].
        sigma (torch.Tensor): Tensor of shape [N], containing local bandwidth values.

    Returns:
        torch.Tensor: Interference kernel matrix of shape [N, N], where higher values
            represent stronger local interactions based on distances and bandwidth.
    """
    P = sigma.outer(sigma)
    S = (sigma**2)[:, None] + (sigma**2)[None, :]

    interference_kernel = torch.sqrt(2 * P / S) * torch.exp(
        -(distance_matrix**2) / (2 * S)
    )

    return interference_kernel


def get_anisotropic_kernel(distance_matrix, sigma, epsilon=1.0):
    """
    Computes the anisotropic kernel with adjustable diffusion scaling.

    Args:
        distance_matrix (torch.Tensor): Distance matrix of shape [N, N].
        sigma (torch.Tensor): Bandwidth tensor of shape [N].
        epsilon (float): Scaling parameter that adjusts global diffusion strength.

    Returns:
        torch.Tensor: Anisotropic kernel matrix of shape [N, N].
    """
    S = sigma.outer(sigma)
    return torch.exp(-(distance_matrix**2) / (epsilon * S))


def get_normalized_kernel(kernel, alpha=0.5):
    """
    Normalizes the kernel matrix using a degree-based normalization.

    Args:
        kernel (torch.Tensor): Kernel matrix of shape [N, N].
        alpha (float): Exponent for normalization. alpha=0.5 gives symmetric normalization.

    Returns:
        torch.Tensor: Normalized kernel matrix of shape [N, N].
    """
    row_sum = kernel.sum(dim=1)
    normalized_kernel = kernel / torch.outer(row_sum**alpha, row_sum**alpha)
    return normalized_kernel


def get_asymmetric_transition_matrix(W):
    """
    Converts a kernel or affinity matrix into an asymmetric transition matrix.

    Args:
        W (torch.Tensor): Weighted adjacency matrix of shape [N, N].

    Returns:
        torch.Tensor: Asymmetric transition matrix of shape [N, N],
            where each row sums to 1.
    """
    return W / W.sum(dim=1, keepdim=True)


def get_diffusion_map(weights, features, t=1):
    """
    Computes a diffusion map by diffusing features over a transition weight matrix.

    Args:
        weights (torch.Tensor): Weight vector of shape [N], used for scaling.
        features (torch.Tensor): Feature matrix of shape [N, D].
        t (int): Diffusion time (controls the strength of diffusion). Default is 1.

    Returns:
        torch.Tensor: Diffused feature representation of shape [N, D-1],
            excluding the first dimension.
    """
    diffusion_map = (1 - weights) ** t
    return (torch.diag(diffusion_map[1:]) @ features[:, 1:].T).T


def diffuse_spectrum(asymmetric_transition_matrix, raw_spectrum):
    """
    Iteratively applies diffusion to a signal until it stabilizes or a correlation threshold is met.

    Args:
        asymmetric_transition_matrix (torch.Tensor): Transition matrix of shape [N, N].
        raw_spectrum (torch.Tensor): Initial signal to be diffused, of shape [N, D].

    Returns:
        torch.Tensor: Diffused spectrum with the same shape as `raw_spectrum`.
    """
    prior_diffused_spectrum = raw_spectrum.clone()

    iteration_count = 1
    max_iterations = 8
    min_correlation_coefficient = 0.95

    while iteration_count <= max_iterations:
        diffused_spectrum = torch.matmul(
            asymmetric_transition_matrix, prior_diffused_spectrum
        )

        numerator = torch.sum(
            (prior_diffused_spectrum - prior_diffused_spectrum.mean())
            * (diffused_spectrum - diffused_spectrum.mean())
        )
        denominator = torch.sqrt(
            torch.sum((prior_diffused_spectrum - prior_diffused_spectrum.mean()) ** 2)
        ) * torch.sqrt(torch.sum((diffused_spectrum - diffused_spectrum.mean()) ** 2))

        correlation_coefficient = numerator / (denominator + 1e-10)

        if correlation_coefficient**2 >= min_correlation_coefficient:
            break

        prior_diffused_spectrum = diffused_spectrum
        iteration_count += 1

    raw_max = raw_spectrum.max(dim=0)[0]
    diffused_max = diffused_spectrum.max(dim=0)[0]

    safe_diffused_max = torch.where(
        diffused_max == 0,
        torch.tensor(1e-10, device=diffused_max.device),
        diffused_max,
    )
    normalization_factor = raw_max / safe_diffused_max

    return diffused_spectrum * normalization_factor[None, :]
