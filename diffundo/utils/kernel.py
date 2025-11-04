import torch
from torch import Tensor


def get_distance_matrix(input_tensor: Tensor) -> Tensor:
    return torch.cdist(input_tensor, input_tensor)


def get_sigma(distance_matrix: Tensor, k: int = 8) -> Tensor:
    return torch.kthvalue(distance_matrix, k + 1).values


def get_interference_kernel(distance_matrix: Tensor, sigma: Tensor) -> Tensor:
    P = sigma.outer(sigma)
    S = (sigma**2)[:, None] + (sigma**2)[None, :]
    return torch.sqrt(2 * P / S) * torch.exp(-(distance_matrix**2) / (2 * S))


def get_anisotropic_kernel(
    distance_matrix: Tensor, sigma: Tensor, epsilon: float = 1.0
) -> Tensor:
    S = sigma.outer(sigma)
    return torch.exp(-(distance_matrix**2) / (epsilon * S))


def get_normalized_kernel(kernel: Tensor, alpha: float = 0.5) -> Tensor:
    row_sum = kernel.sum(dim=1)
    return kernel / torch.outer(row_sum**alpha, row_sum**alpha)


def get_asymmetric_transition_matrix(W: Tensor) -> Tensor:
    return W / W.sum(dim=1, keepdim=True)
