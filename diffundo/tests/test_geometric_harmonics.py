import pytest
import torch
from diffundo.geometric_harmonics import (gaussian_kernel_matrix,
                                          multiscale_extension,
                                          standardize_tensor)


def test_standardize_tensor():
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    X_std = standardize_tensor(X)

    assert X_std.shape == X.shape
    assert torch.allclose(X_std.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(X_std.std(dim=0), torch.ones(2), atol=1e-6)


def test_gaussian_kernel_matrix():
    X = torch.tensor([[0.0], [1.0], [2.0]])
    sigma = 1.0
    K = gaussian_kernel_matrix(X, sigma)

    assert K.shape == (3, 3)
    assert torch.allclose(K, K.T, atol=1e-6)  # Symmetric
    assert torch.all(K <= 1.0)
    assert torch.all(K >= 0.0)


def test_multiscale_extension_identity():
    # Create synthetic data where the extension target is identical to the reference
    X = torch.randn(10, 5)
    Y = torch.sin(X[:, 0:1])  # Single function to extend

    Y_ext = multiscale_extension(
        X_train=X,
        Y_train=Y,
        X_query=X,
        Y_query=Y,
        sigma_0=1.0,
        sigma_min=0.01,
        condition_number=100,
        tolerance=0.1,
    )

    # Expect reconstruction to be reasonably close
    assert Y_ext.shape == Y.shape
    error = torch.norm(Y - Y_ext) / torch.norm(Y)
    assert error < 0.15  # Loose threshold for generality
