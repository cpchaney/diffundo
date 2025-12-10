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

import torch
import tqdm


def standardize_tensor(X: torch.Tensor) -> torch.Tensor:
    """
    Standardize tensor so each feature has zero mean and unit variance.

    Args:
        X (torch.Tensor): Tensor of shape (n_samples, n_features)

    Returns:
        torch.Tensor: Standardized tensor
    """
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True)
    std[std == 0] = 1  # Avoid division by zero
    return (X - mean) / std


def gaussian_kernel_matrix(
    X: torch.Tensor, sigma: float, Y: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute the Gaussian (RBF) kernel matrix.

    If Y is None, computes the self-kernel: K[i, j] = exp(-||X[i] - X[j]||^2 / (2 * sigma^2))
    If Y is provided, computes the cross-kernel: K[i, j] = exp(-||X[i] - Y[j]||^2 / (2 * sigma^2))

    Args:
        X (torch.Tensor): Tensor of shape (n_samples_X, n_features)
        sigma (float): Kernel bandwidth
        Y (torch.Tensor, optional): Tensor of shape (n_samples_Y, n_features)

    Returns:
        torch.Tensor: Kernel matrix of shape (n_samples_X, n_samples_Y) if Y is given,
                      else (n_samples_X, n_samples_X)
    """
    if Y is None:
        sq_dists = torch.cdist(X, X, p=2).pow(2)
    else:
        sq_dists = torch.cdist(X, Y, p=2).pow(2)

    return torch.exp(-sq_dists / (2 * sigma**2))


def calculate_radii(
    adata,
    features=None,
    feature_source=None,
    group_name="cluster",
    standardize=True,
):
    """
    Estimate kernel bandwidths (sigma_0, sigma_min) from intra-cluster distances
    in an arbitrary feature space.

    Args:
        adata (AnnData): Scanpy AnnData object.
        features (torch.Tensor | np.ndarray, optional): Feature matrix of shape [n_cells, n_features].
            If provided, this is used directly.
        feature_source (dict, optional): Dict specifying how to extract features from `adata`.
            Supported keys:
                - "layer" + "genes": extract gene expression from a named layer.
                - "obsm": extract features from `adata.obsm["..."]`.
        group_name (str): Name of categorical column in adata.obs for clustering.
        standardize (bool): Whether to z-score standardize the features.

    Returns:
        Tuple[float, float]: (sigma_0, sigma_min)
    """
    import torch

    if features is None:
        if feature_source is None:
            raise ValueError("Must provide either `features` or `feature_source`.")

        if "layer" in feature_source and "genes" in feature_source:
            X = adata[:, feature_source["genes"]].layers[feature_source["layer"]]
            X = X.toarray() if not isinstance(X, np.ndarray) else X
            features = torch.tensor(X)

        elif "obsm" in feature_source:
            features = torch.tensor(adata.obsm[feature_source["obsm"]])

        else:
            raise ValueError(
                "Unsupported `feature_source` format. Use 'layer'+'genes' or 'obsm'."
            )

    elif not isinstance(features, torch.Tensor):
        features = torch.tensor(features)

    if standardize:
        features = standardize_tensor(features)

    dist = torch.cdist(features, features, p=2)

    radii = []
    groups = adata.obs[group_name].astype("category").cat.categories.tolist()
    for group in groups[1:]:  # Skip first to avoid 0-radius singleton issues
        idx = torch.tensor((adata.obs[group_name] == group).values, dtype=torch.bool)
        if idx.sum() < 2:
            continue
        radii.append(dist[idx][:, idx].max().item())

    sigma_0 = max(radii) if radii else dist.max().item()
    sigma_min = dist[dist > 0].min().item()

    return sigma_0, sigma_min


def multiscale_extension(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_query: torch.Tensor,
    sigma_0: float,
    sigma_min: float,
    condition_number: float = 100,
    tolerance: float = 0.1,
) -> torch.Tensor:
    """
    Extend function values from training to query points using geometric harmonics.

    Args:
        X_train (torch.Tensor): (n_samples, n_features) ? features for reference
        Y_train (torch.Tensor): (n_samples, n_functions) ? function values (e.g., diffusion map)
        X_query (torch.Tensor): (m_samples, n_features) ? features for query set
        sigma_0 (float): Initial kernel bandwidth
        sigma_min (float): Minimum allowable bandwidth
        condition_number (float): Max ratio of largest to smallest eigenvalue retained
        tolerance (float): Acceptable relative reconstruction error

    Returns:
        torch.Tensor: (m_samples, n_functions) extended values
    """
    m = X_query.shape[0]
    n_functions = Y_train.shape[1]
    Y_extended = torch.zeros((m, n_functions))

    for i in tqdm.tqdm(range(n_functions), desc="Extending functions"):
        sigma = sigma_0
        reconstruction_error = 1

        while reconstruction_error > tolerance and sigma > sigma_min:
            K = gaussian_kernel_matrix(X_train, sigma)
            evals, evecs = torch.linalg.eigh(K)
            evals = torch.flip(evals, dims=[0])
            evecs = torch.flip(evecs, dims=[1])

            coeffs = (Y_train.T @ evecs).T
            num_retained = (torch.sum((evals[0] / evals) < condition_number) + 1).item()

            reconstruction_error = torch.norm(
                coeffs[num_retained:, i], p=2
            ) / torch.norm(Y_train[:, i], p=2)

            sigma /= 2

        sigma *= 2  # Backtrack to last good sigma
        K_query = torch.exp(-torch.cdist(X_query, X_train).pow(2) / (2 * sigma**2))
        evecs_scaled = evecs / evals
        K_ext = K_query @ evecs_scaled[:, :num_retained]
        Y_extended[:, i] = K_ext @ coeffs[:num_retained, i]

    return Y_extended
