import scanpy as sc
import torch
import tqdm

torch.set_default_dtype(torch.float32)


def get_emd(adata, obs_name, layer_name=None, device="cpu"):
    """
    Computes the Earth Mover's Distance (EMD) for genes between a given observation level and its complement.

    Parameters:
        adata (AnnData): The AnnData object containing single-cell data.
        obs_name (str): The key for observation data.
        obs_level (any): The specific observation level to compare.
        layer_name (str, optional): The name of the data layer to use. Defaults to "imputed".
        device (str, optional): The computation device ("cpu" or "cuda"). Defaults to "cpu".

    Returns:
        torch.Tensor: EMD values for each gene.
    """
    # Load expression matrix onto the specified device
    if layer_name == None:
        X = torch.tensor(adata.X, device=device)
    else:
        X = torch.tensor(
            adata.layers[layer_name].toarray(), device=device
        )  # Shape: (n_cells, n_genes)
    num_genes = X.shape[1]

    obs_levels = adata.obs[obs_name].cat.categories.tolist()
    num_levels = len(obs_levels)

    # Initialize EMD tensor
    emd = torch.zeros((num_genes, num_levels), device=device)

    for l in tqdm.tqdm(range(num_levels), desc=f"Computing EMD for {obs_name}"):

        obs_level = obs_levels[l]

        # Boolean masks for observation level and its complement
        level_mask = torch.tensor(adata.obs[obs_name] == obs_level, device=device)
        complement_mask = ~level_mask

        # Extract subset of cells
        X_level = X[level_mask]

        X_complement = X[complement_mask]

        # If either subset is empty, return NaNs
        if X_level.size(0) == 0 or X_complement.size(0) == 0:
            emd[:, l] = torch.full((num_genes,), float("nan"), device=device)
        else:
            # Process each gene
            for g in tqdm.tqdm(
                range(num_genes), desc=f"Computing EMD for {obs_level}", leave=False
            ):
                # Sort gene expression values
                gene_level = torch.sort(X_level[:, g]).values
                gene_complement = torch.sort(X_complement[:, g]).values

                # Compute empirical cumulative distribution functions (ECDFs)
                n_level, n_complement = gene_level.size(0), gene_complement.size(0)
                cdf_level = torch.arange(1, n_level + 1, device=device) / n_level
                cdf_complement = (
                    torch.arange(1, n_complement + 1, device=device) / n_complement
                )

                # Define a common support grid
                common_grid = (
                    torch.cat([gene_level, gene_complement]).unique().sort().values
                )

                # Interpolate CDFs onto the common grid
                cdf_level_interp = (
                    torch.searchsorted(gene_level, common_grid, right=True).float()
                    / n_level
                )
                cdf_complement_interp = (
                    torch.searchsorted(gene_complement, common_grid, right=True).float()
                    / n_complement
                )

                # Compute Wasserstein-1 distance (EMD)
                distance = torch.sum(
                    torch.abs(cdf_level_interp - cdf_complement_interp)
                    * torch.diff(
                        common_grid, prepend=torch.tensor([0.0], device=device)
                    )
                )

                # Assign a sign based on mean expression difference
                emd[g, l] = distance * torch.sign(
                    gene_level.mean() - gene_complement.mean()
                )

    return emd


def recursive_emd(adata, obs_names, layer_name=None, device="cpu", parent_levels={}):
    """
    Recursively traverses nested observation levels in an AnnData object and computes
    the EMD at the leaf level.

    Parameters:
        adata (AnnData): Single-cell data object.
        obs_names (list of str): List of observation keys defining the hierarchy (e.g., ["cluster", "age", "zone"]).
        layer_name (str, optional): Name of the data layer for gene expression. Defaults to "imputed".
        device (str, optional): Computation device ("cpu" or "cuda"). Defaults to "cpu".
        parent_levels (dict, optional): Tracks the hierarchy path for debugging or structured output.

    Returns:
        dict: Nested dictionary of computed EMDs for each hierarchical level.
    """
    # Base case: Leaf node (last observation level)
    if len(obs_names) == 1:
        obs_name = obs_names[0]

        emd = get_emd(adata, obs_name, layer_name, device)

        return {obs_name: {"emd": emd}}

    # Recursive case: Extract current observation level and restrict the dataset
    obs_name = obs_names[0]
    nested_results = {obs_name: {}}

    nested_results[obs_name]["emd"] = get_emd(adata, obs_name, layer_name, device)

    for level in tqdm.tqdm(
        adata.obs[obs_name].cat.categories.tolist(),
        desc=f"Computing EMD for {obs_name}",
    ):

        adata_subset = adata[adata.obs[obs_name] == level].copy()

        # Track the hierarchy for structured output
        current_levels = {**parent_levels, obs_name: level}

        # Recursively process the next observation levels
        nested_results[obs_name][level] = recursive_emd(
            adata_subset, obs_names[1:], layer_name, device, current_levels
        )

    return nested_results  # Return nested structure
