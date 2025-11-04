import numpy as np
from sklearn.decomposition import PCA


def project_to_reference_pca(
    adata,
    pca_model: dict,
    layer: str | None = None,
    out_key="X_pca_ref",
    strict_genes=False,
):
    """
    Project `adata` into the PCA space defined by `pca_model` (from WT).
    Writes scores to .obsm[out_key].

    Parameters
    ----------
    adata : AnnData
    pca_model : dict  The .uns["pca_model"] dict from fit_pca_on_reference
    layer : str|None  If None, use pca_model["layer"]
    out_key : str     obsm key for projected PCs
    strict_genes : bool
        If True, error if any model genes are missing in adata; else fill missing with zeros after standardization.

    Returns
    -------
    None
    """
    model_genes = pca_model["genes"]
    mean = np.array(pca_model["mean"], dtype=np.float32)
    std = np.array(pca_model["std"], dtype=np.float32)
    U = np.array(pca_model["components"], dtype=np.float32)  # (d, G)
    U_T = U.T  # (G, d)

    if layer is None:
        layer = pca_model.get("layer", None)
        if layer is None:
            raise ValueError("Layer must be provided if not stored in pca_model.")

    # Build matrix in the exact gene order expected by the model.
    present = np.array([g in adata.var_names for g in model_genes])
    if strict_genes and not np.all(present):
        missing = [g for g, ok in zip(model_genes, present) if not ok]
        raise KeyError(
            f"{len(missing)} model genes missing from adata: {missing[:10]} ..."
        )

    X_list = []
    if np.any(present):
        X_sub = adata[:, [g for g, ok in zip(model_genes, present) if ok]].layers[layer]
        if hasattr(X_sub, "toarray"):
            X_sub = X_sub.toarray()
        X_sub = np.asarray(X_sub, dtype=np.float32)

        # standardize with WT stats (only for present genes)
        mean_sub = mean[present]
        std_sub = std[present]
        std_sub[std_sub == 0] = 1.0
        Xz_sub = (X_sub - mean_sub) / std_sub
        X_list.append((present, Xz_sub))

    # If there are missing genes, fill standardized values with zeros (i.e., equal to WT mean)
    if np.any(~present):
        Xz_missing = np.zeros((adata.n_obs, int((~present).sum())), dtype=np.float32)
        X_list.append((~present, Xz_missing))

    # Stitch back to [N, G] in model order
    Xz = np.zeros((adata.n_obs, len(model_genes)), dtype=np.float32)
    cursor = 0
    for mask, block in X_list:
        # place block into positions where mask is True (preserves model order)
        Xz[:, np.where(mask)[0]] = block

    # Project
    scores = Xz @ U_T  # (N, d)
    adata.obsm[out_key] = scores.astype(np.float32)


def apply_pca_from_reference(
    adata,
    ref_pca_model: dict,
    layer: str = "logcounts",
    out_key: str = "X_pca_ref",
    strict_genes: bool = False,
    dtype=np.float32,
    chunk_size: int | None = None,
):
    """
    Project `adata` into the PCA space defined by a reference PCA model.

    Parameters
    ----------
    adata : AnnData
    ref_pca_model : dict
        The dict saved by fit_pca_on_reference; must contain:
        - "genes" (ordered list used for PCA fit)
        - "mean" (G,), "std" (G,), "components" (d, G)
    layer : str
        Layer to read (should match preprocessing used for the model).
    out_key : str
        obsm key to store projected scores.
    strict_genes : bool
        If True, raise if any model genes are missing in `adata`.
        If False, missing genes are filled with standardized zeros (= WT mean).
    dtype : np.dtype
        Numeric dtype for computation (float32 saves RAM).
    chunk_size : int | None
        If set, project cells in batches of this size to reduce peak memory.

    Returns
    -------
    None (writes scores to adata.obsm[out_key])
    """
    model_genes = list(ref_pca_model["genes"])
    mean = np.asarray(ref_pca_model["mean"], dtype=dtype)
    std = np.asarray(ref_pca_model["std"], dtype=dtype)
    std[std == 0] = 1.0
    U = np.asarray(ref_pca_model["components"], dtype=dtype)  # (d, G)
    UT = U.T  # (G, d)

    # Figure out which model genes are present in this AnnData
    present_mask = np.array([g in adata.var_names for g in model_genes], dtype=bool)
    if strict_genes and not np.all(present_mask):
        missing = [g for g, ok in zip(model_genes, present_mask) if not ok]
        raise KeyError(
            f"{len(missing)} model genes missing from adata; e.g., {missing[:10]}"
        )

    # Pull the present block in model order
    present_genes_in_order = [g for g, ok in zip(model_genes, present_mask) if ok]
    if present_genes_in_order:
        X_block = adata[:, present_genes_in_order].layers[layer]
        if hasattr(X_block, "toarray"):
            X_block = X_block.toarray()
        X_block = np.asarray(X_block, dtype=dtype)
        # Standardize using WT stats (only for present genes)
        mean_sub = mean[present_mask]
        std_sub = std[present_mask]
        Xz_block = (X_block - mean_sub) / std_sub
    else:
        # Degenerate case: no overlap
        Xz_block = np.zeros((adata.n_obs, 0), dtype=dtype)

    # Allocate full standardized matrix in the model-gene order
    N = adata.n_obs
    G = len(model_genes)
    Xz = np.zeros((N, G), dtype=dtype)
    if Xz_block.shape[1] > 0:
        Xz[:, present_mask] = Xz_block
    # Missing genes remain zero after standardization (i.e., equal to WT mean)

    # Project: scores = Xz @ UT  (N × G) @ (G × d) = (N × d)
    if chunk_size is None or N <= (chunk_size or 0):
        scores = Xz @ UT
    else:
        d = UT.shape[1]
        scores = np.empty((N, d), dtype=dtype)
        for i in range(0, N, chunk_size):
            j = min(i + chunk_size, N)
            scores[i:j] = Xz[i:j] @ UT

    adata.obsm[out_key] = scores
