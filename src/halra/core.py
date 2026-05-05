import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd


def validate_matrix(matrix, cells, genes):
    """
    Validate and standardize an input expression matrix and its labels.

    Dense NumPy arrays are converted to sparse CSC format, and sparse inputs are
    converted to CSC format. Cell and gene labels are converted to NumPy object
    arrays after checking that their lengths match the matrix dimensions.
    """
    if not (isinstance(matrix, np.ndarray) or sp.issparse(matrix)):
        raise TypeError("matrix must be a NumPy ndarray or a SciPy sparse matrix")

    if isinstance(matrix, np.ndarray):
        matrix = sp.csc_matrix(matrix)
    elif sp.issparse(matrix):
        matrix = matrix.tocsc()

    if len(cells) != matrix.shape[0]:
        raise ValueError(
            "cells length must match the number of matrix rows "
            f"({len(cells)} != {matrix.shape[0]})"
        )

    if len(genes) != matrix.shape[1]:
        raise ValueError(
            "genes length must match the number of matrix columns "
            f"({len(genes)} != {matrix.shape[1]})"
        )

    cells = np.asarray(cells, dtype=object)
    genes = np.asarray(genes, dtype=object)
    return matrix, cells, genes


def filter_matrix(matrix, cells, genes, verbose=True):
    """
    Remove all-zero genes and cells from a sparse expression matrix.

    Genes with zero total counts are removed first, followed by cells with zero
    total counts after gene filtering. The returned matrix is in CSC format, and
    the returned cell and gene labels are filtered to match.
    """
    col_sums = np.asarray(matrix.sum(axis=0)).ravel()
    keep_cols = col_sums > 0
    n_omit = np.sum(~keep_cols)
    if verbose:
        print(f"Omitted {n_omit} genes with zero counts")
    matrix = matrix[:, keep_cols]
    genes = genes[keep_cols]

    matrix = matrix.tocsr()
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    keep_rows = row_sums > 0
    n_omit = np.sum(~keep_rows)
    if verbose:
        print(f"Omitted {n_omit} cells with zero counts")
    matrix = matrix[keep_rows, :]
    cells = cells[keep_rows]

    matrix = matrix.tocsc()
    return matrix, cells, genes


def log_normalize(matrix, scale_factor=1e4):
    """
    Apply cell-wise library-size normalization followed by log1p transform.

    Each row is divided by its total count, multiplied by scale_factor, and then
    transformed with log1p. The returned matrix is in CSC format.
    """
    matrix = matrix.tocsr()
    inv_row_sums = 1.0 / np.asarray(matrix.sum(axis=1)).ravel()
    matrix = matrix.multiply(inv_row_sums[:, None]) * scale_factor
    matrix.data = np.log1p(matrix.data)
    matrix = matrix.tocsc()
    return matrix


def choose_matrix_rank(matrix, n_iter, seed, n_comps=100, thresh=6, noise_start=80, verbose=True):
    """
    Choose an approximate matrix rank from randomized SVD singular values.

    The rank is selected by comparing consecutive singular-value differences to
    a noise distribution estimated from later singular-value gaps.
    """
    if n_comps >= min(matrix.shape):
        raise ValueError(
            "n_comps must be smaller than the smallest matrix dimension "
            f"({n_comps} >= {min(matrix.shape)})"
        )

    if noise_start > n_comps - 5:
        raise ValueError(
            "noise_start must be at least 5 components before n_comps "
            f"({noise_start} > {n_comps - 5})"
        )

    noise_svals = np.arange(noise_start, n_comps)
    _, s, _ = randomized_svd(matrix, n_components=n_comps, n_iter=n_iter, random_state=seed)
    diffs = s[:-1] - s[1:]
    mu = np.mean(diffs[noise_svals - 1])
    sigma = np.std(diffs[noise_svals - 1])
    n_std_devs = (diffs - mu) / sigma
    rank = np.max(np.where(n_std_devs > thresh)[0]) + 1
    if verbose:
        print(f"Chose rank={rank}")
    return rank


def create_reconstruction(matrix, rank, n_iter, seed, verbose=True):
    """
    Compute a dense low-rank reconstruction from a sparse input matrix.

    Randomized SVD is used to compute rank components, which are multiplied back
    together to form the reconstructed matrix.
    """
    u, s, vh = randomized_svd(matrix, n_components=rank, n_iter=n_iter, random_state=seed)
    recon_matrix = u @ np.diag(s) @ vh
    if verbose:
        print("Computed low-rank reconstruction")
    return recon_matrix


def threshold_reconstruction(recon_matrix, quantile_prob):
    """
    Threshold a dense reconstructed matrix by gene-wise quantiles.

    Values less than or equal to the absolute quantile threshold for each column
    are set to zero. The returned thresholded matrix is sparse CSC.
    """
    thresholds = np.abs(np.quantile(recon_matrix, quantile_prob, axis=0))
    mask = recon_matrix > thresholds[np.newaxis, :]
    thresh_matrix = sp.csc_matrix(recon_matrix * mask)
    thresh_matrix.eliminate_zeros()
    return thresh_matrix


def column_mean_std_nnz(matrix):
    """
    Compute column-wise means and standard deviations over nonzero values.

    Columns with no stored nonzero values receive mean and standard deviation of
    zero. The input is converted to CSC format for efficient column access.
    """
    matrix = matrix.tocsc()
    n_cols = matrix.shape[1]
    means = np.zeros(n_cols, dtype=float)
    stds = np.zeros(n_cols, dtype=float)

    for j in range(n_cols):
        start, end = matrix.indptr[j], matrix.indptr[j + 1]
        values = matrix.data[start:end]
        if values.size:
            means[j] = values.mean()
            stds[j] = values.std()

    return means, stds


def create_scaling_factors(thresh_matrix, matrix, verbose=True):
    """
    Compute per-gene scale factors and offsets for thresholded values.

    The thresholded reconstruction is scaled to match the nonzero mean and
    standard deviation of the corresponding original input columns where possible.
    """
    thresh_matrix = thresh_matrix.tocsc()
    matrix = matrix.tocsc()

    mu_1, sigma_1 = column_mean_std_nnz(thresh_matrix)
    mu_2, sigma_2 = column_mean_std_nnz(matrix)

    scale_mask = sigma_1 != 0
    scale_factors = np.divide(
        sigma_2,
        sigma_1,
        out=np.zeros_like(sigma_2, dtype=float),
        where=scale_mask,
    )
    offsets = mu_2 - mu_1 * scale_factors

    if verbose:
        print("Computed scaling factors")
    return scale_mask, scale_factors, offsets


def apply_scaling(thresh_matrix, to_scale, sigma_1_2, to_add, verbose=True):
    """
    Apply per-gene scaling factors and offsets to a thresholded sparse matrix.

    Only columns marked by to_scale are modified. The returned matrix is sparse
    CSC and contains the scaled imputed values.
    """
    imputed_matrix = sp.csc_matrix(thresh_matrix)
    for j in np.where(to_scale)[0]:
        start, end = imputed_matrix.indptr[j], imputed_matrix.indptr[j + 1]
        if start == end:
            continue
        vals = imputed_matrix.data[start:end]
        vals = vals * sigma_1_2[j] + to_add[j]
        imputed_matrix.data[start:end] = vals
    if verbose:
        print(f"Scaled all except {np.sum(~to_scale)} genes")
    return imputed_matrix


def clip_negative_values(imputed_matrix, verbose=True):
    """
    Set negative imputed values to zero and remove explicit zeros.

    The percentage reported, when verbose is enabled, is relative to all matrix
    entries rather than only stored nonzero entries.
    """
    neg = imputed_matrix.data < 0
    pct_neg = round(100 * np.sum(neg) / (imputed_matrix.shape[0] * imputed_matrix.shape[1]), 2)
    imputed_matrix.data[neg] = 0
    imputed_matrix.eliminate_zeros()
    if verbose:
        print(f"{pct_neg}% of the values became negative and were set to zero")
    return imputed_matrix


def restore_observed_values(imputed_matrix, matrix):
    """
    Restore originally nonzero values that became zero during thresholding and
    scaling.

    Parameters
    ----------
    imputed_matrix : scipy.sparse matrix
        Imputed sparse matrix.
    matrix : scipy.sparse matrix
        Original sparse input matrix.

    Returns
    -------
    scipy.sparse.csc_matrix
        Sparse matrix with zeroed observed values restored.
    """
    imputed_matrix = imputed_matrix.tocsc()
    matrix = matrix.tocsc()
    # Binary sparsity mask of imputed_matrix
    imputed_mask = imputed_matrix.copy()
    imputed_mask.data = np.ones_like(imputed_mask.data)
    # Original observed entries that are still present in imputed_matrix
    observed_retained = matrix.multiply(imputed_mask)
    # Original observed entries that were zeroed out and need restoration
    observed_missing = matrix - observed_retained
    restored = imputed_matrix + observed_missing
    return restored


def report_density(matrix, imputed_matrix, verbose=True):
    """
    Report the percentage of nonzero entries before and after imputation.

    Percentages are computed relative to the full matrix size.
    """
    if not verbose:
        return
    start_nnz = round(100 * matrix.nnz / (matrix.shape[0] * matrix.shape[1]), 2)
    end_nnz = round(100 * imputed_matrix.nnz / (imputed_matrix.shape[0] * imputed_matrix.shape[1]), 2)
    print(f"Original nonzero values: {start_nnz}%")
    print(f"Imputed nonzero values: {end_nnz}%")


def impute_matrix(matrix, rank, n_iter, quantile_prob, seed, verbose=True):
    """
    Run HALRA imputation on an already validated and optionally normalized matrix.

    This function computes the low-rank reconstruction, thresholds it, rescales
    imputed values, clips negative values, restores observed values, and reports
    the final density when verbose is enabled.
    """
    recon_matrix = create_reconstruction(matrix, rank, n_iter, seed, verbose=verbose)
    thresh_matrix = threshold_reconstruction(recon_matrix, quantile_prob)
    del recon_matrix
    to_scale, sigma_1_2, to_add = create_scaling_factors(thresh_matrix, matrix, verbose=verbose)
    imputed_matrix = apply_scaling(thresh_matrix, to_scale, sigma_1_2, to_add, verbose=verbose)
    imputed_matrix = clip_negative_values(imputed_matrix, verbose=verbose)
    imputed_matrix = restore_observed_values(imputed_matrix, matrix)
    imputed_matrix.eliminate_zeros()
    report_density(matrix, imputed_matrix, verbose=verbose)
    return imputed_matrix


def halra(matrix, cells, genes, normalize=False, n_iter=12,
          quantile_prob=0.001, matrix_rank=None, seed=0, verbose=True):
    """
    Run HALRA imputation on a dense or sparse expression matrix.

    The input matrix is validated, converted to CSC format, filtered to remove
    all-zero genes and cells, optionally log-normalized, and then imputed. The
    returned tuple contains the imputed matrix and the filtered cell and gene
    labels.
    """
    matrix, cells, genes = validate_matrix(matrix, cells, genes)
    matrix, cells, genes = filter_matrix(matrix, cells, genes, verbose=verbose)
    if normalize:
        matrix = log_normalize(matrix)
    n_row, n_col = matrix.shape
    if verbose:
        print(f"Read matrix with {n_row} cells and {n_col} genes")
    if matrix_rank is None:
        matrix_rank = choose_matrix_rank(matrix, n_iter, seed, verbose=verbose)
    imputed_matrix = impute_matrix(matrix, matrix_rank, n_iter, quantile_prob, seed, verbose=verbose)
    return imputed_matrix, cells, genes
