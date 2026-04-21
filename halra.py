import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd


def validate_matrix(matrix, cells, genes):
    assert isinstance(matrix, np.ndarray) or sp.issparse(matrix)
    if isinstance(matrix, np.ndarray):
        matrix = sp.csc_matrix(matrix)
    elif sp.issparse(matrix):
        matrix = matrix.tocsc()
    assert len(cells) == matrix.shape[0]
    assert len(genes) == matrix.shape[1]
    cells = np.asarray(cells, dtype=object)
    genes = np.asarray(genes, dtype=object)
    return matrix, cells, genes


def filter_matrix(matrix, cells, genes):
    col_sums = np.asarray(matrix.sum(axis=0)).ravel()
    keep_cols = col_sums > 0
    n_omit = np.sum(~keep_cols)
    print(f"Omitted {n_omit} genes with zero counts")
    matrix = matrix[:, keep_cols]
    genes = genes[keep_cols]
    matrix = matrix.tocsr()
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    keep_rows = row_sums > 0
    n_omit = np.sum(~keep_rows)
    print(f"Omitted {n_omit} cells with zero counts")
    matrix = matrix[keep_rows, :]
    cells = cells[keep_rows]
    matrix = matrix.tocsc()
    return matrix, cells, genes


def log_normalize(matrix, scale_factor=1e4):
    matrix = matrix.tocsr()
    inv_row_sums = 1.0 / np.asarray(matrix.sum(axis=1)).ravel()
    matrix = matrix.multiply(inv_row_sums[:, None]) * scale_factor
    matrix.data = np.log1p(matrix.data)
    matrix = matrix.tocsc()
    return matrix


def choose_matrix_rank(matrix, n_iter, seed, n_comps=100, thresh=6, noise_start=80):
    assert n_comps < min(matrix.shape)
    assert noise_start <= n_comps - 5
    noise_svals = np.arange(noise_start, n_comps)
    _, s, _ = randomized_svd(matrix, n_components=n_comps, n_iter=n_iter, random_state=seed)
    diffs = s[:-1] - s[1:]
    mu = np.mean(diffs[noise_svals - 1])
    sigma = np.std(diffs[noise_svals - 1])
    n_std_devs = (diffs - mu) / sigma
    rank = np.max(np.where(n_std_devs > thresh)[0]) + 1
    print(f"Chose rank={rank}")
    return rank


def create_reconstruction(matrix, rank, n_iter, seed):
    u, s, vh = randomized_svd(matrix, n_components=rank, n_iter=n_iter, random_state=seed)
    recon_matrix = u @ np.diag(s) @ vh
    print("Computed low-rank reconstruction")
    return recon_matrix


def threshold_reconstruction(recon_matrix, quantile_prob):
    thresholds = np.abs(np.quantile(recon_matrix, quantile_prob, axis=0))
    mask = recon_matrix > thresholds[np.newaxis, :]
    thresh_matrix = sp.csc_matrix(recon_matrix * mask)
    thresh_matrix.eliminate_zeros()
    return thresh_matrix


def column_mean_std_nnz(matrix):
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


def create_scaling_factors(thresh_matrix, matrix):
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

    print("Computed scaling factors")
    return scale_mask, scale_factors, offsets


def apply_scaling(thresh_matrix, to_scale, sigma_1_2, to_add):
    imputed_matrix = sp.csc_matrix(thresh_matrix)
    for j in np.where(to_scale)[0]:
        start, end = imputed_matrix.indptr[j], imputed_matrix.indptr[j + 1]
        if start == end:
            continue
        vals = imputed_matrix.data[start:end]
        vals = vals * sigma_1_2[j] + to_add[j]
        imputed_matrix.data[start:end] = vals
    print(f"Scaled all except {np.sum(~to_scale)} genes")
    return imputed_matrix


def clip_negative_values(imputed_matrix):
    neg = imputed_matrix.data < 0
    pct_neg = round(100 * np.sum(neg) / (imputed_matrix.shape[0] * imputed_matrix.shape[1]), 2)
    imputed_matrix.data[neg] = 0
    imputed_matrix.eliminate_zeros()
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


def report_density(matrix, imputed_matrix):
    start_nnz = round(100 * matrix.nnz / (matrix.shape[0] * matrix.shape[1]), 2)
    end_nnz = round(100 * imputed_matrix.nnz / (imputed_matrix.shape[0] * imputed_matrix.shape[1]), 2)
    print(f"Original nonzero values: {start_nnz}%")
    print(f"Imputed nonzero values: {end_nnz}%")


def impute_matrix(matrix, rank, n_iter, quantile_prob, seed):
    recon_matrix = create_reconstruction(matrix, rank, n_iter, seed)
    thresh_matrix = threshold_reconstruction(recon_matrix, quantile_prob)
    del recon_matrix
    to_scale, sigma_1_2, to_add = create_scaling_factors(thresh_matrix, matrix)
    imputed_matrix = apply_scaling(thresh_matrix, to_scale, sigma_1_2, to_add)
    imputed_matrix = clip_negative_values(imputed_matrix)
    imputed_matrix = restore_observed_values(imputed_matrix, matrix)
    imputed_matrix.eliminate_zeros()
    report_density(matrix, imputed_matrix)
    return imputed_matrix


def halra(matrix, cells, genes, normalize=False, n_iter=12,
          quantile_prob=0.001, matrix_rank=None, seed=0):
    matrix, cells, genes = validate_matrix(matrix, cells, genes)
    matrix, cells, genes = filter_matrix(matrix, cells, genes)
    if normalize: matrix = log_normalize(matrix)
    n_row, n_col = matrix.shape
    print(f"Read matrix with {n_row} cells and {n_col} genes")
    if matrix_rank is None: matrix_rank = choose_matrix_rank(matrix, n_iter, seed)
    return impute_matrix(matrix, matrix_rank, n_iter, quantile_prob, seed)
