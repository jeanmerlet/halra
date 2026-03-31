import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd


def ensure_csr(mtx):
    if isinstance(mtx, np.ndarray):
        return sp.csr_matrix(mtx)
    elif sp.issparse(mtx):
        return mtx.tocsr()
    else:
        raise TypeError("Input must be numpy.ndarray or scipy.sparse matrix")


def remove_zero_genes_cells(mtx):
    # genes (columns)
    col_sums = np.asarray(mtx.sum(axis=0)).ravel()
    keep_cols = col_sums > 0
    n_omit = np.sum(~keep_cols)
    if n_omit > 0:
        print(f"Omitted {n_omit} genes with zero counts")
    mtx = mtx[:, keep_cols]

    # cells (rows)
    row_sums = np.asarray(mtx.sum(axis=1)).ravel()
    keep_rows = row_sums > 0
    n_omit = np.sum(~keep_rows)
    if n_omit > 0:
        print(f"Omitted {n_omit} cells with zero counts")
    mtx = mtx[keep_rows, :]

    return mtx, row_sums[keep_rows]


def normalize_total(mtx, row_sums, scale_factor=1e4):
    inv_row_sums = 1.0 / row_sums
    return mtx.multiply(inv_row_sums[:, None]) * scale_factor


def log1p_sparse(mtx):
    mtx.data = np.log1p(mtx.data)
    return mtx


def log_normalize_counts(mtx, scale_factor=1e4):
    mtx = ensure_csr(mtx)
    mtx, row_sums = remove_zero_genes_cells(mtx)
    mtx = normalize_total(mtx, row_sums, scale_factor)
    mtx = log1p_sparse(mtx)
    return mtx


def choose_rank(mtx, n_iter, seed, n_comps=100, thresh=6, noise_start=80):
    if n_comps >= min(mtx.shape):
        raise ValueError("n_comps must be smaller than the smallest dimension of mtx.")
    if noise_start > n_comps - 5:
        raise ValueError("There need to be at least 5 singular values considered noise.")

    noise_svals = np.arange(noise_start, n_comps)
    _, s, _ = randomized_svd(mtx, n_components=n_comps, n_iter=n_iter, random_state=seed)

    diffs = s[:-1] - s[1:]
    mu = np.mean(diffs[noise_svals - 1])
    sigma = np.std(diffs[noise_svals - 1])
    n_std_devs = (diffs - mu) / sigma
    rank = np.max(np.where(n_std_devs > thresh)[0]) + 1
    print(f"Chose rank={rank}")
    return rank


def sparse_col_std_nnz(mtx):
    mtx = mtx.tocsc()
    out = np.full(mtx.shape[1], 0.0, dtype=float)
    for j in range(mtx.shape[1]):
        start, end = mtx.indptr[j], mtx.indptr[j + 1]
        vals = mtx.data[start:end]
        if vals.size > 0:
            out[j] = np.std(vals)
    return out


def sparse_col_mean_nnz(mtx):
    mtx = mtx.tocsc()
    out = np.full(mtx.shape[1], 0.0, dtype=float)
    for j in range(mtx.shape[1]):
        start, end = mtx.indptr[j], mtx.indptr[j + 1]
        vals = mtx.data[start:end]
        if vals.size > 0:
            out[j] = np.mean(vals)
    return out


def std_dev_nnz(x):
    vals = x[x != 0]
    return np.std(vals) if vals.size > 0 else 0.0


def mean_nnz(x):
    vals = x[x != 0]
    return np.mean(vals) if vals.size > 0 else 0.0


def compute_low_rank_reconstruction(mtx, rank, n_iter, seed):
    print("Computing low-rank reconstruction")
    u, s, vh = randomized_svd(mtx, n_components=rank, n_iter=n_iter, random_state=seed)
    recon_mtx = u @ np.diag(s) @ vh
    return recon_mtx


def threshold_reconstruction(recon_mtx, quantile_prob):
    print("Thresholding")
    recon_mtx_mins = np.abs(np.quantile(recon_mtx, quantile_prob, axis=0))
    condition = recon_mtx <= recon_mtx_mins[np.newaxis, :]
    thresh_mtx = np.where(condition, 0, recon_mtx)
    return thresh_mtx


def compute_scaling_factors(thresh_mtx, mtx):
    print("Computing scaling factors")
    sigma_1 = np.apply_along_axis(std_dev_nnz, 0, thresh_mtx)
    sigma_2 = sparse_col_std_nnz(mtx)

    nnz_1 = np.sum(thresh_mtx != 0, axis=0)
    mu_1 = np.divide(
        np.sum(thresh_mtx, axis=0),
        nnz_1,
        out=np.zeros(thresh_mtx.shape[1], dtype=float),
        where=nnz_1 != 0
    )
    mu_2 = sparse_col_mean_nnz(mtx)

    to_scale = (
        (~np.isnan(sigma_1))
        & (~np.isnan(sigma_2))
        & ~((sigma_1 == 0) & (sigma_2 == 0))
        & (sigma_1 != 0)
    )

    sigma_1_2 = np.divide(
        sigma_2,
        sigma_1,
        out=np.zeros_like(sigma_2, dtype=float),
        where=sigma_1 != 0
    )
    to_add = -mu_1 * sigma_1_2 + mu_2
    return to_scale, sigma_1_2, to_add


def scale_thresholded_matrix(thresh_mtx, to_scale, sigma_1_2, to_add):
    print(f"Scaling all except for {np.sum(~to_scale)} columns")
    recon_mtx_tmp = thresh_mtx[:, to_scale]
    recon_mtx_tmp = recon_mtx_tmp * sigma_1_2[to_scale]
    recon_mtx_tmp = recon_mtx_tmp + to_add[to_scale]

    imputed_mtx = thresh_mtx.copy()
    imputed_mtx[:, to_scale] = recon_mtx_tmp
    imputed_mtx[thresh_mtx == 0] = 0
    return imputed_mtx


def zero_negatives(imputed_mtx):
    neg = imputed_mtx < 0
    imputed_mtx[neg] = 0
    pct_neg = round(100 * np.sum(neg) / imputed_mtx.size, 2)
    print(f"{pct_neg}% of the values became negative in the scaling process and were set to zero.")
    return imputed_mtx


def restore_observed_values(imputed_mtx, mtx, pres_obs="zeroed"):
    coo = mtx.tocoo()

    if pres_obs == "zeroed":
        restore = imputed_mtx[coo.row, coo.col] == 0
        imputed_mtx[coo.row[restore], coo.col[restore]] = coo.data[restore]
    elif pres_obs == "all":
        imputed_mtx[coo.row, coo.col] = coo.data
    elif pres_obs == "none":
        pass
    else:
        raise ValueError("pres_obs must be 'zeroed', 'all', or 'none'")

    return imputed_mtx


def report_density(mtx, imputed_mtx):
    start_nnz = round(mtx.nnz / (mtx.shape[0] * mtx.shape[1]), 2)
    end_nnz = round(np.count_nonzero(imputed_mtx) / imputed_mtx.size, 2)
    print(f"Original nonzero values: {start_nnz}%")
    print(f"Imputed nonzero values: {end_nnz}%")


def halra(mtx, n_iter=12, quantile_prob=0.001, seed=1, normalize=False, pres_obs="zeroed"):
    if normalize:
        mtx = log_normalize_counts(mtx)
    else:
        mtx = ensure_csr(mtx)
    mtx = mtx.tocsc()
    n_row, n_col = mtx.shape
    print(f"Read matrix with {n_row} cells and {n_col} genes")

    rank = choose_rank(mtx, n_iter, seed)
    recon_mtx = compute_low_rank_reconstruction(mtx, rank, n_iter, seed)
    thresh_mtx = threshold_reconstruction(recon_mtx, quantile_prob)

    to_scale, sigma_1_2, to_add = compute_scaling_factors(thresh_mtx, mtx)
    imputed_mtx = scale_thresholded_matrix(thresh_mtx, to_scale, sigma_1_2, to_add)
    imputed_mtx = zero_negatives(imputed_mtx)
    imputed_mtx = restore_observed_values(imputed_mtx, mtx)
    imputed_mtx = restore_observed_values(imputed_mtx, mtx, pres_obs=pres_obs)

    report_density(mtx, imputed_mtx)
    return imputed_mtx
