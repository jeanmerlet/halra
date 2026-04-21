import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
import os


def parse_input(mtx, cells, genes):
    assert isinstance(mtx, np.ndarray) or sp.issparse(mtx)
    if isinstance(mtx, np.ndarray):
        mtx = sp.csc_matrix(mtx)
    elif sp.issparse(mtx):
        mtx = mtx.tocsc()
    assert len(cells) == mtx.shape[0]
    assert len(genes) == mtx.shape[1]
    cells = np.asarray(cells, dtype=object)
    genes = np.asarray(genes, dtype=object)
    return mtx, cells, genes


def remove_zero_genes_cells(mtx, cells, genes):
    col_sums = np.asarray(mtx.sum(axis=0)).ravel()
    keep_cols = col_sums > 0
    n_omit = np.sum(~keep_cols)
    print(f"Omitted {n_omit} genes with zero counts")
    mtx = mtx[:, keep_cols]
    genes = genes[keep_cols]
    mtx = mtx.tocsr()
    row_sums = np.asarray(mtx.sum(axis=1)).ravel()
    keep_rows = row_sums > 0
    n_omit = np.sum(~keep_rows)
    print(f"Omitted {n_omit} cells with zero counts")
    mtx = mtx[keep_rows, :]
    cells = cells[keep_rows]
    mtx = mtx.tocsc()
    return mtx, cells, genes


def log_normalize(mtx, scale_factor=1e4):
    mtx = mtx.tocsr()
    inv_row_sums = 1.0 / np.asarray(mtx.sum(axis=1)).ravel()
    mtx = mtx.multiply(inv_row_sums[:, None]) * scale_factor
    mtx.data = np.log1p(mtx.data)
    mtx = mtx.tocsc()
    return mtx


def choose_mtx_rank(mtx, n_iter, seed, n_comps=100, thresh=6, noise_start=80):
    assert n_comps < min(mtx.shape)
    assert noise_start <= n_comps - 5
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


def compute_low_rank_reconstruction(mtx, rank, n_iter, seed):
    u, s, vh = randomized_svd(mtx, n_components=rank, n_iter=n_iter, random_state=seed)
    recon_mtx = u @ np.diag(s) @ vh
    print("Computed low-rank reconstruction")
    return recon_mtx


def reconstruct_col_block(us, vh, start, stop):
    return us @ vh[:, start:stop]


def threshold_block_sparse(recon_block, thresholds):
    mask = recon_block > thresholds[np.newaxis, :]
    thresh_block = sp.csc_matrix(recon_block * mask)
    thresh_block.eliminate_zeros()
    return thresh_block


def threshold_full_sparse(recon_mtx, quantile_prob):
    thresholds = np.abs(np.quantile(recon_mtx, quantile_prob, axis=0))
    return threshold_block_sparse(recon_mtx, thresholds)


def scale_thresholded_matrix_sparse(thresh_mtx, to_scale, sigma_1_2, to_add):
    imputed_mtx = sp.csc_matrix(thresh_mtx)

    for j in np.where(to_scale)[0]:
        start, end = imputed_mtx.indptr[j], imputed_mtx.indptr[j + 1]
        if start == end:
            continue
        vals = imputed_mtx.data[start:end]
        vals = vals * sigma_1_2[j] + to_add[j]
        imputed_mtx.data[start:end] = vals

    print(f"Scaled all except {np.sum(~to_scale)} genes")
    return imputed_mtx


def compute_scaling_factors_sparse(thresh_block, mtx_block):
    thresh_block = thresh_block.tocsc()
    mtx_block = mtx_block.tocsc()

    n_col = thresh_block.shape[1]

    sigma_1 = np.zeros(n_col, dtype=float)
    mu_1 = np.zeros(n_col, dtype=float)

    for j in range(n_col):
        start, end = thresh_block.indptr[j], thresh_block.indptr[j + 1]
        vals = thresh_block.data[start:end]
        if vals.size > 0:
            sigma_1[j] = np.std(vals)
            mu_1[j] = np.mean(vals)

    sigma_2 = sparse_col_std_nnz(mtx_block)
    mu_2 = sparse_col_mean_nnz(mtx_block)

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

    print("Computed scaling factors")
    return to_scale, sigma_1_2, to_add


def zero_negatives_sparse(imputed_block):
    neg = imputed_block.data < 0
    pct_neg = round(100 * np.sum(neg) / (imputed_block.shape[0] * imputed_block.shape[1]), 2)
    imputed_block.data[neg] = 0
    imputed_block.eliminate_zeros()
    print(f"{pct_neg}% of the values became negative in the scaling process and were set to zero")
    return imputed_block


def restore_observed_values_sparse(imputed_block, mtx_block, pres_obs="zeroed"):
    mtx_coo = mtx_block.tocoo()
    imputed_lil = imputed_block.tolil()

    if pres_obs == "zeroed":
        for i, j, v in zip(mtx_coo.row, mtx_coo.col, mtx_coo.data):
            if imputed_lil[i, j] == 0:
                imputed_lil[i, j] = v
    elif pres_obs == "all":
        for i, j, v in zip(mtx_coo.row, mtx_coo.col, mtx_coo.data):
            imputed_lil[i, j] = v
    elif pres_obs == "none":
        pass
    else:
        raise ValueError("pres_obs must be 'zeroed', 'all', or 'none'")

    return imputed_lil.tocsc()


def report_density(mtx, imputed_mtx):
    start_nnz = round(100 * mtx.nnz / (mtx.shape[0] * mtx.shape[1]), 2)
    end_nnz = round(100 * imputed_mtx.nnz / (imputed_mtx.shape[0] * imputed_mtx.shape[1]), 2)
    print(f"Original nonzero values: {start_nnz}%")
    print(f"Imputed nonzero values: {end_nnz}%")


def impute(mtx, rank, n_iter, quantile_prob, pres_obs, seed):
    recon_mtx = compute_low_rank_reconstruction(mtx, rank, n_iter, seed)
    thresh_mtx = threshold_full_sparse(recon_mtx, quantile_prob)
    del recon_mtx

    to_scale, sigma_1_2, to_add = compute_scaling_factors_sparse(thresh_mtx, mtx)
    imputed_mtx = scale_thresholded_matrix_sparse(thresh_mtx, to_scale, sigma_1_2, to_add)
    imputed_mtx = zero_negatives_sparse(imputed_mtx)
    imputed_mtx = restore_observed_values_sparse(imputed_mtx, mtx, pres_obs=pres_obs)
    imputed_mtx.eliminate_zeros()

    report_density(mtx, imputed_mtx)
    return imputed_mtx


def halra(mtx, cells, genes, normalize=False, n_iter=12,
          quantile_prob=0.001, mtx_rank=None, pres_obs="zeroed", seed=0):
    mtx, cells, genes = parse_input(mtx, cells, genes)
    mtx, cells, genes = remove_zero_genes_cells(mtx, cells, genes)
    if normalize: mtx = log_normalize(mtx)
    n_row, n_col = mtx.shape
    print(f"Read matrix with {n_row} cells and {n_col} genes")
    if mtx_rank is None: mtx_rank = choose_mtx_rank(mtx, n_iter, seed)
    return impute(mtx, mtx_rank, n_iter, quantile_prob, pres_obs, seed)
