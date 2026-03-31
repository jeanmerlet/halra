import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd

def log_normalize_counts(mtx, scale_factor=1e4):
    # ensure sparse CSR
    if isinstance(mtx, np.ndarray):
        mtx = sp.csr_matrix(mtx)
    elif not sp.issparse(mtx):
        raise TypeError("Input must be numpy.ndarray or scipy.sparse matrix")
    else:
        mtx = mtx.tocsr()
    # remove zero-sum cols
    col_sums = np.asarray(mtx.sum(axis=0)).ravel()
    keep = col_sums > 0
    n_omit = np.sum(~keep)
    if n_omit > 0: print(f"Omitted {n_omit} genes with zero counts")
    mtx = mtx[:, keep]
    # remove zero-sum rows
    row_sums = np.asarray(mtx.sum(axis=1)).ravel()
    keep = row_sums > 0
    n_omit = np.sum(~keep)
    if n_omit > 0: print(f"Omitted {n_omit} cells with zero counts")
    mtx = mtx[keep, :]
    row_sums = row_sums[keep]
    # normalize + scale
    inv_row_sums = 1.0 / row_sums
    mtx = mtx.multiply(inv_row_sums[:, None]) * scale_factor
    # log1p transform (preserve sparsity)
    mtx.data = np.log1p(mtx.data)
    return mtx

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

def choose_rank(mtx, n_iter, seed, n_comps=100, thresh=6, noise_start=80):
    # mtx: scipy.sparse csr mtx
    if n_comps >= min(mtx.shape):
        raise ValueError("n_comps must be smaller than the smallest dimension of mtx.")
    if noise_start > n_comps - 5:
        raise ValueError("There need to be at least 5 singular values considered noise.")
    noise_svals = np.arange(noise_start, n_comps)
    u, s, vh = randomized_svd(mtx, n_components=n_comps, n_iter=n_iter,
                              random_state=seed)
    # Calculate the differences between consecutive singular values
    diffs = s[:-1] - s[1:]
    # Calculate mean and standard deviation of noise singular value differences
    mu = np.mean(diffs[noise_svals - 1])
    sigma = np.std(diffs[noise_svals - 1])
    # Calculate the number of standard deviations from the mean
    n_std_devs = (diffs - mu) / sigma
    # Find the largest k where n_std_devs exceeds the threshold
    rank = np.max(np.where(n_std_devs > thresh)[0]) + 1
    return rank

def std_dev_nnz(x):
    vals = x[x != 0]
    return np.std(vals) if vals.size > 0 else 0

def mean_nnz(x):
    vals = x[x != 0]
    return np.mean(vals) if vals.size > 0 else 0

def halra(mtx, rank=0, n_iter=12, quantile_prob=0.001, seed=1):
    n_row, n_col = mtx.shape
    # visual check for cell x gene for user
    print(f"Read matrix with {n_row} cells and {n_col} genes")

    # find rank if unspecified or 0
    if rank == 0:
        rank = choose_rank(mtx, n_iter, seed)
        print(f"Chose rank={rank}")
    u, s, vh = randomized_svd(mtx, n_components=rank, n_iter=n_iter,
                              random_state=seed)

    rank_mtx = np.dot(u[:, :rank], np.dot(np.diag(s[:rank]), vh[:rank, :]))
    print(f"Find the {quantile_prob} quantile of each gene")
    rank_mtx_mins = np.abs(np.quantile(rank_mtx, quantile_prob, axis=0))
    print("Sweep")
    condition = rank_mtx <= rank_mtx_mins[np.newaxis, :]
    rank_mtx_cor = np.where(condition, 0, rank_mtx)
    mtx = mtx.tocsc()
    print(type(std_dev_nnz), type(rank_mtx_cor), type(mtx))
    sigma_1 = np.apply_along_axis(std_dev_nnz, 0, rank_mtx_cor)
    sigma_2 = sparse_col_std_nnz(mtx)
    mu_1 = np.sum(rank_mtx_cor, axis=0) / np.sum(rank_mtx_cor != 0, axis=0)
    mu_2 = sparse_col_mean_nnz(mtx)
    to_scale = (~np.isnan(sigma_1)) & (~np.isnan(sigma_2)) & ~((sigma_1 == 0) & (sigma_2 == 0)) & (sigma_1 != 0)

    print(f"Scaling all except for {np.sum(~to_scale)} columns")
    sigma_1_2 = sigma_2 / sigma_1
    to_add = -mu_1 * sigma_2 / sigma_1 + mu_2

    rank_mtx_tmp = rank_mtx_cor[:, to_scale]
    rank_mtx_tmp = rank_mtx_tmp * sigma_1_2[to_scale]
    rank_mtx_tmp = rank_mtx_tmp + to_add[to_scale]

    rank_mtx_cor_sc = rank_mtx_cor.copy()
    rank_mtx_cor_sc[:, to_scale] = rank_mtx_tmp
    rank_mtx_cor_sc[rank_mtx_cor == 0] = 0

    # set negative values to 0
    neg = rank_mtx_cor_sc < 0
    rank_mtx_cor_sc[neg] = 0
    pct_neg = round(100 * np.sum(neg) / rank_mtx_cor_sc.size, 2)
    print(f"{pct_neg} of the values became negative in the scaling process and were set to zero.")

    # restore originally observed nonzero values that are zero in the imputed matrix
    coo = mtx.tocoo()
    restore = rank_mtx_cor_sc[coo.row, coo.col] == 0
    rank_mtx_cor_sc[coo.row[restore], coo.col[restore]] = coo.data[restore]

    print(type(mtx), type(rank_mtx_cor_sc))
    start_nnz = round(mtx.nnz / (mtx.shape[0] * mtx.shape[1]), 2)
    end_nnz = round(np.count_nonzero(rank_mtx_cor_sc) / rank_mtx_cor_sc.size, 2)
    print(f"Original nonzero values: {start_nnz}%")
    print(f"Imputed nonzero values: {end_nnz}%")

    return rank_mtx, rank_mtx_cor, rank_mtx_cor_sc
