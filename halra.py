import numpy as np
import scipy.sparse as sp

def log_normalize_counts(mtx, scale_factor=1e4):
    # ensure sparse CSR
    if isinstance(mtx, np.ndarray):
        mtx = sp.csr_matrix(mtx)
    elif not sp.issparse(mtx):
        raise TypeError("Input must be numpy.ndarray or scipy.sparse matrix")
    else:
        mtx = mtx.tocsr()

    # remove zero-sum rows
    row_sums = np.asarray(mtx.sum(axis=1)).ravel()
    keep = row_sums > 0
    n_omit = np.sum(~keep)
    if n_omit > 0: print(f"Omitted {n_omit} cells with zero counts")
    mtx = mtx[keep]
    row_sums = row_sums[keep]

    # normalize + scale
    inv_row_sums = 1.0 / row_sums
    mtx = mtx.multiply(inv_row_sums[:, None]) * scale_factor

    # log1p transform (preserve sparsity)
    mtx.data = np.log1p(mtx.data)

    return mtx
