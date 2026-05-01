import numpy as np
from mpi4py import MPI
from scipy import sparse

from read_write_h5ad_parallel import (
    get_h5ad_shape,
    read_h5ad_row_chunk,
    write_h5ad_parallel_csc_with_csr_layer,
)


def log_normalize_counts(X, target_sum=1e4, data_dtype=np.float32):
    """Log-normalize a CSR count matrix with minimal extra memory.

    This version intentionally does not drop all-zero rows. Avoiding row
    filtering prevents a large temporary copy at Tahoe100M scale and preserves
    the original row count/order in the output h5ad.
    """
    if not sparse.isspmatrix_csr(X):
        X = sparse.csr_matrix(X)

    X = X.tocsr(copy=False)

    if X.data.dtype != np.dtype(data_dtype):
        X.data = X.data.astype(data_dtype, copy=False)

    counts = np.asarray(X.sum(axis=1)).ravel()
    keep = counts > 0

    scale = np.zeros(counts.shape, dtype=data_dtype)
    scale[keep] = target_sum / counts[keep]

    row_nnz = np.diff(X.indptr)

    # One scale factor per stored value. This avoids sparse.diags(scale) @ X.
    X.data *= np.repeat(scale, row_nnz)
    np.log1p(X.data, out=X.data)

    return X, keep


def row_bounds(n_rows, rank, size):
    bounds = np.linspace(0, n_rows, size + 1, dtype=np.int64)
    return int(bounds[rank]), int(bounds[rank + 1])


def normalize_h5ad_parallel(
    in_path,
    out_path,
    target_sum=1e4,
    data_dtype=np.float32,
    csr_layer_name="csr",
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_obs, n_vars = get_h5ad_shape(in_path)
    start, end = row_bounds(n_obs, rank, size)

    X = read_h5ad_row_chunk(in_path, start, end)
    X_norm, _ = log_normalize_counts(
        X,
        target_sum=target_sum,
        data_dtype=data_dtype,
    )

    # Keep all rows to avoid expensive row filtering/copying. This preserves
    # original obs_names and lets the writer copy metadata without dropping rows.
    kept_input_rows = np.arange(start, end, dtype=np.int64)

    write_h5ad_parallel_csc_with_csr_layer(
        in_path=in_path,
        out_path=out_path,
        X_csr=X_norm,
        kept_input_rows=kept_input_rows,
        n_vars=n_vars,
        comm=comm,
        data_dtype=data_dtype,
        csr_layer_name=csr_layer_name,
    )
