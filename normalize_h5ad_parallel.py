import numpy as np
import time
from mpi4py import MPI
from scipy import sparse

from read_write_h5ad_parallel_diag import (
    get_h5ad_shape,
    read_h5ad_row_chunk,
    row_bounds,
    write_h5ad_parallel_from_subblocks,
)


def log_normalize_counts(X, target_sum=1e4, data_dtype=np.float32):
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
    X.data *= np.repeat(scale, row_nnz)
    np.log1p(X.data, out=X.data)

    return X


def normalize_h5ad_parallel(
    in_path,
    out_path,
    target_sum=1e4,
    data_dtype=np.float32,
    csr_layer_name="csr",
    sub_block_size=200_000,
    diagnostics=True,
    diag_every=1,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_obs, n_vars = get_h5ad_shape(in_path)
    start, end = row_bounds(n_obs, rank, size)

    def read_normalized_block(block_start, block_end):
        time.sleep(0.05 * (comm.Get_rank() % 32))
        X = read_h5ad_row_chunk(in_path, block_start, block_end)
        return log_normalize_counts(X, target_sum=target_sum, data_dtype=data_dtype)

    write_h5ad_parallel_from_subblocks(
        in_path=in_path,
        out_path=out_path,
        read_normalized_block=read_normalized_block,
        row_start=start,
        row_end=end,
        n_obs=n_obs,
        n_vars=n_vars,
        comm=comm,
        data_dtype=data_dtype,
        csr_layer_name=csr_layer_name,
        sub_block_size=sub_block_size,
        diagnostics=diagnostics,
        diag_every=diag_every,
    )
