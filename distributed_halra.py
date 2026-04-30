"""Distributed HALRA column-block imputation utilities.

This module assumes the input h5ad has:

* ``adata.X`` stored as CSC, used for gene/column-oriented HALRA steps.
* ``adata.layers['csr']`` stored as CSR, used by ``svd_h5ad_parallel`` for
  row-distributed randomized SVD.

This first implementation returns each rank's imputed CSC column block instead
of writing the final matrix to disk. The downstream imputation is distributed by
columns/genes.
"""

from __future__ import annotations

import h5py
import numpy as np
from mpi4py import MPI
from scipy import sparse

from read_write_h5ad_parallel import get_h5ad_shape
from svd_h5ad_parallel import compute_distributed_svd


_INT_DTYPE = np.int64


def column_bounds(n_cols: int, rank: int, size: int) -> tuple[int, int]:
    bounds = np.linspace(0, n_cols, size + 1, dtype=np.int64)
    return int(bounds[rank]), int(bounds[rank + 1])


def _read_csc_column_block(path: str, start: int, end: int) -> sparse.csc_matrix:
    """Read columns ``start:end`` from ``adata.X`` assuming CSC storage."""
    with h5py.File(path, "r") as f:
        group = f["X"]
        encoding = group.attrs.get("encoding-type")
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        if encoding != "csc_matrix":
            raise ValueError(f"Expected adata.X to be csc_matrix, found {encoding!r}")

        n_obs, n_vars = (int(x) for x in group.attrs["shape"])
        if start < 0 or end < start or end > n_vars:
            raise ValueError(f"Invalid column bounds: start={start}, end={end}, n_vars={n_vars}")

        indptr = group["indptr"][start : end + 1].astype(_INT_DTYPE, copy=False)
        data_start = int(indptr[0])
        data_end = int(indptr[-1])

        data = group["data"][data_start:data_end]
        indices = group["indices"][data_start:data_end].astype(_INT_DTYPE, copy=False)
        local_indptr = indptr - data_start

    return sparse.csc_matrix((data, indices, local_indptr), shape=(n_obs, end - start))


def _gather_distributed_u(U_local: np.ndarray, comm: MPI.Comm) -> np.ndarray:
    """Gather row-distributed U onto every rank.

    This is intentionally simple for the first downstream HALRA implementation.
    It is not the final Tahoe100M-scalable strategy because it replicates U.
    """
    blocks = comm.allgather(np.asarray(U_local))
    return np.vstack(blocks)


def reconstruct_column_block(
    U: np.ndarray,
    s: np.ndarray,
    vh_block: np.ndarray,
    dtype=np.float32,
) -> np.ndarray:
    """Create dense low-rank reconstruction for this rank's gene block."""
    return ((U * s) @ vh_block).astype(dtype, copy=False)


def threshold_reconstruction_block(
    recon_block: np.ndarray,
    quantile_prob: float,
) -> sparse.csc_matrix:
    """Threshold a dense reconstructed gene block and return sparse CSC."""
    thresholds = np.abs(np.quantile(recon_block, quantile_prob, axis=0))
    mask = recon_block > thresholds[np.newaxis, :]
    thresh_block = sparse.csc_matrix(recon_block * mask)
    thresh_block.eliminate_zeros()
    return thresh_block


def _column_mean_std_nnz_csc(matrix: sparse.csc_matrix) -> tuple[np.ndarray, np.ndarray]:
    matrix = matrix.tocsc()
    n_cols = matrix.shape[1]
    means = np.zeros(n_cols, dtype=np.float64)
    stds = np.zeros(n_cols, dtype=np.float64)

    for j in range(n_cols):
        start, end = matrix.indptr[j], matrix.indptr[j + 1]
        values = matrix.data[start:end]
        if values.size:
            means[j] = values.mean()
            stds[j] = values.std()

    return means, stds


def create_scaling_factors_block(
    thresh_block: sparse.csc_matrix,
    input_block: sparse.csc_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-gene scaling factors for one local CSC column block."""
    mu_1, sigma_1 = _column_mean_std_nnz_csc(thresh_block)
    mu_2, sigma_2 = _column_mean_std_nnz_csc(input_block)

    to_scale = sigma_1 != 0
    scale_factors = np.divide(
        sigma_2,
        sigma_1,
        out=np.zeros_like(sigma_2, dtype=np.float64),
        where=to_scale,
    )
    offsets = mu_2 - mu_1 * scale_factors

    return to_scale, scale_factors, offsets


def apply_scaling_block(
    thresh_block: sparse.csc_matrix,
    to_scale: np.ndarray,
    scale_factors: np.ndarray,
    offsets: np.ndarray,
) -> sparse.csc_matrix:
    """Apply per-gene scaling factors to a local CSC column block."""
    imputed_block = thresh_block.copy().tocsc()

    for j in np.where(to_scale)[0]:
        start, end = imputed_block.indptr[j], imputed_block.indptr[j + 1]
        if start == end:
            continue
        imputed_block.data[start:end] = imputed_block.data[start:end] * scale_factors[j] + offsets[j]

    return imputed_block


def clip_negative_values_block(imputed_block: sparse.csc_matrix) -> tuple[sparse.csc_matrix, int]:
    """Set negative imputed values to zero for one local block."""
    neg = imputed_block.data < 0
    n_neg = int(np.sum(neg))
    if n_neg:
        imputed_block.data[neg] = 0
        imputed_block.eliminate_zeros()
    return imputed_block, n_neg


def restore_observed_values_block(
    imputed_block: sparse.csc_matrix,
    input_block: sparse.csc_matrix,
) -> sparse.csc_matrix:
    """Restore original nonzero values missing from the imputed sparse block."""
    imputed_block = imputed_block.tocsc()
    input_block = input_block.tocsc()

    imputed_mask = imputed_block.copy()
    imputed_mask.data = np.ones_like(imputed_mask.data)

    observed_retained = input_block.multiply(imputed_mask)
    observed_missing = input_block - observed_retained
    restored = imputed_block + observed_missing
    restored.eliminate_zeros()
    return restored.tocsc()


def _report_distributed_density(
    input_block: sparse.csc_matrix,
    imputed_block: sparse.csc_matrix,
    n_neg_local: int,
    comm: MPI.Comm,
) -> None:
    rank = comm.Get_rank()
    local = np.array([input_block.nnz, imputed_block.nnz, n_neg_local], dtype=np.int64)
    total = np.empty_like(local)
    comm.Allreduce(local, total, op=MPI.SUM)

    n_obs = input_block.shape[0]
    n_cols_local = np.array(input_block.shape[1], dtype=np.int64)
    n_cols_total = np.array(0, dtype=np.int64)
    comm.Allreduce(n_cols_local, n_cols_total, op=MPI.SUM)

    if rank == 0:
        denom = float(n_obs * int(n_cols_total))
        start_density = 100.0 * int(total[0]) / denom if denom else 0.0
        end_density = 100.0 * int(total[1]) / denom if denom else 0.0
        neg_density = 100.0 * int(total[2]) / denom if denom else 0.0
        print(f"Original nonzero values: {start_density:.2f}%")
        print(f"Imputed nonzero values: {end_density:.2f}%")
        print(f"{neg_density:.2f}% of values became negative and were set to zero")


def impute_h5ad_column_block(
    in_path: str,
    matrix_rank: int,
    quantile_prob: float = 0.001,
    svd_oversample: int = 20,
    svd_n_iter: int = 2,
    seed: int = 0,
    csr_layer_name: str = "csr",
    comm: MPI.Comm = MPI.COMM_WORLD,
    work_dtype=np.float64,
    recon_dtype=np.float32,
    report: bool = True,
) -> tuple[sparse.csc_matrix, tuple[int, int], np.ndarray, np.ndarray, tuple[int, int]]:
    """Run distributed HALRA imputation through the local column-block result.

    Returns
    -------
    imputed_block
        This rank's imputed CSC matrix block, shape ``(n_obs, local_n_genes)``.
    col_range
        Global gene-column bounds ``(start, end)`` for ``imputed_block``.
    s
        Singular values from distributed SVD, replicated on all ranks.
    vh_block
        This rank's columns of ``vh``, shape ``(matrix_rank, local_n_genes)``.
    u_row_range
        Global row bounds for this rank's original row-distributed ``U_local``.

    Notes
    -----
    This first version gathers the row-distributed ``U`` matrix onto every rank
    before doing column-block reconstruction. That keeps the downstream code
    simple and is useful for correctness/performance testing, but it should be
    replaced by a streamed/distributed-U reconstruction before Tahoe100M-scale
    runs.
    """
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    U_local, s, vh, u_row_range = compute_distributed_svd(
        in_path=in_path,
        rank=matrix_rank,
        oversample=svd_oversample,
        n_iter=svd_n_iter,
        seed=seed,
        csr_layer_name=csr_layer_name,
        comm=comm,
        work_dtype=work_dtype,
    )

    U = _gather_distributed_u(U_local, comm).astype(work_dtype, copy=False)

    n_obs, n_vars = get_h5ad_shape(in_path)
    col_start, col_end = column_bounds(n_vars, mpi_rank, mpi_size)
    input_block = _read_csc_column_block(in_path, col_start, col_end)

    vh_block = vh[:, col_start:col_end]
    recon_block = reconstruct_column_block(U, s, vh_block, dtype=recon_dtype)

    thresh_block = threshold_reconstruction_block(recon_block, quantile_prob)
    del recon_block

    to_scale, scale_factors, offsets = create_scaling_factors_block(thresh_block, input_block)
    imputed_block = apply_scaling_block(thresh_block, to_scale, scale_factors, offsets)
    del thresh_block

    imputed_block, n_neg = clip_negative_values_block(imputed_block)
    imputed_block = restore_observed_values_block(imputed_block, input_block)

    if report:
        _report_distributed_density(input_block, imputed_block, n_neg, comm)

    return imputed_block, (col_start, col_end), s, vh_block, u_row_range
