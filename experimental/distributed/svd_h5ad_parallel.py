"""Distributed HALRA SVD utilities.

This module assumes the input h5ad has a normalized CSR matrix stored at
``adata.layers[csr_layer_name]``. Each MPI rank reads a contiguous row block
and participates in a row-distributed randomized SVD.
"""

from __future__ import annotations

import h5py
import numpy as np
from mpi4py import MPI
from scipy import sparse

from read_write_h5ad_parallel import (
    get_csr_layer_shape,
    read_local_csr_layer_block,
)


def _qr_reduced(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[0] == 0:
        return np.empty((0, 0), dtype=X.dtype), np.empty((0, X.shape[1]), dtype=X.dtype)
    return np.linalg.qr(X, mode="reduced")


def distributed_tsqr(Y_local: np.ndarray, comm: MPI.Comm = MPI.COMM_WORLD) -> np.ndarray:
    """Tall-skinny QR for a row-distributed dense matrix.

    Each rank owns a row block of ``Y``. The returned ``Q_local`` has the same
    number of local rows and globally orthonormal columns.
    """
    Q_local, R_local = _qr_reduced(np.asarray(Y_local))

    R_blocks = comm.allgather(R_local)
    R_row_counts = np.asarray([R.shape[0] for R in R_blocks], dtype=np.int64)

    if R_row_counts.sum() == 0:
        return np.empty((Y_local.shape[0], 0), dtype=Y_local.dtype)

    R_stacked = np.vstack(R_blocks)
    Q2, _ = np.linalg.qr(R_stacked, mode="reduced")

    rank = comm.Get_rank()
    q2_start = int(R_row_counts[:rank].sum())
    q2_end = q2_start + int(R_row_counts[rank])

    return Q_local @ Q2[q2_start:q2_end, :]


def _allreduce_sum(local: np.ndarray, comm: MPI.Comm) -> np.ndarray:
    local = np.ascontiguousarray(local)
    global_arr = np.empty_like(local)
    comm.Allreduce(local, global_arr, op=MPI.SUM)
    return global_arr


def compute_distributed_svd(
    in_path: str,
    rank: int,
    oversample: int = 20,
    n_iter: int = 2,
    seed: int = 0,
    csr_layer_name: str = "csr",
    comm: MPI.Comm = MPI.COMM_WORLD,
    work_dtype=np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Compute a row-distributed randomized SVD from a CSR h5ad layer.

    Parameters
    ----------
    in_path
        Path to an h5ad file whose ``layers[csr_layer_name]`` is CSR.
    rank
        Target SVD rank.
    oversample
        Extra randomized range-finder dimensions.
    n_iter
        Number of randomized power iterations.
    seed
        Random seed used identically on all ranks.
    csr_layer_name
        Name of the CSR layer, usually ``"csr"``.
    comm
        MPI communicator.
    work_dtype
        Floating dtype for dense randomized SVD work arrays.

    Returns
    -------
    U_local
        This rank's row block of left singular vectors, shape
        ``(local_n_obs, rank)``.
    s
        Singular values, replicated on all ranks.
    vh
        Right singular vectors, replicated on all ranks, shape
        ``(rank, n_vars)``.
    row_range
        ``(start, end)`` global row bounds owned by this rank.
    """
    A_local, start, end = read_local_csr_layer_block(
        in_path,
        comm=comm,
        csr_layer_name=csr_layer_name,
    )

    n_obs, n_vars = get_csr_layer_shape(in_path, csr_layer_name=csr_layer_name)
    ell = min(rank + oversample, n_obs, n_vars)
    if rank > ell:
        raise ValueError(f"rank={rank} is too large for matrix shape {(n_obs, n_vars)}")

    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((n_vars, ell)).astype(work_dtype, copy=False)

    Y = (A_local @ omega).astype(work_dtype, copy=False)

    for _ in range(n_iter):
        Q_local = distributed_tsqr(Y, comm=comm)

        Z_local = (A_local.T @ Q_local).astype(work_dtype, copy=False)
        Z = _allreduce_sum(Z_local, comm)
        Z, _ = np.linalg.qr(Z, mode="reduced")

        Y = (A_local @ Z).astype(work_dtype, copy=False)

    Q_local = distributed_tsqr(Y, comm=comm)

    B_local = (Q_local.T @ A_local).astype(work_dtype, copy=False)
    B = _allreduce_sum(B_local, comm)

    U_hat, s, vh = np.linalg.svd(B, full_matrices=False)

    U_local = Q_local @ U_hat[:, :rank]
    s = s[:rank]
    vh = vh[:rank, :]

    return U_local, s, vh, (start, end)
