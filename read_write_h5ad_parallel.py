from __future__ import annotations
import h5py
import numpy as np
import anndata as ad
from scipy import sparse
from mpi4py import MPI


_INT_DTYPE = np.int64


def get_h5ad_shape(path):
    with h5py.File(path, "r") as f:
        return tuple(f["X"].attrs["shape"])


def read_h5ad_row_chunk(path, start, end):
    adata = ad.read_h5ad(path, backed="r")
    X = adata.X[start:end, :]
    adata.file.close()

    if sparse.issparse(X):
        return X.tocsr()

    return sparse.csr_matrix(X)


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode()
    return value


def _read_string_index(f, group_name):
    group = f[group_name]
    index_key = _decode_attr(group.attrs.get("_index", "_index"))
    values = group[index_key][:]

    if values.dtype.kind == "S":
        return values.astype(str)

    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in values])


def _write_minimal_dataframe_group(f, name, index_values):
    group = f.create_group(name)
    group.attrs["encoding-type"] = "dataframe"
    group.attrs["encoding-version"] = "0.2.0"
    group.attrs["_index"] = "_index"
    group.attrs["column-order"] = np.array([], dtype="S1")

    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset("_index", data=np.asarray(index_values, dtype=object), dtype=dtype)


def _write_empty_mapping(f, name):
    group = f.create_group(name)
    group.attrs["encoding-type"] = "dict"
    group.attrs["encoding-version"] = "0.1.0"


def _write_minimal_h5ad_metadata(in_path, out_path, kept_input_rows):
    with h5py.File(in_path, "r") as src:
        obs_names = _read_string_index(src, "obs")[kept_input_rows]
        var_names = _read_string_index(src, "var")

    with h5py.File(out_path, "r+") as dst:
        _write_minimal_dataframe_group(dst, "obs", obs_names)
        _write_minimal_dataframe_group(dst, "var", var_names)

        for name in ["uns", "obsm", "varm", "obsp", "varp"]:
            _write_empty_mapping(dst, name)


def _create_sparse_group(parent, name, encoding_type, shape):
    group = parent.create_group(name)
    group.attrs["encoding-type"] = encoding_type
    group.attrs["encoding-version"] = "0.1.0"
    group.attrs["shape"] = np.asarray(shape, dtype=np.int64)
    return group


def write_h5ad_parallel_csc_with_csr_layer(
    in_path,
    out_path,
    X_csr,
    kept_input_rows,
    n_vars,
    comm,
    data_dtype=np.float32,
    csr_layer_name="csr",
):
    """Write normalized data to one h5ad using parallel HDF5.

    ``X`` is written as CSC for column/gene-oriented HALRA steps.
    ``layers[csr_layer_name]`` is written as CSR for row-distributed SVD.

    This fast MPI-HDF5 path intentionally does not use HDF5 compression,
    because filter compression is not compatible with these independent
    parallel sparse writes.
    """
    if not h5py.get_config().mpi:
        raise RuntimeError("This function requires h5py compiled with MPI support.")

    rank = comm.Get_rank()

    X_csr = X_csr.tocsr()
    X_csc = X_csr.tocsc()

    local_n_obs = np.int64(X_csr.shape[0])
    local_nnz = np.int64(X_csr.nnz)
    local_col_counts = np.diff(X_csc.indptr).astype(np.int64, copy=False)

    all_n_obs = np.asarray(comm.allgather(local_n_obs), dtype=np.int64)
    all_nnz = np.asarray(comm.allgather(local_nnz), dtype=np.int64)
    all_col_counts = np.vstack(comm.allgather(local_col_counts))

    n_obs_out = int(all_n_obs.sum())
    row_offset = int(all_n_obs[:rank].sum())
    nnz_offset = int(all_nnz[:rank].sum())
    nnz = int(all_nnz.sum())

    col_counts = all_col_counts.sum(axis=0, dtype=np.int64)
    csc_indptr = np.empty(n_vars + 1, dtype=np.int64)
    csc_indptr[0] = 0
    csc_indptr[1:] = np.cumsum(col_counts)

    local_col_offsets = csc_indptr[:-1] + all_col_counts[:rank].sum(axis=0, dtype=np.int64)

    with h5py.File(out_path, "w", driver="mpio", comm=comm) as f:
        f.attrs["encoding-type"] = "anndata"
        f.attrs["encoding-version"] = "0.1.0"

        x_group = _create_sparse_group(f, "X", "csc_matrix", (n_obs_out, n_vars))
        x_data = x_group.create_dataset("data", shape=(nnz,), dtype=data_dtype)
        x_indices = x_group.create_dataset("indices", shape=(nnz,), dtype=np.int64)
        x_group.create_dataset("indptr", data=csc_indptr, dtype=np.int64)

        layers_group = f.create_group("layers")
        layers_group.attrs["encoding-type"] = "dict"
        layers_group.attrs["encoding-version"] = "0.1.0"

        csr_group = _create_sparse_group(
            layers_group,
            csr_layer_name,
            "csr_matrix",
            (n_obs_out, n_vars),
        )
        csr_data = csr_group.create_dataset("data", shape=(nnz,), dtype=data_dtype)
        csr_indices = csr_group.create_dataset("indices", shape=(nnz,), dtype=np.int64)
        csr_indptr = csr_group.create_dataset("indptr", shape=(n_obs_out + 1,), dtype=np.int64)

        for col in range(n_vars):
            src_start, src_end = X_csc.indptr[col], X_csc.indptr[col + 1]
            if src_start == src_end:
                continue

            dst_start = int(local_col_offsets[col])
            dst_end = dst_start + (src_end - src_start)

            x_data[dst_start:dst_end] = X_csc.data[src_start:src_end].astype(
                data_dtype,
                copy=False,
            )
            x_indices[dst_start:dst_end] = X_csc.indices[src_start:src_end] + row_offset

        local_start = nnz_offset
        local_end = nnz_offset + int(local_nnz)

        csr_data[local_start:local_end] = X_csr.data.astype(data_dtype, copy=False)
        csr_indices[local_start:local_end] = X_csr.indices.astype(np.int64, copy=False)

        if rank == 0:
            csr_indptr[0] = 0

        if local_n_obs > 0:
            ptr_start = row_offset + 1
            ptr_end = row_offset + int(local_n_obs) + 1
            csr_indptr[ptr_start:ptr_end] = X_csr.indptr[1:] + nnz_offset

    kept_rows_by_rank = comm.gather(np.asarray(kept_input_rows, dtype=np.int64), root=0)

    if rank == 0:
        kept_rows = (
            np.concatenate(kept_rows_by_rank)
            if kept_rows_by_rank
            else np.array([], dtype=np.int64)
        )
        _write_minimal_h5ad_metadata(in_path, out_path, kept_rows)

    comm.Barrier()


def row_bounds(n_rows: int, rank: int, size: int) -> tuple[int, int]:
    bounds = np.linspace(0, n_rows, size + 1, dtype=np.int64)
    return int(bounds[rank]), int(bounds[rank + 1])


def get_csr_layer_shape(path: str, csr_layer_name: str = "csr") -> tuple[int, int]:
    with h5py.File(path, "r") as f:
        group = f["layers"][csr_layer_name]
        return tuple(int(x) for x in group.attrs["shape"])


def read_csr_layer_row_block(
    path: str,
    start: int,
    end: int,
    csr_layer_name: str = "csr",
) -> sparse.csr_matrix:
    """Read rows ``start:end`` from a CSR matrix stored in ``layers``.

    Because the source is CSR, the requested row block corresponds to one
    contiguous slice of ``data`` and ``indices``.
    """
    with h5py.File(path, "r") as f:
        group = f["layers"][csr_layer_name]
        n_obs, n_vars = (int(x) for x in group.attrs["shape"])

        if start < 0 or end < start or end > n_obs:
            raise ValueError(f"Invalid row bounds: start={start}, end={end}, n_obs={n_obs}")

        indptr = group["indptr"][start : end + 1].astype(_INT_DTYPE, copy=False)
        data_start = int(indptr[0])
        data_end = int(indptr[-1])

        data = group["data"][data_start:data_end]
        indices = group["indices"][data_start:data_end].astype(_INT_DTYPE, copy=False)
        local_indptr = indptr - data_start

    return sparse.csr_matrix((data, indices, local_indptr), shape=(end - start, n_vars))


def read_local_csr_layer_block(
    path: str,
    comm: MPI.Comm = MPI.COMM_WORLD,
    csr_layer_name: str = "csr",
) -> tuple[sparse.csr_matrix, int, int]:
    """Read this rank's row block from ``adata.layers[csr_layer_name]``."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_obs, _ = get_csr_layer_shape(path, csr_layer_name=csr_layer_name)
    start, end = row_bounds(n_obs, rank, size)
    return read_csr_layer_row_block(path, start, end, csr_layer_name), start, end

