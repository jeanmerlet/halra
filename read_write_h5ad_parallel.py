from __future__ import annotations

import h5py
import numpy as np
import anndata as ad
from scipy import sparse
from mpi4py import MPI


_INT_DTYPE = np.int64
_INDEX_DTYPE = np.int64

# Keep individual HDF5 hyperslab writes comfortably below sizes that can
# trigger HDF5/h5py "size to size_i" conversion errors on large sparse arrays.
_DEFAULT_WRITE_BLOCK = 10_000_000


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


def _write_1d_blocks(dataset, dst_start, values, dtype=None, block_size=_DEFAULT_WRITE_BLOCK):
    """Write a 1D array to an HDF5 dataset in bounded contiguous blocks."""
    values = np.asarray(values)
    n = int(values.shape[0])

    if n == 0:
        return

    dst_start = int(dst_start)

    for src_start in range(0, n, block_size):
        src_end = min(src_start + block_size, n)
        block = values[src_start:src_end]

        if dtype is not None:
            block = np.asarray(block, dtype=dtype)

        block = np.ascontiguousarray(block)

        out_start = dst_start + src_start
        out_end = dst_start + src_end
        dataset[out_start:out_end] = block


def _write_sparse_column_blocks(
    x_data,
    x_indices,
    X_csc,
    local_col_offsets,
    row_offset,
    data_dtype,
    write_block_size,
):
    """Write a local CSC chunk into global CSC datasets column by column."""
    row_offset = np.int64(row_offset)

    for col in range(X_csc.shape[1]):
        src_start, src_end = X_csc.indptr[col], X_csc.indptr[col + 1]
        if src_start == src_end:
            continue

        dst_start = int(local_col_offsets[col])

        _write_1d_blocks(
            x_data,
            dst_start,
            X_csc.data[src_start:src_end],
            dtype=data_dtype,
            block_size=write_block_size,
        )

        # Cast before adding row_offset, otherwise SciPy's int32 indices can
        # overflow before the final HDF5 write.
        global_rows = (
            X_csc.indices[src_start:src_end].astype(_INDEX_DTYPE, copy=False)
            + row_offset
        )
        _write_1d_blocks(
            x_indices,
            dst_start,
            global_rows,
            dtype=_INDEX_DTYPE,
            block_size=write_block_size,
        )


def write_h5ad_parallel_csc_with_csr_layer(
    in_path,
    out_path,
    X_csr,
    kept_input_rows,
    n_vars,
    comm,
    data_dtype=np.float32,
    csr_layer_name="csr",
    write_block_size=_DEFAULT_WRITE_BLOCK,
):
    """Write normalized data to one h5ad using parallel HDF5.

    ``X`` is written as CSC for column/gene-oriented HALRA steps.
    ``layers[csr_layer_name]`` is written as CSR for row-distributed SVD.

    This fast MPI-HDF5 path intentionally does not use HDF5 compression,
    because filter compression is not compatible with these independent
    parallel sparse writes.

    Large 1D sparse arrays are written in bounded blocks to avoid HDF5/h5py
    failures on very large hyperslab writes.
    """
    if not h5py.get_config().mpi:
        raise RuntimeError("This function requires h5py compiled with MPI support.")

    rank = comm.Get_rank()

    X_csr = X_csr.tocsr()
    X_csc = X_csr.tocsc()

    local_n_obs = np.int64(X_csr.shape[0])
    local_nnz = np.int64(X_csr.nnz)
    local_col_counts = np.diff(X_csc.indptr).astype(_INT_DTYPE, copy=False)

    all_n_obs = np.asarray(comm.allgather(local_n_obs), dtype=_INT_DTYPE)
    all_nnz = np.asarray(comm.allgather(local_nnz), dtype=_INT_DTYPE)
    all_col_counts = np.vstack(comm.allgather(local_col_counts)).astype(_INT_DTYPE, copy=False)

    n_obs_out = int(all_n_obs.sum())
    row_offset = np.int64(all_n_obs[:rank].sum())
    nnz_offset = np.int64(all_nnz[:rank].sum())
    nnz = int(all_nnz.sum())

    col_counts = all_col_counts.sum(axis=0, dtype=_INT_DTYPE)
    csc_indptr = np.empty(n_vars + 1, dtype=_INT_DTYPE)
    csc_indptr[0] = 0
    csc_indptr[1:] = np.cumsum(col_counts, dtype=_INT_DTYPE)

    local_col_offsets = csc_indptr[:-1] + all_col_counts[:rank].sum(axis=0, dtype=_INT_DTYPE)

    with h5py.File(out_path, "w", driver="mpio", comm=comm) as f:
        f.attrs["encoding-type"] = "anndata"
        f.attrs["encoding-version"] = "0.1.0"

        x_group = _create_sparse_group(f, "X", "csc_matrix", (n_obs_out, n_vars))
        x_data = x_group.create_dataset("data", shape=(nnz,), dtype=data_dtype)
        x_indices = x_group.create_dataset("indices", shape=(nnz,), dtype=_INDEX_DTYPE)
        x_group.create_dataset("indptr", data=csc_indptr, dtype=_INT_DTYPE)

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
        csr_indices = csr_group.create_dataset("indices", shape=(nnz,), dtype=_INDEX_DTYPE)
        csr_indptr = csr_group.create_dataset("indptr", shape=(n_obs_out + 1,), dtype=_INT_DTYPE)

        _write_sparse_column_blocks(
            x_data=x_data,
            x_indices=x_indices,
            X_csc=X_csc,
            local_col_offsets=local_col_offsets,
            row_offset=row_offset,
            data_dtype=data_dtype,
            write_block_size=write_block_size,
        )

        local_start = int(nnz_offset)

        _write_1d_blocks(
            csr_data,
            local_start,
            X_csr.data,
            dtype=data_dtype,
            block_size=write_block_size,
        )

        _write_1d_blocks(
            csr_indices,
            local_start,
            X_csr.indices.astype(_INDEX_DTYPE, copy=False),
            dtype=_INDEX_DTYPE,
            block_size=write_block_size,
        )

        if rank == 0:
            csr_indptr[0] = np.array(0, dtype=_INT_DTYPE)

        if local_n_obs > 0:
            ptr_start = int(row_offset) + 1

            # Critical Tahoe-scale fix:
            # X_csr.indptr is commonly int32. Cast to int64 BEFORE adding the
            # global nnz_offset; otherwise NumPy tries to perform int32 addition
            # and raises OverflowError when nnz_offset > 2^31 - 1.
            global_indptr = (
                X_csr.indptr[1:].astype(_INT_DTYPE, copy=False)
                + nnz_offset
            )

            _write_1d_blocks(
                csr_indptr,
                ptr_start,
                global_indptr,
                dtype=_INT_DTYPE,
                block_size=write_block_size,
            )

    kept_rows_by_rank = comm.gather(np.asarray(kept_input_rows, dtype=_INT_DTYPE), root=0)

    if rank == 0:
        kept_rows = (
            np.concatenate(kept_rows_by_rank)
            if kept_rows_by_rank
            else np.array([], dtype=_INT_DTYPE)
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
        indices = group["indices"][data_start:data_end].astype(_INDEX_DTYPE, copy=False)
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
