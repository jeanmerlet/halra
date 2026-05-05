from __future__ import annotations

import resource
import socket

import h5py
import numpy as np
import anndata as ad
from mpi4py import MPI
from scipy import sparse


_INT_DTYPE = np.int64
_INDEX_DTYPE = np.int64
_DEFAULT_WRITE_BLOCK = 10_000_000


def get_h5ad_shape(path):
    with h5py.File(path, "r") as f:
        return tuple(int(x) for x in f["X"].attrs["shape"])


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode()
    return value


def read_h5ad_row_chunk(path, start, end):
    with h5py.File(path, "r") as f:
        group = f["X"]
        encoding = _decode_attr(group.attrs.get("encoding-type", ""))

        if encoding == "csr_matrix":
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

    adata = ad.read_h5ad(path, backed="r")
    X = adata.X[start:end, :]
    adata.file.close()
    return X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(X)


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
        dataset[dst_start + src_start : dst_start + src_end] = block


def _meminfo_gb():
    out = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                key, value = line.split(":", 1)
                out[key] = float(value.strip().split()[0]) / 1024**2
    except OSError:
        pass
    return out


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _diag_record(stage, rank, X=None, extra=None):
    mem = _meminfo_gb()
    extra = {} if extra is None else dict(extra)
    return {
        "stage": stage,
        "rank": int(rank),
        "host": socket.gethostname(),
        "rows": 0 if X is None else int(X.shape[0]),
        "cols": 0 if X is None else int(X.shape[1]),
        "nnz": 0 if X is None else int(X.nnz),
        "rss_gb": float(_rss_gb()),
        "mem_available_gb": float(mem.get("MemAvailable", np.nan)),
        **extra,
    }


def _print_diag(record, comm, prefix="[subblock_diag]"):
    rank = comm.Get_rank()
    rows = comm.gather(record, root=0)

    if rank != 0:
        return

    nnz = np.asarray([r["nnz"] for r in rows], dtype=np.float64)
    rss = np.asarray([r["rss_gb"] for r in rows], dtype=np.float64)
    avail = np.asarray([r["mem_available_gb"] for r in rows], dtype=np.float64)
    worst_nnz = rows[int(np.argmax(nnz))]
    worst_rss = rows[int(np.argmax(rss))]
    lowest_avail = rows[int(np.nanargmin(avail))] if not np.all(np.isnan(avail)) else rows[0]

    print("=" * 80, flush=True)
    print(f"{prefix} stage={rows[0]['stage']}", flush=True)
    print(
        f"{prefix} nnz min={int(np.min(nnz))} median={int(np.median(nnz))} "
        f"max={int(np.max(nnz))} imbalance_max_over_median={(np.max(nnz) / max(np.median(nnz), 1)):.3f}",
        flush=True,
    )
    print(
        f"{prefix} rss_gb min={np.min(rss):.3f} median={np.median(rss):.3f} max={np.max(rss):.3f}",
        flush=True,
    )
    print(
        f"{prefix} mem_available_gb min={np.nanmin(avail):.3f} "
        f"median={np.nanmedian(avail):.3f} max={np.nanmax(avail):.3f}",
        flush=True,
    )
    print(
        f"{prefix} worst_nnz rank={worst_nnz['rank']} host={worst_nnz['host']} "
        f"rows={worst_nnz['rows']} nnz={worst_nnz['nnz']}",
        flush=True,
    )
    print(
        f"{prefix} worst_rss rank={worst_rss['rank']} host={worst_rss['host']} "
        f"rss_gb={worst_rss['rss_gb']:.3f} available_gb={worst_rss['mem_available_gb']:.3f}",
        flush=True,
    )
    print(
        f"{prefix} lowest_available rank={lowest_avail['rank']} host={lowest_avail['host']} "
        f"available_gb={lowest_avail['mem_available_gb']:.3f} rss_gb={lowest_avail['rss_gb']:.3f}",
        flush=True,
    )


def row_bounds(n_rows: int, rank: int, size: int) -> tuple[int, int]:
    bounds = np.linspace(0, n_rows, size + 1, dtype=np.int64)
    return int(bounds[rank]), int(bounds[rank + 1])


def iter_row_subblocks(start: int, end: int, block_size: int):
    for block_start in range(start, end, block_size):
        yield block_start, min(block_start + block_size, end)


def write_h5ad_parallel_from_subblocks(
    in_path,
    out_path,
    read_normalized_block,
    row_start,
    row_end,
    n_obs,
    n_vars,
    comm,
    data_dtype=np.float32,
    csr_layer_name="csr",
    sub_block_size=50_000,
    write_block_size=_DEFAULT_WRITE_BLOCK,
    diagnostics=True,
    diag_every=10,
):
    if not h5py.get_config().mpi:
        raise RuntimeError("This function requires h5py compiled with MPI support.")

    rank = comm.Get_rank()
    local_col_counts = np.zeros(n_vars, dtype=_INT_DTYPE)
    local_nnz = np.int64(0)

    for i, (block_start, block_end) in enumerate(iter_row_subblocks(row_start, row_end, sub_block_size)):
        X_csr = read_normalized_block(block_start, block_end).tocsr()
        X_csc = X_csr.tocsc()

        local_col_counts += np.diff(X_csc.indptr).astype(_INT_DTYPE, copy=False)
        local_nnz += np.int64(X_csr.nnz)

        if diagnostics and i % diag_every == 0:
            _print_diag(
                _diag_record("count_subblock", rank, X_csr, {"block_start": block_start, "block_end": block_end}),
                comm,
            )

        del X_csc, X_csr

    local_n_obs = np.int64(row_end - row_start)

    all_n_obs = np.asarray(comm.allgather(local_n_obs), dtype=_INT_DTYPE)
    all_nnz = np.asarray(comm.allgather(local_nnz), dtype=_INT_DTYPE)
    all_col_counts = np.vstack(comm.allgather(local_col_counts)).astype(_INT_DTYPE, copy=False)

    n_obs_out = int(all_n_obs.sum())
    nnz = int(all_nnz.sum())
    row_offset = np.int64(all_n_obs[:rank].sum())
    nnz_offset = np.int64(all_nnz[:rank].sum())

    col_counts = all_col_counts.sum(axis=0, dtype=_INT_DTYPE)
    csc_indptr = np.empty(n_vars + 1, dtype=_INT_DTYPE)
    csc_indptr[0] = 0
    csc_indptr[1:] = np.cumsum(col_counts, dtype=_INT_DTYPE)

    csc_write_pos = csc_indptr[:-1] + all_col_counts[:rank].sum(axis=0, dtype=_INT_DTYPE)
    csr_write_pos = int(nnz_offset)

    if diagnostics:
        _print_diag(
            _diag_record("after_global_offsets", rank, None, {"local_nnz": int(local_nnz), "global_nnz": nnz}),
            comm,
        )

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

        csr_group = _create_sparse_group(layers_group, csr_layer_name, "csr_matrix", (n_obs_out, n_vars))
        csr_data = csr_group.create_dataset("data", shape=(nnz,), dtype=data_dtype)
        csr_indices = csr_group.create_dataset("indices", shape=(nnz,), dtype=_INDEX_DTYPE)
        csr_indptr = csr_group.create_dataset("indptr", shape=(n_obs_out + 1,), dtype=_INT_DTYPE)

        if rank == 0:
            csr_indptr[0] = np.array(0, dtype=_INT_DTYPE)

        if diagnostics:
            _print_diag(_diag_record("after_dataset_create", rank), comm)

        for i, (block_start, block_end) in enumerate(iter_row_subblocks(row_start, row_end, sub_block_size)):
            X_csr = read_normalized_block(block_start, block_end).tocsr()
            X_csc = X_csr.tocsc()

            global_row_start = row_offset + np.int64(block_start - row_start)
            block_nnz = int(X_csr.nnz)

            _write_1d_blocks(csr_data, csr_write_pos, X_csr.data, dtype=data_dtype, block_size=write_block_size)
            _write_1d_blocks(
                csr_indices,
                csr_write_pos,
                X_csr.indices.astype(_INDEX_DTYPE, copy=False),
                dtype=_INDEX_DTYPE,
                block_size=write_block_size,
            )

            ptr_start = int(global_row_start) + 1
            global_indptr = X_csr.indptr[1:].astype(_INT_DTYPE, copy=False) + np.int64(csr_write_pos)
            _write_1d_blocks(csr_indptr, ptr_start, global_indptr, dtype=_INT_DTYPE, block_size=write_block_size)

            for col in range(n_vars):
                src_start = int(X_csc.indptr[col])
                src_end = int(X_csc.indptr[col + 1])
                if src_start == src_end:
                    continue

                dst_start = int(csc_write_pos[col])
                n = src_end - src_start

                _write_1d_blocks(x_data, dst_start, X_csc.data[src_start:src_end], dtype=data_dtype, block_size=write_block_size)
                global_rows = X_csc.indices[src_start:src_end].astype(_INDEX_DTYPE, copy=False) + global_row_start
                _write_1d_blocks(x_indices, dst_start, global_rows, dtype=_INDEX_DTYPE, block_size=write_block_size)

                csc_write_pos[col] += n

            csr_write_pos += block_nnz

            if diagnostics and i % diag_every == 0:
                _print_diag(
                    _diag_record("write_subblock", rank, X_csr, {"block_start": block_start, "block_end": block_end}),
                    comm,
                )

            del X_csc, X_csr

    kept_rows = np.arange(row_start, row_end, dtype=_INT_DTYPE)
    kept_rows_by_rank = comm.gather(kept_rows, root=0)

    if rank == 0:
        kept_input_rows = np.concatenate(kept_rows_by_rank) if kept_rows_by_rank else np.array([], dtype=_INT_DTYPE)
        _write_minimal_h5ad_metadata(in_path, out_path, kept_input_rows)

    comm.Barrier()


def get_csr_layer_shape(path: str, csr_layer_name: str = "csr") -> tuple[int, int]:
    with h5py.File(path, "r") as f:
        group = f["layers"][csr_layer_name]
        return tuple(int(x) for x in group.attrs["shape"])


def read_csr_layer_row_block(path: str, start: int, end: int, csr_layer_name: str = "csr") -> sparse.csr_matrix:
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


def read_local_csr_layer_block(path: str, comm: MPI.Comm = MPI.COMM_WORLD, csr_layer_name: str = "csr") -> tuple[sparse.csr_matrix, int, int]:
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_obs, _ = get_csr_layer_shape(path, csr_layer_name=csr_layer_name)
    start, end = row_bounds(n_obs, rank, size)
    return read_csr_layer_row_block(path, start, end, csr_layer_name), start, end

def write_h5ad_parallel_csc_from_column_blocks(
    in_path,
    out_path,
    X_block,
    col_range,
    n_obs,
    n_vars,
    comm,
    data_dtype=np.float32,
    write_block_size=_DEFAULT_WRITE_BLOCK,
):
    """Write one CSC column block per MPI rank into an AnnData-compatible h5ad.

    Each rank owns a contiguous column range of the final matrix and provides a
    CSC matrix with shape ``(n_obs, col_end - col_start)``. The output h5ad
    stores the imputed matrix in ``X`` as CSC.
    """
    if not h5py.get_config().mpi:
        raise RuntimeError("This function requires h5py compiled with MPI support.")

    rank = comm.Get_rank()
    col_start, col_end = (int(col_range[0]), int(col_range[1]))

    X_block = X_block.tocsc()
    expected_shape = (int(n_obs), col_end - col_start)
    if X_block.shape != expected_shape:
        raise ValueError(f"X_block has shape {X_block.shape}, expected {expected_shape}")

    local_counts_full = np.zeros(n_vars, dtype=_INT_DTYPE)
    local_counts_full[col_start:col_end] = np.diff(X_block.indptr).astype(
        _INT_DTYPE,
        copy=False,
    )

    all_counts = np.vstack(comm.allgather(local_counts_full)).astype(
        _INT_DTYPE,
        copy=False,
    )
    col_counts = all_counts.sum(axis=0, dtype=_INT_DTYPE)

    csc_indptr = np.empty(n_vars + 1, dtype=_INT_DTYPE)
    csc_indptr[0] = 0
    csc_indptr[1:] = np.cumsum(col_counts, dtype=_INT_DTYPE)

    nnz = int(csc_indptr[-1])
    local_start = int(csc_indptr[col_start])
    local_end = int(csc_indptr[col_end])

    if local_end - local_start != X_block.nnz:
        raise ValueError(
            f"Rank {rank}: global offsets imply {local_end - local_start} nnz, "
            f"but X_block.nnz={X_block.nnz}"
        )

    with h5py.File(out_path, "w", driver="mpio", comm=comm) as f:
        f.attrs["encoding-type"] = "anndata"
        f.attrs["encoding-version"] = "0.1.0"

        x_group = _create_sparse_group(f, "X", "csc_matrix", (n_obs, n_vars))
        x_data = x_group.create_dataset("data", shape=(nnz,), dtype=data_dtype)
        x_indices = x_group.create_dataset("indices", shape=(nnz,), dtype=_INDEX_DTYPE)
        x_group.create_dataset("indptr", data=csc_indptr, dtype=_INT_DTYPE)

        _write_1d_blocks(
            x_data,
            local_start,
            X_block.data,
            dtype=data_dtype,
            block_size=write_block_size,
        )
        _write_1d_blocks(
            x_indices,
            local_start,
            X_block.indices.astype(_INDEX_DTYPE, copy=False),
            dtype=_INDEX_DTYPE,
            block_size=write_block_size,
        )

        layers_group = f.create_group("layers")
        layers_group.attrs["encoding-type"] = "dict"
        layers_group.attrs["encoding-version"] = "0.1.0"

    comm.Barrier()

    if rank == 0:
        kept_input_rows = np.arange(n_obs, dtype=_INT_DTYPE)
        _write_minimal_h5ad_metadata(in_path, out_path, kept_input_rows)

    comm.Barrier()

