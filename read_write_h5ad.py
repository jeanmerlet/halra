import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from anndata.io import write_elem
import h5py
import os


def parse_input_matrix(mtx, cell_names, gene_names):
    if isinstance(mtx, ad.AnnData):
        x = mtx.X
        cell_names = np.asarray(mtx.obs_names, dtype=object)
        gene_names = np.asarray(mtx.var_names, dtype=object)
    elif isinstance(mtx, np.ndarray) or sp.issparse(mtx):
        x = mtx
        if cell_names is None or gene_names is None:
            raise ValueError(
                "For numpy.ndarray or scipy.sparse input, both cell_names and gene_names must be provided."
            )
        cell_names = np.asarray(cell_names, dtype=object)
        gene_names = np.asarray(gene_names, dtype=object)
    else:
        raise TypeError("mtx must be AnnData, numpy.ndarray, or scipy.sparse matrix")

    if x.shape[0] != len(cell_names):
        raise ValueError("Length of cell_names must match number of rows in mtx")
    if x.shape[1] != len(gene_names):
        raise ValueError("Length of gene_names must match number of columns in mtx")

    return x, cell_names, gene_names


def init_h5ad_csc(out_path, n_obs, n_var, obs_names, var_names):
    obs = pd.DataFrame(index=pd.Index(np.asarray(obs_names, dtype=str), name="obs_names"))
    var = pd.DataFrame(index=pd.Index(np.asarray(var_names, dtype=str), name="var_names"))

    f = h5py.File(out_path, "w")

    # AnnData root metadata
    f.attrs["encoding-type"] = "anndata"
    f.attrs["encoding-version"] = "0.1.0"

    # Required AnnData elements: obs and var must exist as dataframes
    obs = pd.DataFrame(index=pd.Index(obs_names, name="obs_names"))
    var = pd.DataFrame(index=pd.Index(var_names, name="var_names"))
    write_elem(f, "obs", obs)
    write_elem(f, "var", var)

    # Empty required/standard mapping groups
    for key in ["layers", "obsm", "obsp", "varm", "varp", "uns"]:
        grp = f.create_group(key)
        grp.attrs["encoding-type"] = "dict"
        grp.attrs["encoding-version"] = "0.1.0"

    # X as on-disk CSC sparse matrix
    x = f.create_group("X")
    x.attrs["encoding-type"] = "csc_matrix"
    x.attrs["encoding-version"] = "0.1.0"
    x.attrs["shape"] = np.array([n_obs, n_var], dtype=np.int64)

    x.create_dataset(
        "data",
        shape=(0,),
        maxshape=(None,),
        dtype=np.float32,
        chunks=True,
    )
    x.create_dataset(
        "indices",
        shape=(0,),
        maxshape=(None,),
        dtype=np.int64,
        chunks=True,
    )
    # indptr length for CSC is n_var + 1
    indptr = x.create_dataset(
        "indptr",
        shape=(1,),
        maxshape=(n_var + 1,),
        dtype=np.int64,
        chunks=True,
    )
    indptr[0] = 0

    return f


def append_csc_block_to_h5ad(h5_file, block_csc):
    if not sp.isspmatrix_csc(block_csc):
        block_csc = block_csc.tocsc()

    x = h5_file["X"]
    data_ds = x["data"]
    indices_ds = x["indices"]
    indptr_ds = x["indptr"]

    nnz_old = int(data_ds.shape[0])
    nnz_add = int(block_csc.nnz)
    n_ptr_old = int(indptr_ds.shape[0])
    n_ptr_add = int(block_csc.shape[1])

    # Cast explicitly to match on-disk dtypes and avoid overflow in arithmetic
    data_to_write = block_csc.data.astype(data_ds.dtype, copy=False)
    indices_to_write = block_csc.indices.astype(indices_ds.dtype, copy=False)

    # Promote indptr to int64 before adding nnz_old
    indptr_to_write = block_csc.indptr[1:].astype(np.int64, copy=False) + np.int64(nnz_old)

    # Append data
    data_ds.resize((nnz_old + nnz_add,))
    data_ds[nnz_old:] = data_to_write

    # Append row indices
    indices_ds.resize((nnz_old + nnz_add,))
    indices_ds[nnz_old:] = indices_to_write

    # Append indptr, skipping the leading 0 and offsetting by prior nnz
    indptr_ds.resize((n_ptr_old + n_ptr_add,))
    indptr_ds[n_ptr_old:] = indptr_to_write
