import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
from read_write_adata import *
from mpi4py import MPI
import os


def master_print(msg):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(msg, flush=True)


def ensure_csr(mtx):
    if isinstance(mtx, np.ndarray):
        return sp.csr_matrix(mtx)
    elif sp.issparse(mtx):
        return mtx.tocsr()
    else:
        raise TypeError("Input must be numpy.ndarray or scipy.sparse matrix")


def remove_zero_genes_cells(mtx, cell_names, gene_names):
    col_sums = np.asarray(mtx.sum(axis=0)).ravel()
    keep_cols = col_sums > 0
    n_omit = np.sum(~keep_cols)
    if n_omit > 0:
        master_print(f"Omitted {n_omit} genes with zero counts")
    mtx = mtx[:, keep_cols]
    gene_names = gene_names[keep_cols]

    row_sums = np.asarray(mtx.sum(axis=1)).ravel()
    keep_rows = row_sums > 0
    n_omit = np.sum(~keep_rows)
    if n_omit > 0:
        master_print(f"Omitted {n_omit} cells with zero counts")
    mtx = mtx[keep_rows, :]
    cell_names = cell_names[keep_rows]

    return mtx, row_sums[keep_rows], cell_names, gene_names


def normalize_total(mtx, row_sums, scale_factor=1e4):
    inv_row_sums = 1.0 / row_sums
    return mtx.multiply(inv_row_sums[:, None]) * scale_factor


def log1p_sparse(mtx):
    mtx.data = np.log1p(mtx.data)
    return mtx


def log_normalize_counts(mtx, cell_names, gene_names, scale_factor=1e4):
    mtx = ensure_csr(mtx)
    mtx, row_sums, cell_names, gene_names = remove_zero_genes_cells(mtx, cell_names, gene_names)
    mtx = normalize_total(mtx, row_sums, scale_factor)
    mtx = log1p_sparse(mtx)
    return mtx, cell_names, gene_names


def choose_mtx_rank(mtx, n_iter, seed, n_comps=100, thresh=6, noise_start=80):
    if n_comps >= min(mtx.shape):
        raise ValueError("n_comps must be smaller than the smallest dimension of mtx")
    if noise_start > n_comps - 5:
        raise ValueError("There need to be at least 5 singular values considered noise")

    noise_svals = np.arange(noise_start, n_comps)
    _, s, _ = randomized_svd(mtx, n_components=n_comps, n_iter=n_iter, random_state=seed)

    diffs = s[:-1] - s[1:]
    mu = np.mean(diffs[noise_svals - 1])
    sigma = np.std(diffs[noise_svals - 1])
    n_std_devs = (diffs - mu) / sigma
    rank = np.max(np.where(n_std_devs > thresh)[0]) + 1
    master_print(f"Chose rank={rank}")
    return rank


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


def compute_low_rank_reconstruction(mtx, rank, n_iter, seed):
    master_print("Computing low-rank reconstruction")
    u, s, vh = randomized_svd(mtx, n_components=rank, n_iter=n_iter, random_state=seed)
    recon_mtx = u @ np.diag(s) @ vh
    return recon_mtx


def compute_svd_factors(mtx, rank, n_iter, seed):
    master_print("Computing low-rank factors")
    u, s, vh = randomized_svd(mtx, n_components=rank, n_iter=n_iter, random_state=seed)
    us = u * s
    return us, vh


def reconstruct_col_block(us, vh, start, stop):
    return us @ vh[:, start:stop]


def compute_block_thresholds(us, vh, quantile_prob, block_size):
    n_col = vh.shape[1]
    thresholds = np.empty(n_col, dtype=float)

    for start in range(0, n_col, block_size):
        stop = min(start + block_size, n_col)
        recon_block = reconstruct_col_block(us, vh, start, stop)
        thresholds[start:stop] = np.abs(np.quantile(recon_block, quantile_prob, axis=0))

    return thresholds


def threshold_block_sparse(recon_block, thresholds):
    mask = recon_block > thresholds[np.newaxis, :]
    thresh_block = sp.csc_matrix(recon_block * mask)
    thresh_block.eliminate_zeros()
    return thresh_block


def threshold_full_sparse(recon_mtx, quantile_prob):
    thresholds = np.abs(np.quantile(recon_mtx, quantile_prob, axis=0))
    return threshold_block_sparse(recon_mtx, thresholds)


def scale_thresholded_matrix_sparse(thresh_block, to_scale, sigma_1_2, to_add):
    master_print(f"Scaling all except for {np.sum(~to_scale)} columns")
    imputed_block = sp.csc_matrix(thresh_block)

    for j in np.where(to_scale)[0]:
        start, end = imputed_block.indptr[j], imputed_block.indptr[j + 1]
        if start == end:
            continue
        vals = imputed_block.data[start:end]
        vals = vals * sigma_1_2[j] + to_add[j]
        imputed_block.data[start:end] = vals

    return imputed_block


def compute_scaling_factors_sparse(thresh_block, mtx_block):
    master_print("Computing scaling factors")
    thresh_block = thresh_block.tocsc()
    mtx_block = mtx_block.tocsc()

    n_col = thresh_block.shape[1]

    sigma_1 = np.zeros(n_col, dtype=float)
    mu_1 = np.zeros(n_col, dtype=float)

    for j in range(n_col):
        start, end = thresh_block.indptr[j], thresh_block.indptr[j + 1]
        vals = thresh_block.data[start:end]
        if vals.size > 0:
            sigma_1[j] = np.std(vals)
            mu_1[j] = np.mean(vals)

    sigma_2 = sparse_col_std_nnz(mtx_block)
    mu_2 = sparse_col_mean_nnz(mtx_block)

    to_scale = (
        (~np.isnan(sigma_1))
        & (~np.isnan(sigma_2))
        & ~((sigma_1 == 0) & (sigma_2 == 0))
        & (sigma_1 != 0)
    )

    sigma_1_2 = np.divide(
        sigma_2,
        sigma_1,
        out=np.zeros_like(sigma_2, dtype=float),
        where=sigma_1 != 0
    )
    to_add = -mu_1 * sigma_1_2 + mu_2

    return to_scale, sigma_1_2, to_add


def zero_negatives_sparse(imputed_block):
    neg = imputed_block.data < 0
    pct_neg = round(100 * np.sum(neg) / (imputed_block.shape[0] * imputed_block.shape[1]), 2)
    imputed_block.data[neg] = 0
    imputed_block.eliminate_zeros()
    master_print(f"{pct_neg}% of the values became negative in the scaling process and were set to zero.")
    return imputed_block


def restore_observed_values_sparse(imputed_block, mtx_block, pres_obs="zeroed"):
    mtx_coo = mtx_block.tocoo()
    imputed_lil = imputed_block.tolil()

    if pres_obs == "zeroed":
        for i, j, v in zip(mtx_coo.row, mtx_coo.col, mtx_coo.data):
            if imputed_lil[i, j] == 0:
                imputed_lil[i, j] = v
    elif pres_obs == "all":
        for i, j, v in zip(mtx_coo.row, mtx_coo.col, mtx_coo.data):
            imputed_lil[i, j] = v
    elif pres_obs == "none":
        pass
    else:
        raise ValueError("pres_obs must be 'zeroed', 'all', or 'none'")

    return imputed_lil.tocsc()


def report_density(mtx, imputed_mtx):
    start_nnz = round(mtx.nnz / (mtx.shape[0] * mtx.shape[1]), 2)
    if sp.issparse(imputed_mtx):
        end_nnz = round(100 * imputed_mtx.nnz / (imputed_mtx.shape[0] * imputed_mtx.shape[1]), 2)
    else:
        end_nnz = round(100 * np.count_nonzero(imputed_mtx) / imputed_mtx.size, 2)
    print(f"Original nonzero values: {start_nnz}%")
    print(f"Imputed nonzero values: {end_nnz}%")


def process_gene_block(mtx, us, vh, start, stop, quantile_prob, pres_obs):
    print(f"Processing genes {start} to {stop}", flush=True)

    mtx_block = mtx[:, start:stop]
    recon_block = reconstruct_col_block(us, vh, start, stop)
    thresholds = np.abs(np.quantile(recon_block, quantile_prob, axis=0))
    thresh_block = threshold_block_sparse(recon_block, thresholds)
    del recon_block

    to_scale, sigma_1_2, to_add = compute_scaling_factors_sparse(thresh_block, mtx_block)
    imputed_block = scale_thresholded_matrix_sparse(thresh_block, to_scale, sigma_1_2, to_add)
    imputed_block = zero_negatives_sparse(imputed_block)
    imputed_block = restore_observed_values_sparse(imputed_block, mtx_block, pres_obs=pres_obs)
    imputed_block.eliminate_zeros()

    return imputed_block


def halra_blockwise(mtx, cell_names, gene_names, rank, n_iter, quantile_prob,
                    seed, pres_obs, block_size, out_path, comm=MPI.COMM_WORLD):
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    n_obs, n_var = mtx.shape
    block_starts = list(range(0, n_var, block_size))
    n_blocks = len(block_starts)

    # one MPI rank per block
    if mpi_size != n_blocks:
        if mpi_rank == 0:
            raise ValueError(
                f"One MPI rank per block required, but got "
                f"{mpi_size} ranks for {n_blocks} blocks. "
                f"Set block_size so n_blocks == number of MPI ranks."
            )
        return None

    us, vh = compute_svd_factors(mtx, rank, n_iter, seed)

    start = block_starts[mpi_rank]
    stop = min(start + block_size, n_var)

    imputed_block = process_gene_block(
        mtx=mtx,
        us=us,
        vh=vh,
        start=start,
        stop=stop,
        quantile_prob=quantile_prob,
        pres_obs=pres_obs,
    )

    # each rank writes its own temporary block file
    tmp_block_path = f"{out_path}.rank{mpi_rank}.npz"
    sp.save_npz(tmp_block_path, imputed_block)

    block_nnz = imputed_block.nnz
    del imputed_block

    # collect nnz counts for reporting
    nnz_counts = comm.gather(block_nnz, root=0)

    # wait until all block files exist before rank 0 starts reading/appending
    comm.Barrier()

    if mpi_rank == 0:
        h5f = init_h5ad_csc(
            out_path,
            n_obs,
            n_var,
            obs_names=cell_names,
            var_names=gene_names,
        )

        try:
            start_nnz = round(100 * mtx.nnz / (mtx.shape[0] * mtx.shape[1]), 2)
            total_nnz = 0

            for src_rank in range(mpi_size):
                tmp_path = f"{out_path}.rank{src_rank}.npz"
                block = sp.load_npz(tmp_path).tocsc()
                append_csc_block_to_h5ad(h5f, block)
                total_nnz += block.nnz
                del block

            end_nnz = round(100 * total_nnz / (mtx.shape[0] * mtx.shape[1]), 2)
            print(f"Original nonzero values: {start_nnz}%", flush=True)
            print(f"Imputed nonzero values: {end_nnz}%", flush=True)

        finally:
            h5f.close()

        # cleanup of temp block files
        for src_rank in range(mpi_size):
            tmp_path = f"{out_path}.rank{src_rank}.npz"
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return out_path

    return None


def halra_full(mtx, rank, n_iter, quantile_prob, seed, pres_obs):
    recon_mtx = compute_low_rank_reconstruction(mtx, rank, n_iter, seed)
    thresh_mtx = threshold_full_sparse(recon_mtx, quantile_prob)
    del recon_mtx

    to_scale, sigma_1_2, to_add = compute_scaling_factors_sparse(thresh_mtx, mtx)
    imputed_mtx = scale_thresholded_matrix_sparse(thresh_mtx, to_scale, sigma_1_2, to_add)
    imputed_mtx = zero_negatives_sparse(imputed_mtx)
    imputed_mtx = restore_observed_values_sparse(imputed_mtx, mtx, pres_obs=pres_obs)
    imputed_mtx.eliminate_zeros()

    report_density(mtx, imputed_mtx)
    return imputed_mtx


def halra(mtx, cell_names=None, gene_names=None, n_iter='auto', quantile_prob=0.001, mtx_rank=None,
          seed=1, normalize=False, pres_obs="zeroed", block_size=None, out_path=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    mtx, cell_names, gene_names = parse_input_matrix(mtx, cell_names, gene_names)
    if block_size is not None and out_path is None:
        raise ValueError("out_path required for blockwise imputation")
    if normalize:
        mtx, cell_names, gene_names = log_normalize_counts(mtx, cell_names, gene_names)
    else:
        mtx = ensure_csr(mtx)
    mtx = mtx.tocsc()
    n_row, n_col = mtx.shape
    master_print(f"Read matrix with {n_row} cells and {n_col} genes")

    if mtx_rank is None:
        mtx_rank = choose_mtx_rank(mtx, n_iter, seed)
    if block_size is None:
        return halra_full(mtx, mtx_rank, n_iter, quantile_prob, seed, pres_obs)
    else:
        return halra_blockwise(
            mtx, cell_names, gene_names, mtx_rank, n_iter,
            quantile_prob, seed, pres_obs, block_size, out_path
        )

