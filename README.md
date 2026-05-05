# HALRA

HALRA (High-performance ALRA) is a Python implementation of the ALRA
algorithm for imputing missing values in single-cell RNA-seq data. It is
designed to operate efficiently on sparse matrices and scale to large
datasets by preserving sparsity throughout the pipeline wherever
possible.

HALRA performs: - Low-rank matrix reconstruction via randomized SVD -
Gene-wise thresholding of reconstructed values - Per-gene rescaling to
match observed statistics - Restoration of observed (nonzero) values

The goal is to denoise and impute dropout values while preserving
biological signal.

------------------------------------------------------------------------

## Expected Input

HALRA supports two input types:

### 1. AnnData (recommended)

-   `.X` should contain a dense NumPy array or a SciPy sparse matrix
    (CSR/CSC)
-   `.obs_names` and `.var_names` are used as cell and gene labels

### 2. Raw matrix + labels

-   `matrix`: NumPy ndarray or SciPy sparse matrix (cells x genes)
-   `cells`: list/array of cell names (length = n_rows)
-   `genes`: list/array of gene names (length = n_cols)

------------------------------------------------------------------------

## Installation

Recommended environment:

``` bash
conda create -n halra_env python=3.10
conda activate halra_env
```

Then install HALRA locally:

``` bash
pip install -e .
```

------------------------------------------------------------------------

## Usage Example (AnnData)

``` python
import anndata as ad
from halra import halra

# Load your AnnData object
adata = ad.read_h5ad("your_data.h5ad")

# Run HALRA
adata_imputed = halra(adata, normalize=True)

# Result:
# adata_imputed.X now contains imputed values
# All metadata (.obs, .var, etc.) is preserved (filtered if needed)
```

------------------------------------------------------------------------

## Dependency Notes

HALRA depends on:

-   numpy
-   scipy
-   scikit-learn (for randomized SVD)
-   anndata (\>=0.10)

Optional ecosystem: - scanpy (for typical workflows) - h5py (for large
file IO, especially distributed workflows)

------------------------------------------------------------------------

## References

<a id="1">[1]</a>
Linderman, G. C. et al. Zero-preserving imputation of single-cell RNA-seq data. Nat Commun 13, (2022).

------------------------------------------------------------------------

## Current Limitations

-   Reconstruction step is dense (SVD-based), which may limit
    scalability for extremely large datasets (>1M cells)
-   No GPU support
-   Distributed/HPC version is not yet integrated into the pip package

------------------------------------------------------------------------

## Notes

HALRA is under active development and is intended for research use. APIs
may change as distributed and large-scale support evolves.
