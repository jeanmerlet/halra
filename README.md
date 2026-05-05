# HALRA

HALRA (High-performance ALRA) is a Python implementation of the ALRA
algorithm for imputing missing values in single-cell RNA-seq data 
[[1](https://github.com/jeanmerlet/halra#references)]. It is designed 
to operate efficiently on sparse matrices and scale to large
datasets by preserving sparsity throughout the pipeline wherever
possible.

HALRA performs:
- Low-rank matrix reconstruction via randomized SVD
- Gene-wise thresholding of reconstructed values
- Per-gene rescaling to match observed statistics
- Restoration of observed (nonzero) values

The goal is to denoise and impute dropout values while preserving
biological signal.

## Expected Input

HALRA supports two input types:

### 1. AnnData

-   `.X` should contain a dense NumPy array or a SciPy sparse matrix
    (CSR/CSC)
-   `.obs_names` and `.var_names` are used as cell and gene labels

### 2. Raw matrix + labels

-   `matrix`: NumPy ndarray or SciPy sparse matrix (cell x gene)
-   `cells`: list/array of cell names (length = n_rows)
-   `genes`: list/array of gene names (length = n_cols)


## Installation

HALRA can be installed as a local pip package. First install Python v3.10. 
Example:

``` bash
conda create -n halra_env python=3.10
conda activate halra_env
```

Then, git clone this repo and install HALRA locally:

``` bash
pip install -e .
```


## Usage Example (AnnData)

``` python
import anndata as ad
from halra import halra

# Load your AnnData object
adata = ad.read_h5ad("anndata.h5ad")

# Run HALRA
adata_imputed = halra(adata, normalize=True)

# Result:
# adata_imputed.X now contains imputed values
# All metadata (.obs, .var, etc.) is preserved (filtered if needed)
```


## Usage Example (10x Matrix Market)

```python
import os
import pandas as pd
from scipy.io import mmread
from halra import halra

# Load 10x files
mtx_dir = "/path/to/dir"
matrix = mmread(os.path.join(mtx_dir, "matrix.mtx")).T
features = pd.read_csv(os.path.join(mtx_dir, "features.tsv"), sep="\t", header=None, usecols=[0])
barcodes = pd.read_csv(os.path.join(mtx_dir, "barcodes.tsv"), sep="\t", header=None)

# Run HALRA
imputed_matrix, cells, genes = halra(matrix, barcodes, features, normalize=True)

# Result:
# imputed_matrix contains imputed values
# cells and genes contain the filtered cell/gene labels


## Dependency Notes

HALRA depends on:

-   numpy
-   scipy
-   scikit-learn (for randomized SVD)
-   anndata (\>=0.10)


## Current Limitations and Experimental Features

-   Reconstruction step is dense (SVD-based), which may limit
    scalability for extremely large datasets (>1M cells)
-   Distributed and HPC-oriented implementations of HALRA are under active development
and can be found in the `experimental/` directory. These are not yet part of the
stable package API.


## References

[1] Linderman, G. C. et al. Zero-preserving imputation of single-cell RNA-seq data. Nat Commun 13, (2022).


## License

This project is licensed under the MIT License - see the LICENSE file for details.
