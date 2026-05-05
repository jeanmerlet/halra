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
adata = ad.read_h5ad("your_data.h5ad")

# Run HALRA
adata_imputed = halra(adata, normalize=True)

# Result:
# adata_imputed.X now contains imputed values
# All metadata (.obs, .var, etc.) is preserved (filtered if needed)
```


## Dependency Notes

HALRA depends on:

-   numpy
-   scipy
-   scikit-learn (for randomized SVD)
-   anndata (\>=0.10)


## Current Limitations

-   Reconstruction step is dense (SVD-based), which may limit
    scalability for extremely large datasets (>1M cells)
-   No GPU support
-   Distributed/HPC version is not yet integrated into the pip package


## References

[1] Linderman, G. C. et al. Zero-preserving imputation of single-cell RNA-seq data. Nat Commun 13, (2022).


## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Notes

HALRA is under active development and is intended for research use. APIs
may change as distributed and large-scale support evolves.
