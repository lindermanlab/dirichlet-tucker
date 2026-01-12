# dirichlet-tucker
This package implements a type of tensor decomposition method called a Tucker decomposition. To make the results more interpretable, we add non-negativity and normalization constraints so that the model can be thought of approximating the original data tensor with a superposition of low-rank components, analogous to the "topics" in Latent Dirichlet Analysis (Blei et al., 2001). 

## Technical Description
A technical description of the model and algorithm can be found in [`notes/method.pdf`](notes/method.pdf).

## Repository organization
This repository is organized to separate source code and model analyses from
experimental or data-dependent analyses. The current branches are:
- `main`: (currently also contains serotonin / 4d analysis)
- `killifish`: Killifish scripts and analyses
- `poisson-tucker`: Model comparisons against Poisson Tucker variations
    - Currently kept in a separate branch due to active development; will be merged into the `main` upon completion

The source code (`./dtd/`) is manually kept in-sync with the main branch.
Selective merging into a target branch (whether updating the main branch or
porting new changes into a working branch) is executed with the following:
```
git checkout BRANCH_TO_MERGE_INTO
git checkout --patch BRANCH_TO_MERGE_FROM FOLDER_OR_PATH_TO_MERGE
```

## Author Information
The method, code, and analyses were developed by Libby Zhang and Scott Linderman.
