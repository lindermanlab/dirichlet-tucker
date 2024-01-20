# dirichlet-tucker
Tucker decompositions with normalization constraints

## Repository organization
This repository is organized to separate source code and model analyses from
analysexperimental or data-dependent analyses. The current branches are:
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