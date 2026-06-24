# AGENTS.md

Local guidance for agents working in `seislip/utils/`.

## Scope

This directory contains reusable numerical helpers and experimental utility scripts.

- `transformation.py`: homogeneous 4x4 coordinate transformations.
- `quadtree.py`: quadtree downsampling for 2-D InSAR-like images.
- `gmsh_dsm.py`: exploratory Gmsh/triangulation script, not a stable library API.

## Transformation Conventions

- `Transformation` uses left multiplication for fixed-coordinate-system transforms.
- Call order matters: rotations/translations are prepended with `M.dot(self.M)`.
- Call `inverse()` after building a transform and before using `inverse_trans()`.
- The public method name `forwars_trans()` is misspelled in existing code; preserve it unless doing a coordinated API cleanup.
- Inputs and outputs are expected as `m x 3` point arrays/lists.

## Quadtree Conventions

- `QTree` accepts 2-D coordinate meshes `x`, `y`, and a 2-D image.
- NaNs are converted to zero internally for subdivision; downstream results use nonzero fractions to filter invalid blocks.
- Consequently, exact zero values are also treated as missing during statistics/results. This is current behavior, not a general scientific assumption; changing it requires explicit regression coverage.
- `mindim`, `maxdim`, and `std_threshold` control splitting.
- `qtresults()` populates `qtscatter`, `qtrect`, `qtxy4GMT`, `qtz4GMT`, and `qtnumber`.
- Preserve rectangle coordinate ordering used for GMT-style output unless intentionally changing export format.

## Maintenance Notes

- Keep utilities import-light where possible; avoid import-time plotting or file generation.
- `gmsh_dsm.py` executes immediately on import: it seeds NumPy, creates synthetic data, writes `.pos`/`.msh`, invokes Gmsh, and plots. Treat it as a standalone prototype; never import it from package initializers, library paths, or automated tests.
- If changing numerical routines, add simple synthetic checks for shape, reversibility, and NaN/zero behavior.
- For `QTree`, require `x`, `y`, and `image` to have matching 2-D shapes in new code/tests, and verify both odd and even image dimensions.
