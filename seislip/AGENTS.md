# AGENTS.md

This file gives local guidance for agents working inside the `seislip/` Python package.
It complements the repository-level `AGENTS.md`; prefer the more specific instruction when files overlap.

## Scope

This package contains the importable SeiSlip library code:

- `seislip.py`: `GeoTrans`, the base geographic/UTM coordinate transformer.
- `data/`: InSAR readers, preprocessing, plotting, and downsampling entry points.
- `fault/`: fault geometry, rectangular patches, triangular patches, and multi-segment faults.
- `utils/`: reusable transformation, quadtree, and meshing utilities.

The implemented library is currently preprocessing and geometry focused. Do not assume inversion, Green's functions, slip solvers, or stress calculations exist because they appear in the project name or `Todo.md`.

## Package Conventions

- Python requirement is `>=3.11`.
- Keep geographic coordinates as `lon`/`lat` in degrees.
- Keep UTM/cartesian coordinates as `x`/`y`/`z` in kilometers unless a local function explicitly documents another unit.
- Vertical depth follows the existing sign convention: depth below the surface is negative, e.g. `-5` km.
- Strike, dip, azimuth, and incidence are usually stored in degrees at public boundaries; convert to radians only at calculation points.
- Avoid changing public import paths in `__init__.py` casually. Add exports only when the API is stable enough for users.
- The package root currently exports only `InSAR`; fault classes are exported from `seislip.fault`.

## Implementation Notes

- `GeoTrans` is the common base for `InSAR`, `Fault`, and `MultiFault`; changes to zone handling or `ll2xy`/`xy2ll` affect the whole package.
- `ll2xy` and `xy2ll` intentionally convert pyproj meter outputs/inputs to kilometers.
- Keep calculations vectorized where it improves clarity, but validate input shape and finite values before numerical work.
- Make physical assumptions, units, array shapes, and numerical tolerances explicit.
- Keep scientific/numerical logic separate from file I/O and plotting.
- Keep relative imports working when modules are imported as a package. Existing `if __name__ == "__main__"` blocks are mainly manual examples and may use absolute research paths.
- Prefer NumPy array operations and explicit shape handling for image or mesh data.
- Do not treat scripts under `__main__` blocks as tests; they are exploratory examples unless moved into a real test framework.

## Before Editing

- Read the local `AGENTS.md` in subdirectories before changing nested code.
- Preserve user research data paths and experimental snippets unless the task explicitly asks to clean them up.
- If adding behavior, add a small focused test or reproducible example when possible.
- Do not introduce import-time side effects through package `__init__.py` files.
- Do not add machine-specific paths to reusable modules or introduce dependencies without checking whether an established scientific library already solves the problem.
