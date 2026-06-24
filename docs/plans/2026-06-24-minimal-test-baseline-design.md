# Minimal Test Baseline Design

## Objective

Create a small pytest suite that protects SeiSlip's current core scientific behavior before bug fixes and structural refactoring. The suite covers coordinate transformations, homogeneous transformations, and planar fault geometry. It uses deterministic synthetic inputs and does not require external research data, GUI interaction, NetCDF, or Gmsh.

## Scope

The baseline covers three components:

1. `GeoTrans`: scalar and NumPy-array longitude/latitude round trips, with explicit verification that projected coordinates are expressed in kilometers.
2. `Transformation`: forward/inverse round trips after translation and rotation composition.
3. `Fault`: initialization from representative reference points, planar surface vertex shape and depth convention, and uniform rectangular/triangular patch counts and vertex shapes.

Tests will use numerical tolerances through `numpy.testing`. Expected patch counts will be derived from simple exactly divisible dimensions so assertions remain readable and independently checkable.

## Exclusions

- Southern-hemisphere explicit UTM handling is a known defect and will be addressed with a failing regression test during the bug-fix phase, not encoded as current passing behavior.
- Quadtree zero/NaN behavior is scientifically ambiguous and excluded until its intended missing-data model is decided.
- InSAR file readers require separate synthetic IO fixtures and optional dependencies; they are outside this minimum baseline.
- No production behavior will be changed in this task.

## Test and dependency configuration

Add a minimal `pyproject.toml` declaring the Python requirement, current runtime dependencies needed by the public import path, a `test` optional-dependency group containing pytest, and pytest discovery settings. Tests will be split into focused files under `tests/`; the exploratory `tests/test.py` will remain untouched for now but will be excluded from pytest discovery so it cannot execute during collection.

Success means the new focused tests collect and pass in the active environment after installing the test dependency, with no modifications to implementation modules.

