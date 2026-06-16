# CLAUDE.md

Local guidance for Claude Code working in `tests/`.

## Scope

This directory currently contains exploratory scratch code rather than a formal test suite.

- `test.py` is mostly commented experiments for coordinate transforms, quadtree logic, plotting, and fault geometry.
- The active code at the bottom tests a line-plane intersection helper and prints a result.

## Testing Guidance

- Do not assume `tests/test.py` is a pytest-compatible suite.
- When adding real tests, prefer focused `pytest` files named `test_*.py`.
- Keep tests independent of local absolute research paths.
- Use small synthetic arrays/geometries to check coordinate transforms, image orientation, and mesh shapes.
- Avoid requiring GUI display in automated tests; use non-interactive Matplotlib backends or assert data structures directly.

## Useful Future Test Targets

- `GeoTrans.ll2xy()` and `xy2ll()` round-trip in kilometers.
- `Transformation` forward/inverse round-trip.
- `QTree` output shape and NaN/zero filtering behavior.
- `Fault.initialize_fault()` for all reference point aliases.
- `RectPatch` and `TriPatch` patch counts and vertex dimensions.
