# AGENTS.md

Local guidance for agents working in `seislip/fault/`.

## Scope

This directory models fault geometry and discretization.

- `fault.py`: planar `Fault` geometry and rectangular/triangular patch construction.
- `rectpatch.py`: rectangular patch generation on a planar fault.
- `tripatch.py`: triangular patch generation on a planar fault.
- `multifault.py`: multi-segment or curved fault container.

## Geometry Conventions

- UTM/cartesian coordinates are `(x, y, z)` in kilometers.
- UTM `z` is zenith-positive; depths below surface are negative.
- Fault coordinate system is right-handed:
  - `X`: along strike.
  - `Y`: opposite the dip direction, down along the plane.
  - `Z`: normal to the fault plane.
- Fault origin for transformations is usually the upper center point: `{"upper center": (x, y, z)}`.
- Strike and dip are public API degrees; convert with `np.radians()` inside calculations.

## Reference Point Names

Accepted `Fault.initialize_fault()` positions include:

- `uo`: upper origin.
- `uc`: upper center.
- `ue`: upper end.
- `bo`: bottom origin.
- `bc`: bottom center.
- `be`: bottom end.
- `cc`: centroid center.

Long names and uppercase variants are supported in existing code. Preserve compatibility when refactoring.

## Patch Data

- Rectangular patches are stored as lists of four UTM vertices:
  `[(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]`.
- Triangular patches are stored as lists of three UTM vertices.
- `Fault.patch_verts` is used for the original fault plane and then overwritten by rectangular patches after `construct_rect_patches()`.
- `Fault.tri_patch_verts` is populated by `construct_tri_patches()`.

## Maintenance Notes

- Keep `RectPatch` and `TriPatch` transformation setup aligned unless intentionally diverging.
- `MultiFault.discretize_triangles()` currently calls `TriPatch.discretize_curved()`, but `tripatch.py` currently implements only `discretize_planar()`. Treat curved triangular meshing as incomplete until implemented and tested.
- Surface breach logic adjusts upper-edge depth and width; test geometry before changing `extend_to_surface()` or `__check_breach_surface()`.
- Avoid changing vertex ordering without checking plotting, meshing, and downstream inversion assumptions.
