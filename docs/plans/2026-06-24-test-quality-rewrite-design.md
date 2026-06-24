# Test Quality Rewrite Design

## Objective

Strengthen the initial regression suite so it protects scientific behavior without depending on degenerate geometry, implementation-owned transformer attributes, misleading tolerances, or unexplained constants. Production code and public API behavior remain unchanged.

## Coordinates and CRS

Coordinate round trips use an explicit northern UTM zone so the passing baseline is independent of the known automatic-zone defect. Cases cover the equator, points close to both zone boundaries, ordinary scalar inputs, one-dimensional arrays, singleton arrays, and two-dimensional grids. Angular comparisons use `rtol=0` and an explicit absolute tolerance of `1e-8` degrees. Kilometer output is checked from the UTM definition at a zone's central meridian and equator, where easting is 500 km, rather than through `GeoTrans.proj2utm`.

The automatic-zone expected failure moves to 1.5 degrees east, 50 degrees north. Its reference point belongs to zone 31 while the current four-degree area overlaps zone 30, clearly exposing the first-result selection defect. The proposed -1.5-degree point is not used because it correctly belongs to zone 30 and does not reproduce the bug.

## Fault geometry

The canonical fixture uses strike 30 degrees, dip 45 degrees, length 6 km, and width 4 km. Assertions cover strike and dip edge lengths, projected and vertical dip components, vertex ordering, planarity, and depth bounds. Patch counts are derived from named dimensions and `ceil`, never written as unexplained constants. All seven reference positions reconstruct the same physical plane. Separate tests cover nonuniform dip discretization, initial surface breach correction, and extension to the surface while preserving the lower edge.

## Transformations

Inverse transformation is checked against independently known translated coordinates, not only as the inverse of forward output. A `(1,3)` single-point matrix verifies the documented `m x 3` contract. Flat `(3,)` input remains outside scope until the API explicitly adopts it.

