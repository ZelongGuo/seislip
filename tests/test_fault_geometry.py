"""
Tests for planar fault geometry and discretization (Fault class).

These tests verify the core geometric calculations that underpin every
forward-modeling and inversion workflow.  If any of these tests fail,
the fault surfaces fed to Okada/EDGRN dislocation codes will be wrong.

Design principles
-----------------
1. **Use a dipping, non-north-striking fault as the default.**
   Early versions of this test suite used a vertical (dip=90°) fault
   striking north (strike=0°).  At those special angles most trigonometric
   functions degenerate to 0, 1, or ∞, masking bugs in rotation logic.
   The default fault here has dip=45° and strike=30° — every matrix
   element is non-trivial.

2. **Verify invariants, not just specific numbers.**
   Where possible, we assert geometric *properties* that must hold
   regardless of parameter choice (e.g. all patch vertices lie on the
   same plane, no triangle has zero area).  These are more robust than
   magic-number assertions.

3. **Compute expected values independently.**
   Tests never call the same code path to generate both "actual" and
   "expected" values.  When we need a reference answer (e.g. the width
   sequence from ``_get_segments_size_varying``), we re-implement the
   logic in a simple, obviously-correct helper.

Fault coordinate conventions (from the source)
----------------------------------------------
* **UTM** : X = easting (km), Y = northing (km), Z = zenith (km).
  Depth below surface is **negative** Z.

* **Fault-local** (right-handed):
  X = along strike,  Y = opposite dip direction,  Z = normal to plane.

* **Corner vertex order** (stored in ``patch_verts[0]``):
  [0] = upper origin  (uo)
  [1] = upper end     (ue)
  [2] = bottom end    (be)
  [3] = bottom origin (bo)

Coverage
--------
* Oblique (dip=45°, strike=30°) surface geometry — strike-edge and
  dip-edge vectors, vertex depths, upper-centre position.
* All seven reference-point aliases (uo, uc, ue, bo, bc, be, cc) —
  each must produce an identical fault.
* Uniform rectangular discretization — patch count, edge lengths,
  coplanarity, and depth bound.
* Uniform triangular discretization — patch count, positive area,
  coplanarity, and depth bound.
* Depth-varying rectangular discretization — geometric width sequence,
  updated fault width, deepest vertex depth.
* Surface breaching — automatic trimming when the upper edge is above
  the surface.
* ``extend_to_surface`` — lower-edge preservation after extension.
"""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from seislip.crs import CoordinateTransformer
from seislip.fault import Fault


# ---------------------------------------------------------------------------
# Module-level constants — the "standard oblique fault"
# ---------------------------------------------------------------------------

# Reference point for UTM zone selection (same as the fault location).
FAULT_LON = 44.0
FAULT_LAT = 35.0

# Depth of the upper fault edge, in km (negative = below surface).
# Chosen as a moderate burial depth — the fault does NOT breach the
# surface by default, so breach-trimming tests can start from this
# baseline and then modify depth.
UPPER_DEPTH_KM = -2.0

# Strike: azimuth of the fault trace measured clockwise from north.
# 30° is a typical oblique-strike value in the Iran collision zone.
STRIKE_DEGREES = 30.0

# Dip: inclination of the fault plane from horizontal.
# 45° is the most general case — neither shallow nor vertical,
# ensuring that both horizontal and vertical projections are
# significantly non-zero.
DIP_DEGREES = 45.0

# Fault dimensions (km).  These are chosen so that discretization
# tests produce small integer patch counts for easy verification:
#   6 × 4 km  with  2.5 km sub-length  →  ceil(6/2.5)=3  strike segments
#                     1.5 km sub-width   →  ceil(4/1.5)=3  dip segments
#   → 9 rectangular patches, 18 triangular patches.
FAULT_LENGTH_KM = 6.0
FAULT_WIDTH_KM = 4.0

# Absolute tolerance for geometric comparisons, in km.
# 1e-8 km = 0.01 mm — essentially machine precision for kilometre-scale
# coordinates stored in float64.  This is tight enough to catch any
# matrix-construction or trigonometric error.
GEOMETRY_ATOL_KM = 1e-8


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def make_oblique_fault(
    point_position: str = "uc",
    lon: float = FAULT_LON,
    lat: float = FAULT_LAT,
    depth: float = UPPER_DEPTH_KM,
) -> Fault:
    """Create a Fault initialised with the standard oblique parameters.

    Parameters
    ----------
    point_position : str
        Which reference point is given by ``lon``, ``lat``, ``depth``.
        One of ``"uo"``, ``"uc"``, ``"ue"``, ``"bo"``, ``"bc"``, ``"be"``,
        ``"cc"`` (and their aliases).  Default is upper centre.
    lon, lat : float
        Geographic coordinates of the reference point (degrees).
    depth : float
        Vertical depth of the reference point (km, negative = below surface).

    Returns
    -------
    Fault
        Fully initialised fault with ``patch_verts`` populated.
    """
    transformer = CoordinateTransformer("fault-transformer", lon0=FAULT_LON, lat0=FAULT_LAT)
    fault = Fault("oblique-fault", transformer=transformer)
    fault.initialize_fault(
        pointpos=point_position,
        lon=lon,
        lat=lat,
        verdepth=depth,
        strike=STRIKE_DEGREES,
        dip=DIP_DEGREES,
        length=FAULT_LENGTH_KM,
        width=FAULT_WIDTH_KM,
    )
    return fault


# ---------------------------------------------------------------------------
# Geometry helper — extract the 4 corner vertices as a (4, 3) array
# ---------------------------------------------------------------------------

def surface_vertices(fault: Fault) -> np.ndarray:
    """Return the faultʼs four corner vertices as a (4, 3) float64 array.

    ``fault.patch_verts`` is a list of fault surfaces.  Immediately after
    ``initialize_fault`` it contains exactly one entry — the four corners
    of the fault plane in UTM coordinates::

        [[(x_uo,y_uo,z_uo), (x_ue,y_ue,z_ue), (x_be,y_be,z_be), (x_bo,y_bo,z_bo)]]

    This helper extracts that single surface.
    """
    return np.asarray(fault.patch_verts, dtype=float)[0]


# ---------------------------------------------------------------------------
# Geometry helper — assert all vertices lie on the same plane
# ---------------------------------------------------------------------------

def assert_vertices_lie_on_plane(
    vertices: np.ndarray,
    reference_surface: np.ndarray,
) -> None:
    """Assert that every vertex in ``vertices`` is coplanar with
    the ``reference_surface``.

    How it works
    ------------
    1. Compute the fault planeʼs unit normal vector from the reference
       surface corners::

           normal = (strike_edge × dip_edge) / |strike_edge × dip_edge|

    2. For each vertex, compute its signed distance to the plane::

           distance = (vertex − origin) · normal

    3. Assert all distances are zero (within tolerance).

    This invariant should hold for **any** set of patches on a planar
    fault — no vertex should ever leave the fault plane.  A non-zero
    distance indicates a bug in ``fault2utm``, ``utm2fault``, or the
    discretization logic.
    """
    # Build the plane from three corners of the reference surface.
    origin = reference_surface[0]                    # upper origin
    strike_edge = reference_surface[1] - origin      # along strike
    dip_edge = reference_surface[3] - origin         # along dip (downward)

    # Unit normal vector (right-hand rule: strike × dip).
    normal = np.cross(strike_edge, dip_edge)
    normal /= np.linalg.norm(normal)

    # Signed distances of every vertex to the plane.
    # Reshape to (-1, 3) to handle both (4, 3) and (N×4, 3) arrays.
    distances = (np.asarray(vertices).reshape(-1, 3) - origin) @ normal

    # All distances must be zero — every vertex on the plane.
    assert_allclose(distances, 0.0, rtol=0.0, atol=GEOMETRY_ATOL_KM)


# ===================================================================
# CRS transformer composition
# ===================================================================

def test_fault_accepts_shared_coordinate_transformer() -> None:
    """Fault should use the CRS from an explicit shared transformer."""
    transformer = CoordinateTransformer("shared-transformer", lon0=FAULT_LON, lat0=FAULT_LAT)
    fault = Fault("composed-fault", transformer=transformer)
    fault.initialize_fault(
        pointpos="uc",
        lon=FAULT_LON,
        lat=FAULT_LAT,
        verdepth=UPPER_DEPTH_KM,
        strike=STRIKE_DEGREES,
        dip=DIP_DEGREES,
        length=FAULT_LENGTH_KM,
        width=FAULT_WIDTH_KM,
    )

    reference_fault = make_oblique_fault()
    assert fault.transformer is transformer
    assert fault.utm == transformer.utm
    assert fault.utmzone == transformer.utmzone
    assert_allclose(
        surface_vertices(fault),
        surface_vertices(reference_fault),
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )

    x_km, y_km = fault.ll2xy(FAULT_LON, FAULT_LAT)
    restored_lon, restored_lat = fault.xy2ll(x_km, y_km)
    assert_allclose(
        [restored_lon, restored_lat],
        [FAULT_LON, FAULT_LAT],
        rtol=0.0,
        atol=1e-8,
    )


# ===================================================================
# Surface geometry — corner vertices of the fault plane
# ===================================================================

def test_oblique_fault_surface_geometry() -> None:
    """The four corner vertices must satisfy known geometric constraints.

    This is the single most important test in the file.  It verifies
    that the fault plane is positioned, oriented, and sized correctly
    in 3-D UTM space.  We decompose the fault into two edge vectors:

    * **strike_edge** = ue − uo  (along strike, horizontal)
    * **dip_edge**    = be − ue  (down-dip, oblique)

    and independently check each vectorʼs magnitude, direction, and
    components against the known strike, dip, length, and width.
    """
    fault = make_oblique_fault()
    vertices = surface_vertices(fault)

    # Edge vectors.
    strike_edge = vertices[1] - vertices[0]   # ue − uo
    dip_edge = vertices[2] - vertices[1]      # be − ue

    # Pre-compute trig values for the assertions below.
    strike_radians = np.radians(STRIKE_DEGREES)
    dip_radians = np.radians(DIP_DEGREES)

    # ---- basic shape ----
    assert vertices.shape == (4, 3), (
        f"Expected (4, 3), got {vertices.shape} — "
        f"the fault should have exactly 4 corner vertices."
    )

    # ---- strike-edge vector ----
    # For strike=30°, the horizontal displacement from uo to ue is:
    #   ΔX = length × sin(strike) = 6.0 × 0.5 = 3.0 km  (easting)
    #   ΔY = length × cos(strike) = 6.0 × 0.866 = 5.196 km (northing)
    #   ΔZ = 0  (strike does not change depth)
    assert_allclose(
        strike_edge,
        [
            FAULT_LENGTH_KM * np.sin(strike_radians),
            FAULT_LENGTH_KM * np.cos(strike_radians),
            0.0,
        ],
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )

    # ---- dip-edge total length ----
    # The dip edge connects the upper end to the bottom end.  Its total
    # length must equal the fault width regardless of dip angle.
    assert_allclose(
        np.linalg.norm(dip_edge),
        FAULT_WIDTH_KM,
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )

    # ---- dip-edge horizontal projection ----
    # The horizontal component of the dip edge is width × cos(dip).
    # For dip=45°, cos(45°) = √2/2 ≈ 0.7071.
    # 4.0 × 0.7071 = 2.828 km.
    assert_allclose(
        np.linalg.norm(dip_edge[:2]),                     # XY (horizontal) magnitude
        FAULT_WIDTH_KM * np.cos(dip_radians),
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )

    # ---- dip-edge vertical component ----
    # The vertical drop is width × sin(dip), negative (deeper).
    # For dip=45°, sin(45°) ≈ 0.7071.
    # 4.0 × 0.7071 = 2.828 km downward (so −2.828 km in Z).
    assert_allclose(
        dip_edge[2],
        -FAULT_WIDTH_KM * np.sin(dip_radians),
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )

    # ---- upper centre position ----
    # The upper centre (uc) is the midpoint of the upper edge,
    # i.e. (uo + ue) / 2.  The fault stores this in ``ucp["upper center"]``.
    assert_allclose(
        vertices[:2].mean(axis=0),
        fault.ucp["upper center"],
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )


# ===================================================================
# Reference-point equivalence
# ===================================================================

@pytest.mark.parametrize("point_position", ["uo", "uc", "ue", "bo", "bc", "be", "cc"])
def test_reference_positions_reproduce_same_fault(point_position: str) -> None:
    """Every reference point must define the **same** fault geometry.

    ``initialize_fault`` accepts seven different ``pointpos`` values
    (uo, uc, ue, bo, bc, be, cc).  Regardless of which one the user
    provides, the resulting four corner vertices must be identical —
    they all describe the same physical fault plane.

    Strategy
    --------
    1. Build a canonical fault using the default (upper centre).
    2. Extract the UTM coordinates of the target reference point from
       the canonical faultʼs own corners.
    3. Convert those UTM coordinates back to lon/lat via ``xy2ll``.
    4. Build a **second** fault using that lon/lat/depth and the target
       ``pointpos``.
    5. Assert the two faults have identical corners.

    This simultaneously tests:
    * the ``__get_corner_vertices`` logic for every ``pointpos``,
    * the lon/lat → UTM → lon/lat round-trip,
    * the consistency between all seven reference frames.
    """
    # Step 1: canonical fault (upper centre).
    canonical_fault = make_oblique_fault()
    expected = surface_vertices(canonical_fault)

    # Step 2: compute the UTM coordinates of each reference point
    # from the canonical faultʼs corners.
    #
    # Vertex layout:  [0]=uo, [1]=ue, [2]=be, [3]=bo
    #
    #   uo (0) ———— ue (1)
    #    |            |
    #    |            |
    #   bo (3) ———— be (2)
    #
    reference_points = {
        "uo": expected[0],                          # upper origin
        "uc": expected[:2].mean(axis=0),            # (uo + ue) / 2
        "ue": expected[1],                          # upper end
        "bo": expected[3],                          # bottom origin
        "bc": expected[[2, 3]].mean(axis=0),        # (be + bo) / 2
        "be": expected[2],                          # bottom end
        "cc": expected.mean(axis=0),                # centroid of all four corners
    }

    x, y, depth = reference_points[point_position]

    # Step 3: UTM → lon/lat.
    lon, lat = canonical_fault.xy2ll(x, y)

    # Step 4: rebuild using the target reference point.
    actual_fault = make_oblique_fault(point_position, lon, lat, depth)

    # Step 5: corners must match exactly.
    assert_allclose(
        surface_vertices(actual_fault),
        expected,
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )


# ===================================================================
# Uniform rectangular discretization
# ===================================================================

def test_uniform_rectangular_patches_cover_oblique_plane() -> None:
    """Uniform rectangular patches must tile the fault without gaps.

    We discretize the 6×4 km oblique fault into sub-length=2.5 km and
    sub-width=1.5 km patches.  This produces ceil(6/2.5)=3 strike
    segments × ceil(4/1.5)=3 dip segments = 9 rectangular patches.

    Verified properties
    -------------------
    * Patch count matches ceil-based expectation.
    * Every patch has the correct edge lengths (total / n_segments).
    * All patch vertices lie on the original fault plane.
    * No patch vertex is above the surface (Z ≤ 0).
    """
    fault = make_oblique_fault()

    # Snapshot the reference plane corners *before* discretization
    # (discretization overwrites fault.patch_verts).
    reference_surface = surface_vertices(fault).copy()

    requested_length_km = 2.5
    requested_width_km = 1.5

    # Compute expected segment counts using the same ceil logic as the
    # production code (RectPatch.discretize).
    strike_segments = math.ceil(FAULT_LENGTH_KM / requested_length_km)   # 3
    dip_segments = math.ceil(FAULT_WIDTH_KM / requested_width_km)        # 3

    fault.construct_rect_patches(requested_length_km, requested_width_km)
    patches = np.asarray(fault.patch_verts)   # shape: (9, 4, 3)

    # ---- patch count ----
    assert patches.shape == (strike_segments * dip_segments, 4, 3), (
        f"Expected ({strike_segments * dip_segments}, 4, 3), "
        f"got {patches.shape}"
    )

    # ---- edge lengths ----
    # Each patchʼs strike edge (vertex 1 − vertex 0) must equal
    # total_length / n_strike_segments.
    assert_allclose(
        np.linalg.norm(patches[:, 1] - patches[:, 0], axis=1),
        FAULT_LENGTH_KM / strike_segments,
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )

    # Each patchʼs dip edge (vertex 2 − vertex 1) must equal
    # total_width / n_dip_segments.
    assert_allclose(
        np.linalg.norm(patches[:, 2] - patches[:, 1], axis=1),
        FAULT_WIDTH_KM / dip_segments,
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )

    # ---- coplanarity invariant ----
    assert_vertices_lie_on_plane(patches, reference_surface)

    # ---- depth bound ----
    # All vertices must be at or below the surface (Z ≤ 0).
    assert np.max(patches[:, :, 2]) <= 0.0, (
        f"Found vertices above the surface: "
        f"max Z = {np.max(patches[:, :, 2])} km"
    )


# ===================================================================
# Uniform triangular discretization
# ===================================================================

def test_uniform_triangular_patches_cover_oblique_plane() -> None:
    """Uniform triangular patches must tile the fault without gaps.

    Each rectangular cell is split into 2 triangles, so::

        n_triangles = 2 × n_strike_segments × n_dip_segments

    For edge_length=2.5 km on a 6×4 km fault:
        n_strike = ceil(6/2.5) = 3
        n_dip    = ceil(4/2.5) = 2
        n_tri    = 2 × 3 × 2 = 12

    Verified properties
    -------------------
    * Patch count matches the expectation.
    * Every triangle has strictly positive area (no degenerate triangles
      where two vertices coincide).
    * All vertices are coplanar with the fault plane.
    * No vertex is above the surface.
    """
    fault = make_oblique_fault()
    reference_surface = surface_vertices(fault).copy()

    requested_edge_km = 2.5

    # Expected segment counts (ceil logic from TriPatch.discretize_planar).
    strike_segments = math.ceil(FAULT_LENGTH_KM / requested_edge_km)   # 3
    dip_segments = math.ceil(FAULT_WIDTH_KM / requested_edge_km)       # 2
    expected_patch_count = 2 * strike_segments * dip_segments           # 12

    patch_count = fault.construct_tri_patches(requested_edge_km)
    patches = np.asarray(fault.tri_patch_verts)   # shape: (12, 3, 3)

    # ---- patch count ----
    assert patch_count == expected_patch_count, (
        f"construct_tri_patches returned {patch_count}, "
        f"expected {expected_patch_count}"
    )
    assert patches.shape == (expected_patch_count, 3, 3)

    # ---- no degenerate triangles ----
    # Compute the area of each triangle via half the magnitude of the
    # cross product of two edge vectors:  area = 0.5 × |AB × AC|.
    # A degenerate triangle (two coincident vertices or collinear points)
    # would have area = 0 (or extremely close to zero).
    triangle_areas = 0.5 * np.linalg.norm(
        np.cross(
            patches[:, 1] - patches[:, 0],   # edge AB
            patches[:, 2] - patches[:, 0],   # edge AC
        ),
        axis=1,
    )
    assert np.all(triangle_areas > 0.0), (
        f"Found {np.sum(triangle_areas <= 0.0)} degenerate triangles "
        f"(area ≤ 0).  Minimum area: {np.min(triangle_areas)}"
    )

    # ---- coplanarity invariant ----
    assert_vertices_lie_on_plane(patches, reference_surface)

    # ---- depth bound ----
    assert np.max(patches[:, :, 2]) <= 0.0


# ===================================================================
# Depth-varying rectangular discretization
# ===================================================================

def geometric_widths_until_covered(
    total_width: float,
    initial_width: float,
    ratio: float,
) -> np.ndarray:
    """Simulate the geometric width sequence produced by
    ``RectPatch._get_segments_size_varying``.

    Starting from ``initial_width``, each successive segment width is
    multiplied by ``ratio`` until the cumulative width reaches or
    exceeds ``total_width``.

    This is an **independent re-implementation** used to compute
    expected values without calling the production code.

    Parameters
    ----------
    total_width : float
        Total fault width to cover (km).
    initial_width : float
        Width of the first segment (km).
    ratio : float
        Growth factor per segment (≥ 1).

    Returns
    -------
    np.ndarray
        1-D array of segment widths.
    """
    widths = []
    covered_width = 0.0
    current_width = initial_width
    while covered_width < total_width:
        widths.append(current_width)
        covered_width += current_width
        current_width *= ratio
    return np.asarray(widths)


def test_depth_varying_rectangular_patches_follow_geometric_widths() -> None:
    """Depth-varying discretization must follow the geometric progression.

    When ``dip_vary_fct > 1``, the patch width grows by that factor with
    each successive row down-dip.  For dip_vary_fct=1.5 starting from
    1.0 km on a 4.0 km wide fault, the expected widths are::

        [1.0, 1.5, 2.25]   (sum = 4.75 km ≥ 4.0 km)

    The production code adjusts ``fault.width`` to the actual covered
    width (4.75 km).  This test verifies that the resulting patches
    have exactly these widths, that the fault width is updated, and
    that the deepest vertex is at the correct depth.
    """
    fault = make_oblique_fault()
    reference_surface = surface_vertices(fault).copy()

    patch_length_km = 2.0
    initial_patch_width_km = 1.0
    dip_ratio = 1.5

    # Independently compute the expected width sequence.
    expected_dip_widths = geometric_widths_until_covered(
        FAULT_WIDTH_KM,
        initial_patch_width_km,
        dip_ratio,
    )
    # → [1.0, 1.5, 2.25], sum = 4.75 km

    strike_segments = math.ceil(FAULT_LENGTH_KM / patch_length_km)

    fault.construct_rect_patches(
        patch_length_km,
        initial_patch_width_km,
        str_vary_fct=1.0,          # uniform along strike
        dip_vary_fct=dip_ratio,    # growing down-dip
        verbose=False,
    )
    patches = np.asarray(fault.patch_verts)

    # ---- patch count ----
    expected_total = len(expected_dip_widths) * strike_segments
    assert patches.shape == (expected_total, 4, 3), (
        f"Expected ({expected_total}, 4, 3), got {patches.shape}"
    )

    # ---- per-patch dip-edge lengths ----
    # The patches are ordered row-by-row (all strike patches of row 0,
    # then row 1, ...).  Within a row, all dip edges have the same length.
    actual_dip_widths = np.linalg.norm(patches[:, 2] - patches[:, 1], axis=1)
    assert_allclose(
        actual_dip_widths,
        np.repeat(expected_dip_widths, strike_segments),
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )

    # ---- updated fault width ----
    # The production code adjusts self.width to the cumulative covered width.
    assert_allclose(
        fault.width,
        expected_dip_widths.sum(),
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )

    # ---- deepest vertex depth ----
    # The deepest vertex should be at:
    #   upper_depth − total_width × sin(dip)
    # = −2.0 − 4.75 × sin(45°) = −2.0 − 3.359 = −5.359 km
    assert_allclose(
        np.min(patches[:, :, 2]),
        UPPER_DEPTH_KM - fault.width * np.sin(np.radians(DIP_DEGREES)),
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )

    # ---- coplanarity invariant ----
    assert_vertices_lie_on_plane(patches, reference_surface)


# ===================================================================
# Surface breaching — automatic trimming
# ===================================================================

def test_initial_surface_breach_is_trimmed_to_surface() -> None:
    """A fault whose upper edge is above the surface must be trimmed.

    When ``initialize_fault`` detects that the upper edge Z > 0
    (i.e. the fault breaches the surface), it must:

    1. Emit a ``UserWarning`` containing the word "breached".
    2. Set the upper-edge verticesʼ Z to exactly 0 (the surface).
    3. Reduce ``fault.width`` by the exposed portion.
    4. Recalculate the bottom-edge depth accordingly.

    This test creates a fault with upper depth = +1.0 km (above surface).
    The exposed portion is 1.0 / sin(45°) = 1.414 km along the dip.
    The remaining width should be 4.0 − 1.414 = 2.586 km.
    """
    # The upper edge is 1 km ABOVE the surface.
    original_upper_depth_km = 1.0
    dip_radians = np.radians(DIP_DEGREES)

    # Exposed portion along the dip direction:
    #   exposed = depth_above_surface / sin(dip) = 1.0 / sin(45°) = 1.414 km
    # Remaining width:
    #   new_width = original_width − exposed = 4.0 − 1.414 = 2.586 km
    expected_width_km = FAULT_WIDTH_KM - original_upper_depth_km / np.sin(dip_radians)

    # Depth of the bottom edge after trimming:
    #   bottom_Z = original_upper_Z − original_width × sin(dip)
    #            = 1.0 − 4.0 × sin(45°) = −1.828 km
    expected_bottom_depth_km = (
        original_upper_depth_km - FAULT_WIDTH_KM * np.sin(dip_radians)
    )

    # The constructor must emit a warning about breaching.
    with pytest.warns(UserWarning, match="breached"):
        fault = make_oblique_fault(depth=original_upper_depth_km)

    vertices = surface_vertices(fault)

    # ---- upper edge at surface ----
    # Both upper-origin and upper-end must have Z = 0.
    assert_allclose(
        vertices[:2, 2],                    # Z of uo and ue
        0.0,
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )

    # ---- bottom edge depth preserved relative to trimmed geometry ----
    assert_allclose(
        vertices[2:, 2],                    # Z of be and bo
        expected_bottom_depth_km,
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )

    # ---- width updated ----
    assert_allclose(
        fault.width,
        expected_width_km,
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )

    # ---- dip-edge length equals the new width ----
    assert_allclose(
        np.linalg.norm(vertices[2] - vertices[1]),
        expected_width_km,
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )


# ===================================================================
# extend_to_surface
# ===================================================================

def test_extend_to_surface_preserves_lower_edge() -> None:
    """Extending a buried fault to the surface must not move the bottom edge.

    ``extend_to_surface`` lengthens the fault upward until the upper edge
    reaches Z = 0.  The lower edge must remain exactly where it was — the
    extension adds material *above* the original upper edge.

    For our standard fault (upper depth = −2 km):
      extension needed to reach surface = 2.0 / sin(45°) = 2.828 km
      new width = 4.0 + 2.828 = 6.828 km
    """
    fault = make_oblique_fault()

    # Snapshot the original lower corners before extension.
    original_surface = surface_vertices(fault).copy()

    # Expected new width after extension:
    #   extra_width = |upper_depth| / sin(dip) = 2.0 / sin(45°) = 2.828 km
    #   new_width   = 4.0 + 2.828 = 6.828 km
    expected_width_km = (
        FAULT_WIDTH_KM - UPPER_DEPTH_KM / np.sin(np.radians(DIP_DEGREES))
    )

    fault.extend_to_surface()
    extended_surface = surface_vertices(fault)

    # ---- upper edge at surface ----
    assert_allclose(
        extended_surface[:2, 2],
        0.0,
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )

    # ---- lower edge UNCHANGED ----
    # This is the key invariant: the bottom of the fault must not move.
    assert_allclose(
        extended_surface[2:],               # new bottom corners
        original_surface[2:],               # original bottom corners
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )

    # ---- width updated ----
    assert_allclose(
        fault.width,
        expected_width_km,
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )
