"""
Tests for geographic ↔ UTM coordinate transformations (GeoTrans class).

These tests verify the core coordinate conversion that every other module
(InSAR, Fault, MultiFault) depends on.  If these tests fail, all downstream
geometry calculations are suspect.

Coverage
--------
* Scalar round-trip: lon/lat → UTM → lon/lat at zone center, zone boundaries,
  and the equator.
* Array round-trip: 1-D vectors, single-element arrays, and 2-D grids.
* Unit contract: ll2xy must return kilometres, not metres.
* Shape preservation: the output arrays must have the same dimensions as
  the input arrays.

Tolerances
----------
COORDINATE_ATOL_DEGREES = 1e-8 degrees ≈ 1.1 mm at the Earthʼs surface.
This is comfortably above the ~1 mm precision of UTM projections while
still being tight enough to catch meaningful errors.  Using a stricter
tolerance (e.g. 1e-10) risks spurious failures when pyproj is upgraded.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from seislip.seislip import GeoTrans


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Absolute tolerance for longitude/latitude comparisons, in degrees.
# 1e-8 deg ≈ 1.1 mm — appropriate for double-precision UTM round-trips.
COORDINATE_ATOL_DEGREES = 1e-8

# Central meridian of UTM zone 38 (covers 42°E – 48°E).
# On the central meridian the UTM easting is exactly 500 000 m = 500 km.
# On the equator the UTM northing is exactly 0 m (northern hemisphere).
# These two facts give us a "known answer" for the unit test below.
ZONE_38_CENTRAL_MERIDIAN = 45.0


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def make_transformer() -> GeoTrans:
    """Return a GeoTrans locked to UTM zone 38N (WGS84).

    We fix the zone explicitly rather than letting pyproj auto-select it.
    This keeps the tests deterministic — their outcome does not depend on
    the internal ordering of ``query_utm_crs_info``, which can vary across
    pyproj versions.
    """
    return GeoTrans("test-coordinates", ellps="WGS84", utmzone="38N")


# ===================================================================
# Scalar round-trip
# ===================================================================

@pytest.mark.parametrize(
    ("lon", "lat"),
    [
        # ---- motivation for each test point ---------------------------------
        #
        # (1) Zone centre, mid-latitude.
        #     This is your actual study area (Iran).  If this point fails
        #     the round-trip, nothing else matters.
        (44.0, 35.0),

        # (2) Western edge of zone 38 (boundary at 42°E).
        #     UTM distortion is largest near zone edges, so boundary-
        #     adjacent points are the most demanding test of precision.
        (42.0001, 35.0),

        # (3) Eastern edge of zone 38 (boundary at 48°E).
        #     Mirrors the western-edge test for symmetry.
        (47.9999, 35.0),

        # (4) Central meridian × equator.
        #     The equator is where northing changes fastest; combined with
        #     the central meridian this is an extreme-but-legal coordinate.
        (ZONE_38_CENTRAL_MERIDIAN, 0.0),
    ],
)
def test_scalar_lonlat_round_trip(lon: float, lat: float) -> None:
    """lon/lat → UTM → lon/lat must recover the original scalar coordinates.

    This is the most fundamental correctness check: a round-trip through
    the forward and inverse projections should return the starting point
    within numerical precision.  Any systematic error in ``ll2xy`` or
    ``xy2ll`` (wrong ellipsoid, wrong UTM zone, unit mistake) will
    immediately show up as a non-zero residual.
    """
    transformer = make_transformer()

    # Forward projection: geographic → UTM (km)
    x_km, y_km = transformer.ll2xy(lon, lat)

    # Inverse projection: UTM (km) → geographic
    restored_lon, restored_lat = transformer.xy2ll(x_km, y_km)

    # Both coordinates must return to their starting values.
    # We use absolute tolerance only (rtol=0) because relative tolerance
    # is meaningless for angles — a point at lon≈0 would require
    # essentially zero absolute error to satisfy any finite rtol.
    assert_allclose(
        [restored_lon, restored_lat],
        [lon, lat],
        rtol=0.0,
        atol=COORDINATE_ATOL_DEGREES,
    )


# ===================================================================
# Array round-trip
# ===================================================================

@pytest.mark.parametrize(
    ("lon", "lat"),
    [
        # ---- motivation for each test shape --------------------------------
        #
        # (1) 1-D arrays of length 3.
        #     Typical use-case: a list of observation points along a profile.
        (np.array([44.0, 44.1, 44.2]), np.array([35.0, 35.1, 35.2])),

        # (2) 1-D single-element arrays.
        #     Edge case: NumPy can accidentally squeeze (N,) → scalar,
        #     which would break downstream code that expects an array.
        (np.array([44.0]), np.array([35.0])),

        # (3) 2-D arrays (2×2 grid).
        #     Typical use-case: an InSAR image subset where lon/lat are
        #     meshgrid outputs.  Shape must survive the round-trip unchanged.
        (
            np.array([[44.00, 44.05], [44.10, 44.15]]),
            np.array([[35.00, 35.05], [35.10, 35.15]]),
        ),
    ],
)
def test_array_lonlat_round_trip_preserves_shape(
    lon: np.ndarray,
    lat: np.ndarray,
) -> None:
    """Array round-trip must preserve both values *and* array shape.

    Why shape matters
    -----------------
    The ``ll2xy`` and ``xy2ll`` methods delegate to pyprojʼs
    ``Transformer.transform``.  Some pyproj versions / configurations may
    flatten or reshape the output.  If, for example, a (100, 200) input
    returns a (20000,) output, every downstream indexing operation that
    assumes a 2-D grid will silently produce wrong results.

    This test guarantees that whatever shape goes in comes back out.
    """
    transformer = make_transformer()

    x_km, y_km = transformer.ll2xy(lon, lat)
    restored_lon, restored_lat = transformer.xy2ll(x_km, y_km)

    # ---- contract: shape preservation ----
    assert x_km.shape == lon.shape
    assert y_km.shape == lat.shape
    assert restored_lon.shape == lon.shape
    assert restored_lat.shape == lat.shape

    # ---- contract: value preservation ----
    assert_allclose(restored_lon, lon, rtol=0.0, atol=COORDINATE_ATOL_DEGREES)
    assert_allclose(restored_lat, lat, rtol=0.0, atol=COORDINATE_ATOL_DEGREES)


# ===================================================================
# Unit contract: ll2xy returns kilometres
# ===================================================================

def test_projected_coordinates_are_returned_in_kilometers() -> None:
    """Verify that ``ll2xy`` returns kilometres, not metres.

    How this test works (without touching private attributes)
    ----------------------------------------------------------
    UTM projection has a well-known property: on the **central meridian**
    of a zone, the easting is exactly 500 000 m (the false easting).
    On the **equator** (northern hemisphere), the northing is exactly 0 m.

    We pick the point (45°E, 0°N), which is the intersection of the
    central meridian of UTM zone 38 with the equator.  If ``ll2xy``
    correctly converts metres → kilometres, we must obtain::

        x = 500.0 km
        y =   0.0 km

    A result of (500000, 0) would mean the ÷1000 step inside ``ll2xy``
    was accidentally removed — a regression that is frighteningly easy
    to introduce during refactoring.
    """
    transformer = make_transformer()

    # Project the "magic point" whose UTM coordinates are known by
    # definition of the projection.
    x_km, y_km = transformer.ll2xy(ZONE_38_CENTRAL_MERIDIAN, 0.0)

    # 500.0 km easting, 0.0 km northing — by definition of UTM.
    assert_allclose([x_km, y_km], [500.0, 0.0], rtol=0.0, atol=1e-9)
