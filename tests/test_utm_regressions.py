"""
Regression tests for previously fixed UTM-zone selection bugs in GeoTrans.

These tests guard two behaviours that have caused coordinate errors:

* **Auto-zone must use the zone containing the reference point**, not the
  first CRS whose area intersects a broader search area.

* **Explicit southern-hemisphere zones must preserve the ``S`` suffix**,
  so ``"36S"`` resolves to EPSG:32736 rather than the northern zone.
"""

import pytest
from numpy.testing import assert_allclose
from pyproj import CRS, Transformer

from seislip.crs import GeoTrans


# ===================================================================
# Bug 1: automatic zone selection picks the wrong zone at boundaries
# ===================================================================

def test_automatic_utm_zone_contains_reference_point() -> None:
    """Auto-selected UTM zone must be the one that *contains* lon0.

    Setup
    -----
    We place the reference point at **lon = 1.5°E, lat = 50°N**.
    The ±2° area-of-interest used by ``__set_zone`` is therefore
    **lon ∈ [−0.5°, 3.5°]**, which straddles the 0° meridian:

    * UTM zone 30 covers  0°W (0°E) to 6°W  — wait, zone 30 is 6°W–0°.
      Actually: zone 30 = 6°W to 0° (Greenwich); zone 31 = 0° to 6°E.

    The reference point (1.5°E) is unambiguously inside zone 31.
    However, the western edge of the AoI (−0.5°) barely touches zone 30.
    If ``query_utm_crs_info`` returns [zone_30, zone_31] and the code
    takes ``utm_crs_list[0]``, it will select the wrong zone.

    Expected behaviour
    ------------------
    * ``transformer.utm`` should be EPSG:32631 (zone 31 North, WGS84).
    * ``transformer.utmzone`` should be ``"31N"``.

    Previous buggy behaviour
    ------------------------
    The code selected EPSG:32630 (zone 30 North) because it queried a
    wider AoI and took the first intersecting CRS.
    """
    # lon0=1.5 → AoI lon ∈ [−0.5, 3.5] crosses the zone 30/31 boundary at 0°.
    transformer = GeoTrans("automatic-zone", lon0=1.5, lat0=50.0)

    # The containing zone for lon=1.5° is zone 31 North.
    expected_crs = CRS.from_epsg(32631)

    assert transformer.utm == expected_crs
    assert transformer.utmzone == "31N"


# ===================================================================
# Bug 2: southern-hemisphere letter is discarded
# ===================================================================

def test_explicit_southern_utm_zone_preserves_hemisphere() -> None:
    """Explicitly specifying ``utmzone="36S"`` must produce a southern CRS.

    Setup
    -----
    We create a ``GeoTrans`` with ``utmzone="36S"``.  The previous buggy
    code path was::

        if isinstance(utmzone, str):
            zone_num = int(''.join(filter(str.isdigit, utmzone)))
            # "36S" → zone_num = 36  (the 'S' is discarded!)
        self.utm = CRS(proj='utm', zone=zone_num, ellps=ellps)
        # CRS(proj='utm', zone=36) defaults to NORTHERN hemisphere.

    Expected behaviour
    ------------------
    * ``self.utm`` should resolve to EPSG:32736 (zone 36 South).
    * ``self.utmzone`` should preserve ``"36S"``.
    * Projecting a southern-hemisphere point (30°E, 25°S) must match
      the pyproj reference transform for EPSG:32736.

    Previous buggy behaviour
    ------------------------
    The 'S' suffix is stripped, so the CRS silently defaults to the
    northern hemisphere (EPSG:32636).  Coordinates in the southern
    hemisphere will be offset by thousands of kilometres.
    """
    # Create transformer with explicit southern-hemisphere zone.
    transformer = GeoTrans("southern-zone", ellps="WGS84", utmzone="36S")

    # The expected CRS: UTM zone 36, SOUTHERN hemisphere, WGS84.
    expected_crs = CRS.from_epsg(32736)

    assert transformer.utm == expected_crs
    assert transformer.utmzone == "36S"

    # Verify the projection numerically for a point in southern Africa
    # (roughly 30°E, 25°S).  We compute the expected UTM coordinates
    # independently using a fresh pyproj Transformer — this is NOT a
    # round-trip test; it is a direct comparison against a known-good
    # projection.
    lon, lat = 30.0, -25.0
    x_km, y_km = transformer.ll2xy(lon, lat)

    # Build an independent reference transformer: WGS84 → EPSG:32736.
    expected_transformer = Transformer.from_crs(
        CRS.from_epsg(4326),    # WGS84 geographic
        expected_crs,            # UTM zone 36 South
        always_xy=True,
    )
    expected_x_m, expected_y_m = expected_transformer.transform(lon, lat)

    # Compare our GeoTrans output against the reference.
    # NOTE: atol=1e-6 km = 1 mm — the projection itself is accurate to
    # ~1 mm, so this tolerance is appropriate.
    assert_allclose(
        [x_km, y_km],
        [expected_x_m / 1000.0, expected_y_m / 1000.0],
        rtol=0.0,
        atol=1e-6,
    )


# ===================================================================
# Bug 3: explicit UTM zones must work with the default ellipsoid name
# ===================================================================

def test_explicit_utm_zone_accepts_default_ellipsoid_name() -> None:
    """The default ``ellps="WGS 84"`` must work for explicit UTM zones.

    Previous buggy behaviour
    ------------------------
    Explicit-zone initialization passed the display name ``"WGS 84"``
    directly into a PROJ UTM constructor, which expects a compact token
    such as ``"WGS84"``.  That made this otherwise valid call raise a
    pyproj ``CRSError``.
    """
    transformer = GeoTrans("default-ellipsoid-explicit-zone", utmzone="38N")

    assert transformer.utm == CRS.from_epsg(32638)
    assert transformer.utmzone == "38N"


# ===================================================================
# Strict manual UTM zone input
# ===================================================================

@pytest.mark.parametrize("utmzone", [38, "38", "38T", "0N", "61S", "N38", ""])
def test_explicit_utm_zone_requires_number_and_hemisphere(utmzone) -> None:
    """Manual UTM zones must be explicit ``<zone><hemisphere>`` strings.

    ``GeoTrans`` no longer guesses the hemisphere from ``lat0`` or from a
    missing suffix.  Users who choose the manual path must specify ``N`` or
    ``S`` themselves, where the suffix means northern/southern hemisphere and
    not an MGRS latitude band.
    """
    with pytest.raises((TypeError, ValueError)):
        GeoTrans("ambiguous-zone", utmzone=utmzone)


# ===================================================================
# Automatic UTM initialization requires both lon0 and lat0
# ===================================================================

@pytest.mark.parametrize(
    ("kwargs"),
    [
        {},
        {"lon0": 44.0},
        {"lat0": 35.0},
    ],
)
def test_automatic_utm_zone_requires_lon0_and_lat0(kwargs) -> None:
    """Automatic UTM selection needs both reference coordinates.

    Without an explicit ``utmzone``, ``GeoTrans`` must know the reference
    longitude and latitude so pyproj can select the UTM CRS containing that
    point.  Missing either coordinate is a caller error and should raise a
    clear ``ValueError`` rather than relying on Python ``assert`` statements.
    """
    with pytest.raises(ValueError, match="either an explicit utmzone"):
        GeoTrans("missing-reference", **kwargs)
