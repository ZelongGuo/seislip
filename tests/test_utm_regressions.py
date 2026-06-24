import pytest
from numpy.testing import assert_allclose
from pyproj import CRS, Transformer

from seislip.seislip import GeoTrans


@pytest.mark.xfail(
    strict=True,
    reason="automatic UTM selection chooses the first intersecting zone instead of the containing zone",
)
def test_automatic_utm_zone_contains_reference_point() -> None:
    transformer = GeoTrans("automatic-zone", lon0=44.0, lat0=35.0)
    expected_crs = CRS.from_epsg(32638)

    assert transformer.utm == expected_crs
    assert transformer.utmzone == "38N"


@pytest.mark.xfail(
    strict=True,
    reason="explicit UTM zone parsing discards the southern-hemisphere suffix",
)
def test_explicit_southern_utm_zone_preserves_hemisphere() -> None:
    transformer = GeoTrans("southern-zone", ellps="WGS84", utmzone="36S")
    expected_crs = CRS.from_epsg(32736)

    assert transformer.utm == expected_crs
    assert transformer.utmzone == "36S"

    lon, lat = 30.0, -25.0
    x_km, y_km = transformer.ll2xy(lon, lat)
    expected_transformer = Transformer.from_crs(
        CRS.from_epsg(4326),
        expected_crs,
        always_xy=True,
    )
    expected_x_m, expected_y_m = expected_transformer.transform(lon, lat)

    assert_allclose(
        [x_km, y_km],
        [expected_x_m / 1000.0, expected_y_m / 1000.0],
        atol=1e-9,
    )

