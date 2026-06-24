import numpy as np
import pytest
from numpy.testing import assert_allclose

from seislip.seislip import GeoTrans


COORDINATE_ATOL_DEGREES = 1e-8
ZONE_38_CENTRAL_MERIDIAN = 45.0


def make_transformer() -> GeoTrans:
    return GeoTrans("test-coordinates", ellps="WGS84", utmzone="38N")


@pytest.mark.parametrize(
    ("lon", "lat"),
    [
        (44.0, 35.0),
        (42.0001, 35.0),
        (47.9999, 35.0),
        (ZONE_38_CENTRAL_MERIDIAN, 0.0),
    ],
)
def test_scalar_lonlat_round_trip(lon: float, lat: float) -> None:
    transformer = make_transformer()

    x_km, y_km = transformer.ll2xy(lon, lat)
    restored_lon, restored_lat = transformer.xy2ll(x_km, y_km)

    assert_allclose(
        [restored_lon, restored_lat],
        [lon, lat],
        rtol=0.0,
        atol=COORDINATE_ATOL_DEGREES,
    )


@pytest.mark.parametrize(
    ("lon", "lat"),
    [
        (np.array([44.0, 44.1, 44.2]), np.array([35.0, 35.1, 35.2])),
        (np.array([44.0]), np.array([35.0])),
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
    transformer = make_transformer()

    x_km, y_km = transformer.ll2xy(lon, lat)
    restored_lon, restored_lat = transformer.xy2ll(x_km, y_km)

    assert x_km.shape == lon.shape
    assert y_km.shape == lat.shape
    assert restored_lon.shape == lon.shape
    assert restored_lat.shape == lat.shape
    assert_allclose(
        restored_lon,
        lon,
        rtol=0.0,
        atol=COORDINATE_ATOL_DEGREES,
    )
    assert_allclose(
        restored_lat,
        lat,
        rtol=0.0,
        atol=COORDINATE_ATOL_DEGREES,
    )


def test_projected_coordinates_are_returned_in_kilometers() -> None:
    transformer = make_transformer()

    x_km, y_km = transformer.ll2xy(ZONE_38_CENTRAL_MERIDIAN, 0.0)

    assert_allclose([x_km, y_km], [500.0, 0.0], rtol=0.0, atol=1e-9)
