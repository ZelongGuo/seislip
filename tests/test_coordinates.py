import numpy as np
from numpy.testing import assert_allclose

from seislip.seislip import GeoTrans


REFERENCE_LON = 44.0
REFERENCE_LAT = 35.0


def make_transformer() -> GeoTrans:
    return GeoTrans("test-coordinates", lon0=REFERENCE_LON, lat0=REFERENCE_LAT)


def test_scalar_lonlat_round_trip() -> None:
    transformer = make_transformer()

    x_km, y_km = transformer.ll2xy(REFERENCE_LON, REFERENCE_LAT)
    lon, lat = transformer.xy2ll(x_km, y_km)

    assert_allclose([lon, lat], [REFERENCE_LON, REFERENCE_LAT], atol=1e-10)


def test_array_lonlat_round_trip_preserves_shape() -> None:
    transformer = make_transformer()
    lon = np.array([[44.00, 44.05], [44.10, 44.15]])
    lat = np.array([[35.00, 35.05], [35.10, 35.15]])

    x_km, y_km = transformer.ll2xy(lon, lat)
    restored_lon, restored_lat = transformer.xy2ll(x_km, y_km)

    assert x_km.shape == lon.shape
    assert y_km.shape == lat.shape
    assert_allclose(restored_lon, lon, atol=1e-10)
    assert_allclose(restored_lat, lat, atol=1e-10)


def test_projected_coordinates_are_returned_in_kilometers() -> None:
    transformer = make_transformer()

    x_km, y_km = transformer.ll2xy(REFERENCE_LON, REFERENCE_LAT)
    x_m, y_m = transformer.proj2utm.transform(REFERENCE_LON, REFERENCE_LAT)

    assert_allclose([x_km, y_km], [x_m / 1000.0, y_m / 1000.0])

