"""Tests for InSAR CRS transformer composition."""

import numpy as np
from numpy.testing import assert_allclose

from seislip import InSAR
from seislip.crs import CoordinateTransformer


def test_insar_accepts_shared_coordinate_transformer(capsys) -> None:
    """InSAR should use the CRS from an explicit shared transformer."""
    transformer = CoordinateTransformer("shared-transformer", lon0=44.0, lat0=35.0)
    insar = InSAR("composed-insar", transformer=transformer)
    capsys.readouterr()

    lon = np.array([[44.0, 44.1], [44.2, 44.3]])
    lat = np.array([[35.0, 35.1], [35.2, 35.3]])

    expected_x, expected_y = transformer.ll2xy(lon, lat)
    actual_x, actual_y = insar.ll2xy(lon, lat)

    assert insar.transformer is transformer
    assert insar.utm == transformer.utm
    assert insar.utmzone == transformer.utmzone
    assert_allclose(actual_x, expected_x, rtol=0.0, atol=1e-12)
    assert_allclose(actual_y, expected_y, rtol=0.0, atol=1e-12)

    restored_lon, restored_lat = insar.xy2ll(actual_x, actual_y)
    assert_allclose(restored_lon, lon, rtol=0.0, atol=1e-8)
    assert_allclose(restored_lat, lat, rtol=0.0, atol=1e-8)
