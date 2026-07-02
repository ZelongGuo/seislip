"""Tests for InSAR CRS transformer composition."""

import numpy as np
from numpy.testing import assert_allclose

from seislip import InSAR
from seislip.crs import CoordinateTransformer


def test_insar_accepts_shared_coordinate_transformer(capsys) -> None:
    """A composed CRS transformer must match legacy InSAR coordinates."""
    legacy = InSAR("legacy-insar", lon0=44.0, lat0=35.0)
    transformer = CoordinateTransformer("shared-transformer", lon0=44.0, lat0=35.0)
    composed = InSAR("composed-insar", transformer=transformer)
    capsys.readouterr()

    lon = np.array([[44.0, 44.1], [44.2, 44.3]])
    lat = np.array([[35.0, 35.1], [35.2, 35.3]])

    legacy_x, legacy_y = legacy.ll2xy(lon, lat)
    composed_x, composed_y = composed.ll2xy(lon, lat)

    assert composed.transformer is transformer
    assert legacy.transformer is legacy
    assert composed.utm == legacy.utm
    assert composed.utmzone == legacy.utmzone
    assert_allclose(composed_x, legacy_x, rtol=0.0, atol=1e-12)
    assert_allclose(composed_y, legacy_y, rtol=0.0, atol=1e-12)

    restored_lon, restored_lat = composed.xy2ll(composed_x, composed_y)
    assert_allclose(restored_lon, lon, rtol=0.0, atol=1e-8)
    assert_allclose(restored_lat, lat, rtol=0.0, atol=1e-8)
