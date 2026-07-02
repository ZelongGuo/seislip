"""Compatibility tests for domain objects that expose CRS behavior."""

import numpy as np
import pytest

from seislip.crs import GeoTrans
from seislip.data.insar import InSAR
from seislip.fault.fault import Fault
from seislip.fault.multifault import MultiFault


DOMAIN_CLASSES = (Fault, InSAR, MultiFault)


@pytest.mark.parametrize("domain_cls", DOMAIN_CLASSES)
def test_legacy_domain_objects_keep_direct_crs_api(domain_cls):
    obj = domain_cls("legacy", lon0=44.0, lat0=35.0)

    assert isinstance(obj, GeoTrans)
    assert obj.transformer is obj

    x_km, y_km = obj.ll2xy(44.0, 35.0)
    lon, lat = obj.xy2ll(x_km, y_km)

    np.testing.assert_allclose(lon, 44.0, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(lat, 35.0, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("domain_cls", DOMAIN_CLASSES)
def test_composed_domain_objects_keep_direct_crs_api(domain_cls):
    transformer = GeoTrans("shared", lon0=44.0, lat0=35.0)
    obj = domain_cls("composed", transformer=transformer)

    assert isinstance(obj, GeoTrans)
    assert obj.transformer is transformer

    x_km, y_km = obj.ll2xy(44.0, 35.0)
    lon, lat = obj.xy2ll(x_km, y_km)

    np.testing.assert_allclose(lon, 44.0, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(lat, 35.0, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("domain_cls", DOMAIN_CLASSES)
def test_composed_domain_objects_share_transformer_state(domain_cls):
    transformer = GeoTrans("shared", lon0=44.0, lat0=35.0, utmzone="38N")
    obj = domain_cls("composed", transformer=transformer)

    for attr in ("lon0", "lat0", "ellps", "utmzone"):
        assert getattr(obj, attr) == getattr(transformer, attr)

    for attr in ("wgs", "utm", "proj2utm", "proj2wgs"):
        assert getattr(obj, attr) is getattr(transformer, attr)
