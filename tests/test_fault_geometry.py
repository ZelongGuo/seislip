import numpy as np
from numpy.testing import assert_allclose

from seislip.fault import Fault


def make_vertical_fault(point_position: str = "uc") -> Fault:
    fault = Fault("vertical-fault", lon0=44.0, lat0=35.0)
    fault.initialize_fault(
        pointpos=point_position,
        lon=44.0,
        lat=35.0,
        verdepth=-1.0,
        strike=0.0,
        dip=90.0,
        length=4.0,
        width=2.0,
    )
    return fault


def test_vertical_fault_surface_geometry() -> None:
    fault = make_vertical_fault()
    vertices = np.asarray(fault.patch_verts)

    assert vertices.shape == (1, 4, 3)
    assert_allclose(vertices[0, :, 2], [-1.0, -1.0, -3.0, -3.0])
    assert_allclose(np.linalg.norm(vertices[0, 1] - vertices[0, 0]), 4.0)
    assert_allclose(np.linalg.norm(vertices[0, 2] - vertices[0, 1]), 2.0)
    assert_allclose(vertices[0, :2].mean(axis=0), fault.ucp["upper center"])


def test_upper_center_aliases_produce_same_geometry() -> None:
    expected = np.asarray(make_vertical_fault("uc").patch_verts)

    for alias in ("UC", "upper center", "upper_center"):
        actual = np.asarray(make_vertical_fault(alias).patch_verts)
        assert_allclose(actual, expected)


def test_uniform_rectangular_patch_count_and_shape() -> None:
    fault = make_vertical_fault()

    fault.construct_rect_patches(sublength=2.0, subwidth=1.0)
    patches = np.asarray(fault.patch_verts)

    assert patches.shape == (4, 4, 3)
    assert_allclose(patches[:, :, 0], patches[0, 0, 0], atol=1e-12)
    assert np.max(patches[:, :, 2]) <= 0.0


def test_uniform_triangular_patch_count_and_shape() -> None:
    fault = make_vertical_fault()

    patch_count = fault.construct_tri_patches(max_edge_length=1.0)
    patches = np.asarray(fault.tri_patch_verts)

    assert patch_count == 16
    assert patches.shape == (16, 3, 3)
    assert_allclose(patches[:, :, 0], patches[0, 0, 0], atol=1e-12)
    assert np.max(patches[:, :, 2]) <= 0.0

