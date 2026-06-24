import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from seislip.fault import Fault


FAULT_LON = 44.0
FAULT_LAT = 35.0
UPPER_DEPTH_KM = -2.0
STRIKE_DEGREES = 30.0
DIP_DEGREES = 45.0
FAULT_LENGTH_KM = 6.0
FAULT_WIDTH_KM = 4.0
GEOMETRY_ATOL_KM = 1e-8


def make_oblique_fault(
    point_position: str = "uc",
    lon: float = FAULT_LON,
    lat: float = FAULT_LAT,
    depth: float = UPPER_DEPTH_KM,
) -> Fault:
    fault = Fault("oblique-fault", lon0=FAULT_LON, lat0=FAULT_LAT)
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


def surface_vertices(fault: Fault) -> np.ndarray:
    return np.asarray(fault.patch_verts, dtype=float)[0]


def assert_vertices_lie_on_plane(
    vertices: np.ndarray,
    reference_surface: np.ndarray,
) -> None:
    origin = reference_surface[0]
    strike_edge = reference_surface[1] - origin
    dip_edge = reference_surface[3] - origin
    normal = np.cross(strike_edge, dip_edge)
    normal /= np.linalg.norm(normal)
    distances = (np.asarray(vertices).reshape(-1, 3) - origin) @ normal

    assert_allclose(distances, 0.0, rtol=0.0, atol=GEOMETRY_ATOL_KM)


def test_oblique_fault_surface_geometry() -> None:
    fault = make_oblique_fault()
    vertices = surface_vertices(fault)
    strike_edge = vertices[1] - vertices[0]
    dip_edge = vertices[2] - vertices[1]
    strike_radians = np.radians(STRIKE_DEGREES)
    dip_radians = np.radians(DIP_DEGREES)

    assert vertices.shape == (4, 3)
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
    assert_allclose(
        np.linalg.norm(dip_edge),
        FAULT_WIDTH_KM,
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )
    assert_allclose(
        np.linalg.norm(dip_edge[:2]),
        FAULT_WIDTH_KM * np.cos(dip_radians),
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )
    assert_allclose(
        dip_edge[2],
        -FAULT_WIDTH_KM * np.sin(dip_radians),
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )
    assert_allclose(
        vertices[:2].mean(axis=0),
        fault.ucp["upper center"],
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )


@pytest.mark.parametrize("point_position", ["uo", "uc", "ue", "bo", "bc", "be", "cc"])
def test_reference_positions_reproduce_same_fault(point_position: str) -> None:
    canonical_fault = make_oblique_fault()
    expected = surface_vertices(canonical_fault)
    reference_points = {
        "uo": expected[0],
        "uc": expected[:2].mean(axis=0),
        "ue": expected[1],
        "bo": expected[3],
        "bc": expected[[2, 3]].mean(axis=0),
        "be": expected[2],
        "cc": expected.mean(axis=0),
    }
    x, y, depth = reference_points[point_position]
    lon, lat = canonical_fault.xy2ll(x, y)

    actual_fault = make_oblique_fault(point_position, lon, lat, depth)

    assert_allclose(
        surface_vertices(actual_fault),
        expected,
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )


def test_uniform_rectangular_patches_cover_oblique_plane() -> None:
    fault = make_oblique_fault()
    reference_surface = surface_vertices(fault).copy()
    requested_length_km = 2.5
    requested_width_km = 1.5
    strike_segments = math.ceil(FAULT_LENGTH_KM / requested_length_km)
    dip_segments = math.ceil(FAULT_WIDTH_KM / requested_width_km)

    fault.construct_rect_patches(requested_length_km, requested_width_km)
    patches = np.asarray(fault.patch_verts)

    assert patches.shape == (strike_segments * dip_segments, 4, 3)
    assert_allclose(
        np.linalg.norm(patches[:, 1] - patches[:, 0], axis=1),
        FAULT_LENGTH_KM / strike_segments,
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )
    assert_allclose(
        np.linalg.norm(patches[:, 2] - patches[:, 1], axis=1),
        FAULT_WIDTH_KM / dip_segments,
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )
    assert_vertices_lie_on_plane(patches, reference_surface)
    assert np.max(patches[:, :, 2]) <= 0.0


def test_uniform_triangular_patches_cover_oblique_plane() -> None:
    fault = make_oblique_fault()
    reference_surface = surface_vertices(fault).copy()
    requested_edge_km = 2.5
    strike_segments = math.ceil(FAULT_LENGTH_KM / requested_edge_km)
    dip_segments = math.ceil(FAULT_WIDTH_KM / requested_edge_km)
    expected_patch_count = 2 * strike_segments * dip_segments

    patch_count = fault.construct_tri_patches(requested_edge_km)
    patches = np.asarray(fault.tri_patch_verts)
    triangle_areas = 0.5 * np.linalg.norm(
        np.cross(patches[:, 1] - patches[:, 0], patches[:, 2] - patches[:, 0]),
        axis=1,
    )

    assert patch_count == expected_patch_count
    assert patches.shape == (expected_patch_count, 3, 3)
    assert np.all(triangle_areas > 0.0)
    assert_vertices_lie_on_plane(patches, reference_surface)
    assert np.max(patches[:, :, 2]) <= 0.0


def geometric_widths_until_covered(
    total_width: float,
    initial_width: float,
    ratio: float,
) -> np.ndarray:
    widths = []
    covered_width = 0.0
    current_width = initial_width
    while covered_width < total_width:
        widths.append(current_width)
        covered_width += current_width
        current_width *= ratio
    return np.asarray(widths)


def test_depth_varying_rectangular_patches_follow_geometric_widths() -> None:
    fault = make_oblique_fault()
    reference_surface = surface_vertices(fault).copy()
    patch_length_km = 2.0
    initial_patch_width_km = 1.0
    dip_ratio = 1.5
    expected_dip_widths = geometric_widths_until_covered(
        FAULT_WIDTH_KM,
        initial_patch_width_km,
        dip_ratio,
    )
    strike_segments = math.ceil(FAULT_LENGTH_KM / patch_length_km)

    fault.construct_rect_patches(
        patch_length_km,
        initial_patch_width_km,
        str_vary_fct=1.0,
        dip_vary_fct=dip_ratio,
        verbose=False,
    )
    patches = np.asarray(fault.patch_verts)
    actual_dip_widths = np.linalg.norm(patches[:, 2] - patches[:, 1], axis=1)

    assert patches.shape == (len(expected_dip_widths) * strike_segments, 4, 3)
    assert_allclose(
        actual_dip_widths,
        np.repeat(expected_dip_widths, strike_segments),
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )
    assert_allclose(
        fault.width,
        expected_dip_widths.sum(),
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )
    assert_allclose(
        np.min(patches[:, :, 2]),
        UPPER_DEPTH_KM - fault.width * np.sin(np.radians(DIP_DEGREES)),
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )
    assert_vertices_lie_on_plane(patches, reference_surface)


def test_initial_surface_breach_is_trimmed_to_surface() -> None:
    original_upper_depth_km = 1.0
    dip_radians = np.radians(DIP_DEGREES)
    expected_width_km = FAULT_WIDTH_KM - original_upper_depth_km / np.sin(dip_radians)
    expected_bottom_depth_km = original_upper_depth_km - FAULT_WIDTH_KM * np.sin(dip_radians)

    with pytest.warns(UserWarning, match="breached"):
        fault = make_oblique_fault(depth=original_upper_depth_km)
    vertices = surface_vertices(fault)

    assert_allclose(vertices[:2, 2], 0.0, rtol=0.0, atol=GEOMETRY_ATOL_KM)
    assert_allclose(
        vertices[2:, 2],
        expected_bottom_depth_km,
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )
    assert_allclose(fault.width, expected_width_km, rtol=0.0, atol=GEOMETRY_ATOL_KM)
    assert_allclose(
        np.linalg.norm(vertices[2] - vertices[1]),
        expected_width_km,
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )


def test_extend_to_surface_preserves_lower_edge() -> None:
    fault = make_oblique_fault()
    original_surface = surface_vertices(fault).copy()
    expected_width_km = FAULT_WIDTH_KM - UPPER_DEPTH_KM / np.sin(np.radians(DIP_DEGREES))

    fault.extend_to_surface()
    extended_surface = surface_vertices(fault)

    assert_allclose(extended_surface[:2, 2], 0.0, rtol=0.0, atol=GEOMETRY_ATOL_KM)
    assert_allclose(
        extended_surface[2:],
        original_surface[2:],
        rtol=0.0,
        atol=GEOMETRY_ATOL_KM,
    )
    assert_allclose(fault.width, expected_width_km, rtol=0.0, atol=GEOMETRY_ATOL_KM)
