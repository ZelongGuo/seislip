import numpy as np
from numpy.testing import assert_allclose

from seislip.utils.transformation import Transformation


TRANSFORM_ATOL = 1e-12


def test_translation_moves_points_by_requested_offset() -> None:
    transformation = Transformation()
    transformation.translation((1.0, -2.0, 3.5))

    points = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, -1.0]])
    transformed = transformation.forwars_trans(points)

    assert_allclose(
        transformed,
        points + np.array([1.0, -2.0, 3.5]),
        rtol=0.0,
        atol=TRANSFORM_ATOL,
    )


def test_inverse_translation_recovers_independently_known_points() -> None:
    transformation = Transformation()
    offset = np.array([4.0, -3.0, 2.0])
    transformation.translation(offset)
    transformation.inverse()
    translated_points = np.array([[5.0, -1.0, 7.0], [0.0, 2.0, -4.0]])

    original_points = transformation.inverse_trans(translated_points)

    assert_allclose(
        original_points,
        translated_points - offset,
        rtol=0.0,
        atol=TRANSFORM_ATOL,
    )


def test_composite_transformation_round_trip() -> None:
    transformation = Transformation()
    transformation.rotation_x(np.radians(35.0))
    transformation.rotation_y(np.radians(-15.0))
    transformation.rotation_z(np.radians(70.0))
    transformation.translation((100.0, 200.0, -5.0))
    transformation.inverse()
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, -2.0, 3.0],
            [-10.0, 4.0, -7.5],
        ]
    )

    transformed = transformation.forwars_trans(points)
    restored = transformation.inverse_trans(transformed)

    assert_allclose(restored, points, rtol=0.0, atol=TRANSFORM_ATOL)


def test_single_point_matrix_preserves_documented_shape() -> None:
    transformation = Transformation()
    transformation.rotation_z(np.radians(20.0))
    transformation.translation((1.0, 2.0, 3.0))
    transformation.inverse()
    point = np.array([[4.0, -2.0, 1.0]])

    transformed = transformation.forwars_trans(point)
    restored = transformation.inverse_trans(transformed)

    assert transformed.shape == (1, 3)
    assert restored.shape == (1, 3)
    assert_allclose(restored, point, rtol=0.0, atol=TRANSFORM_ATOL)
