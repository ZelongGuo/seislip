"""
Tests for 3-D homogeneous transformation matrices (Transformation class).

The ``Transformation`` class builds a 4×4 homogeneous matrix that maps
between the UTM coordinate system and the local fault coordinate system.
It is used by both ``RectPatch`` and ``TriPatch`` as a **composed**
dependency (not via inheritance).

Coordinate systems
------------------
* **UTM** : X = easting, Y = northing, Z = zenith (all in km).
* **Fault** (right-handed):
  X = along strike, Y = opposite dip direction, Z = normal to fault plane.

Transformation order (left-multiply convention)
-----------------------------------------------
The matrix is built by successive left-multiplications::

    M = R_x · R_y · R_z · T

This is a "fixed coordinate system" convention — each rotation is applied
to the original (UTM) axes, not to the body axes.

Coverage
--------
* Pure translation (forward and inverse, tested independently).
* Composite rotation + translation round-trip.
* Single-point shape preservation — guards against unwanted squeezing.
"""

import numpy as np
from numpy.testing import assert_allclose

from seislip.utils.transformation import Transformation


# Absolute tolerance for coordinate comparisons, in km.
# 1e-12 km = 1e-9 m = 1 nanometre.  This is essentially machine epsilon
# for double-precision arithmetic on kilometre-scale coordinates and is
# tight enough to catch any matrix construction error.
TRANSFORM_ATOL = 1e-12


# ===================================================================
# Pure translation — forward
# ===================================================================

def test_translation_moves_points_by_requested_offset() -> None:
    """Forward translation must add the offset vector to every point.

    This is the simplest possible transformation — no rotation, only a
    shift of origin.  We test it in isolation because if translation
    alone is broken, nothing built on top of it can be trusted.
    """
    transformation = Transformation()

    # Define a known translation vector.
    transformation.translation((1.0, -2.0, 3.5))

    # Two test points: the origin and an off-origin point with mixed signs.
    # Using more than one point catches bugs that only affect non-zero
    # components (e.g. a matrix element accidentally left as identity).
    points = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, -1.0]])

    # Apply the forward transformation.
    transformed = transformation.forwars_trans(points)

    # Each transformed point must equal the original point plus the offset.
    # This is a "known input → known output" test — we do NOT compute the
    # expected value by calling another method of the same class.
    assert_allclose(
        transformed,
        points + np.array([1.0, -2.0, 3.5]),
        rtol=0.0,
        atol=TRANSFORM_ATOL,
    )


# ===================================================================
# Pure translation — inverse (independently verified)
# ===================================================================

def test_inverse_translation_recovers_independently_known_points() -> None:
    """Inverse translation must subtract the offset vector.

    Important
    ---------
    This test does **not** compute the expected answer by first calling
    ``forwars_trans`` and then inverting.  That would be a circular test:
    if both methods share a bug, the round-trip could still pass.

    Instead, we manually compute what the original points SHOULD be
    given known translated coordinates, and verify ``inverse_trans``
    recovers them independently.
    """
    transformation = Transformation()

    # Known translation.
    offset = np.array([4.0, -3.0, 2.0])
    transformation.translation(offset)

    # Build the inverse matrix.
    transformation.inverse()

    # These are points *after* translation.  We know the offset, so we
    # can independently compute what the originals must be:
    # original = translated − offset.
    translated_points = np.array([[5.0, -1.0, 7.0], [0.0, 2.0, -4.0]])

    # Apply the inverse transformation.
    original_points = transformation.inverse_trans(translated_points)

    # Compare against the independently computed expectation.
    assert_allclose(
        original_points,
        translated_points - offset,
        rtol=0.0,
        atol=TRANSFORM_ATOL,
    )


# ===================================================================
# Composite transformation — round-trip
# ===================================================================

def test_composite_transformation_round_trip() -> None:
    """Full rotation + translation round-trip must recover the input points.

    This is the most demanding test.  It chains three rotations (around
    X, Y, Z axes) and a translation, then applies the inverse.  Any
    error in:

    * matrix multiplication order (left-multiply convention),
    * rotation direction (sign of sin terms),
    * degree → radian conversion,
    * inverse matrix computation,

    will cause the round-trip to fail.

    Why these specific angles?
    --------------------------
    We deliberately avoid "nice" angles like 90°, 180°, or 0°, whose
    sines and cosines are 0, 1, or −1.  Bugs that swap sin/cos or
    flip a sign can survive at those special values.  Irregular angles
    (35°, −15°, 70°) force every matrix element to a non-trivial value,
    maximising the chance of exposing an error.
    """
    transformation = Transformation()

    # Build the composite matrix:  M = R_x(35°) · R_y(−15°) · R_z(70°) · T
    transformation.rotation_x(np.radians(35.0))
    transformation.rotation_y(np.radians(-15.0))
    transformation.rotation_z(np.radians(70.0))
    transformation.translation((100.0, 200.0, -5.0))

    # Compute the inverse once.
    transformation.inverse()

    # Test points selected to cover diverse regions of 3-D space:
    #   (0, 0, 0)      — the origin (stress-tests translation logic),
    #   (1.5, -2, 3)   — small coordinates with mixed signs,
    #   (-10, 4, -7.5) — larger magnitudes with negative Z.
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, -2.0, 3.0],
            [-10.0, 4.0, -7.5],
        ]
    )

    # Forward → inverse round-trip.
    transformed = transformation.forwars_trans(points)
    restored = transformation.inverse_trans(transformed)

    # Every coordinate of every point must return to its starting value.
    assert_allclose(restored, points, rtol=0.0, atol=TRANSFORM_ATOL)


# ===================================================================
# Single-point shape preservation
# ===================================================================

def test_single_point_matrix_preserves_documented_shape() -> None:
    """A single point (1, 3) must remain (1, 3) after transformation.

    Motivation
    ----------
    The internal ``_homogenous`` / ``_inhomogenous`` helpers add and
    strip the homogeneous coordinate.  A common NumPy pitfall is that
    operations on a (1, 3) array accidentally squeeze it to (3,),
    which would break any downstream code that indexes ``result[0]``
    expecting the first (and only) point.

    This test locks the documented contract: **input shape = output shape**.
    """
    transformation = Transformation()

    # A non-trivial transform so the point actually moves.
    transformation.rotation_z(np.radians(20.0))
    transformation.translation((1.0, 2.0, 3.0))
    transformation.inverse()

    # Shape (1, 3): one point, three coordinates.
    point = np.array([[4.0, -2.0, 1.0]])

    transformed = transformation.forwars_trans(point)
    restored = transformation.inverse_trans(transformed)

    # The critical assertions: shape must not be squeezed.
    assert transformed.shape == (1, 3), (
        f"Expected (1, 3), got {transformed.shape} — "
        f"the homogeneous helper may be squeezing single-point arrays."
    )
    assert restored.shape == (1, 3), (
        f"Expected (1, 3), got {restored.shape} — "
        f"the inverse homogeneous helper may be squeezing single-point arrays."
    )

    # And the value must be correct as well.
    assert_allclose(restored, point, rtol=0.0, atol=TRANSFORM_ATOL)
