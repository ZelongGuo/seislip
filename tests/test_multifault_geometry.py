"""Tests for MultiFault CRS transformer composition."""

import numpy as np

from seislip.crs import CoordinateTransformer
from seislip.fault import MultiFault


def test_multifault_passes_shared_transformer_to_segments(capsys) -> None:
    """Chained segment construction should preserve a shared CRS transformer."""
    transformer = CoordinateTransformer("shared-transformer", lon0=44.0, lat0=35.0)
    multifault = MultiFault("composed-multifault", transformer=transformer)

    trace_points = [
        (44.00, 35.00, -1.0),
        (44.05, 35.05, -1.0),
        (44.10, 35.00, -1.0),
    ]
    multifault.from_trace_rectangles(
        trace_points,
        dips=[45.0, 50.0],
        widths=[3.0, 4.0],
        coord_type="LL",
    )
    capsys.readouterr()

    assert multifault.transformer is transformer
    assert multifault.surface_type == "chained"
    assert len(multifault.segments) == 2

    for segment in multifault.segments:
        assert segment.transformer is transformer
        assert segment.utm == multifault.utm
        assert segment.utmzone == multifault.utmzone
        assert segment.length > 0.0
        vertices = np.asarray(segment.patch_verts, dtype=float)
        assert vertices.shape == (1, 4, 3)
        assert np.max(vertices[:, :, 2]) <= 0.0
