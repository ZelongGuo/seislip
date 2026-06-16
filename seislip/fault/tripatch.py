#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triangular Fault Patch Construction.

Created on 17.01.2026

@author: Zelong Guo
"""
__author__ = "Zelong Guo"

# standard libs
import sys
from typing import List, Callable, Optional
import warnings

import numpy as np
import math


if __name__ == "__main__":
    sys.path.append("../")
    from seislip.utils.transformation import Transformation
else:
    from ..utils.transformation import Transformation


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
class TriPatch(object):
    """Triangular fault patch discretization for planar fault surfaces.

    Creates a triangular mesh on a planar fault surface using grid-based triangulation.
    Each rectangle is split into 2 triangles.

    Coordinate Systems:
    - UTM: X (easting), Y (northing), Z (zenith), units in km
    - Fault coordinate system (right-hand):
        X: along strike direction
        Y: opposite direction of dip (downward along fault plane)
        Z: normal direction from fault plane
    """

    def __init__(self, upc, strike, dip, length, width):
        """Initialize fault parameters and construct transformation matrix.

        Args:
            - upc: Upper center point in UTM coordinates: {"upper center": (x, y, z)}
            - strike: Strike angle (degrees)
            - dip: Dip angle (degrees)
            - length: Fault length along strike (km)
            - width: Fault width along dip (km)
        """
        # Store fault parameters
        self.upc = upc
        self.strike = strike
        self.dip = dip
        self.length = length
        self.width = width

        # Build transformation matrix (same as RectPatch)
        utmx_uc, utmy_uc, utmz_uc = self.upc["upper center"][0], self.upc["upper center"][1], self.upc["upper center"][2]
        trans = Transformation()
        trans.rotation_x(np.radians(self.dip))
        trans.rotation_z(np.radians(90 - self.strike))
        trans.translation(T=(utmx_uc, utmy_uc, utmz_uc))
        trans.inverse()
        self.trans = trans

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def utm2fault(self, points):
        """UTM to fault coordinate system.

        Args:
            - points: Point lists in UTM coordinates, m x 3 list/array

        Returns:
            - Point list in fault coordinates, m x 3 list/array

        Coordinate system:
        Input: UTM (X=easting, Y=northing, Z=zenith) in km
        Output: Fault (X=along strike, Y=opposite dip, Z=normal) in km
        """
        return self.trans.inverse_trans(points)

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def fault2utm(self, points):
        """Fault coordinate system to UTM.

        Args:
            - points: Point lists in fault coordinates, m x 3 list/array

        Returns:
            - Point list in UTM coordinates, m x 3 list/array

        Coordinate system:
        Input: Fault (X=along strike, Y=opposite dip, Z=normal) in km
        Output: UTM (X=easting, Y=northing, Z=zenith) in km
        """
        return self.trans.forwars_trans(points)

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def discretize_planar(self, max_edge_length, refinement=None):
        """Triangulate planar fault using grid-based triangulation.

        Creates a triangular mesh on a planar fault surface by first creating
        a rectangular grid and then triangulating each rectangle into 2 triangles.

        Args:
            - max_edge_length: Maximum triangle edge length (km)
            - refinement: Optional refinement function for edge size variation.
                          Not implemented in current version.

        Returns:
            - List of triangles: [[(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)], ...]
              Each triangle is a list of 3 vertices in UTM coordinates (km).

        Note:
            Triangles are returned in counter-clockwise order when viewed from
            positive Z direction (normal to fault plane), which is
            standard orientation for finite element methods.
        """
        # Calculate grid resolution based on max_edge_length
        x_num = math.ceil(self.length / max_edge_length)
        y_num = math.ceil(self.width / max_edge_length)

        # Create grid points in fault coordinates
        x = np.linspace(-self.length / 2, self.length / 2, x_num + 1)
        y = np.linspace(-self.width, 0, y_num + 1)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((X.shape[0], X.shape[1]))

        # Triangulate each rectangle into 2 triangles
        triangles = []
        for i in range(y_num):
            for j in range(x_num):
                x1, y1, z1 = X[i, j], Y[i, j], Z[i, j]
                x2, y2, z2 = X[i, j + 1], Y[i, j + 1], Z[i, j + 1]
                x3, y3, z3 = X[i + 1, j + 1], Y[i + 1, j + 1], Z[i + 1, j + 1]
                x4, y4, z4 = X[i + 1, j], Y[i + 1, j], Z[i + 1, j]

                tri1 = [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]
                tri2 = [(x1, y1, z1), (x3, y3, z3), (x4, y4, z4)]

                tri1_utm = [(float(v[0]), float(v[1]), float(v[2])) for v in self.fault2utm(tri1)]
                tri2_utm = [(float(v[0]), float(v[1]), float(v[2])) for v in self.fault2utm(tri2)]

                triangles.append(tri1_utm)
                triangles.append(tri2_utm)

        print(f"Created {len(triangles)} triangular patches (planar fault).")
        return triangles


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
if __name__ == "__main__":
    # Example: Planar fault triangulation
    print("=" * 60)
    print("Example 1: Planar fault triangulation")
    print("=" * 60)

    fault = TriPatch({"upper center": (444444, 555555, -4)}, strike=10, dip=30, length=80, width=50)
    triangles = fault.discretize_planar(max_edge_length=5.0)
    print(f"Number of triangles: {len(triangles)}")
