#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rectangle Fault Patch Construction.

Created on 15.01.24

@author: Zelong Guo
"""
__author__ = "Zelong Guo"

# standard libs
import sys
from typing import List

import numpy as np
import math
import warnings


if __name__ == "__main__":
    sys.path.append("../")
    # from seislip.fault.fault import Fault
    from seislip.utils.transformation import Transformation
    from seislip.crs import CoordinateTransformer
else:
    # from .fault import Fault
    from ..utils.transformation import Transformation
    from ..crs import CoordinateTransformer


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
class RectPatch(object):
    """Planar/rectangle fault grid generation with rectangle patches.

    """
    def __init__(self, upc, strike, dip, length, width):
        """Initialize the fault parameters and construct the transformation matrix between UTM
        and fault coordinate systems."""
        # fault parameters
        self.upc = upc
        self.strike = strike
        self.dip = dip
        self.length = length
        self.width = width

        # constructing transformation between UTM and fault coordinate systems
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
            - points:           point lists in UTM ccordinates, m x 3 list/array
        Return:
            - point list in fault coodinates, m x 3 list/array
        """
        return self.trans.inverse_trans(points)

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

    def fault2utm(self, points):
        """fault coordinate system to UTM."""
        return self.trans.forwars_trans(points)

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def discretize(self, patch_length, patch_width):
        """Discretize the fault into rectangle patches uniformly.

        Args:
            - patch_length:         patch length, km
            - patch_width:          patch width, km

        Return:
            - patch:                a list of discretizing fault patches
        """
        x_num = math.ceil(self.length / patch_length)
        y_num = math.ceil(self.width / patch_width)
        patch = []

        x = np.linspace(-self.length / 2, self.length / 2, x_num + 1)
        y = np.linspace(-self.width, 0, y_num + 1)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((X.shape[0], X.shape[1]))

        for i in range(y_num):
            for j in range(x_num):
                x1, y1, z1 = X[i, j], Y[i, j], Z[i, j]
                x2, y2, z2 = X[i, j + 1], Y[i, j + 1], Z[i, j + 1]
                x3, y3, z3 = X[i + 1, j + 1], Y[i + 1, j + 1], Z[i + 1, j + 1]
                x4, y4, z4 = X[i + 1, j], Y[i + 1, j], Z[i + 1, j]
                rectangle = [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
                rectangle = self.fault2utm(rectangle)
                patch.append(rectangle)
        return patch

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def discretize_depth_varying(self, upper_patch_len, upper_patch_wid, str_vary_fct, dip_vary_fct, verbose=True):
        """Discretize the fault into strike- or dip-varying rectangle patches.

        Args:
            - upper_patch_len:             patch length on upper fault edge, km
            - upper_patch_wid:             patch width on upper fault edge, km
            - str_vary_fct:                strike varying factor of patch length along strike direction,
                                           should >= 1
            - dip_vary_fct:                dip varying factor of patch width along dip direction
                                           should >= 1

        Return:
            - patch:                        a list of discretizing fault patches
        """
        patch = []
        current_width, y_coords = self._get_segments_size_varying(self.width, upper_patch_wid, dip_vary_fct, verbose)
        patch_length = upper_patch_len
        for y_coord in y_coords:
            x_num = math.ceil(self.length / patch_length)
            x = np.linspace(-self.length / 2, self.length / 2, x_num + 1)
            y = np.array(y_coord)
            X, Y = np.meshgrid(x, y)
            for i in range(len(y_coord) - 1):
                for j in range(x_num):
                    x1, y1, z1 = X[i, j], Y[i, j], 0
                    x2, y2, z2 = X[i, j + 1], Y[i, j + 1], 0
                    x3, y3, z3 = X[i + 1, j + 1], Y[i + 1, j + 1], 0
                    x4, y4, z4 = X[i + 1, j], Y[i + 1, j], 0
                    rectangle = [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
                    rectangle = self.fault2utm(rectangle)
                    patch.append(rectangle)

            patch_length *= str_vary_fct

        return patch, current_width

    def _get_segments_size_varying(self, width, subwidth, ratio, verbose=True):
        """Calculate the segments number and width,
        based on the total subwidth and the changing ratio of subwidth."""
        if ratio < 1:
            raise ValueError("Infinite segments would be created!")
        current_wid = 0
        count = 0
        segment_coord = []
        while current_wid < width:
            # if current_wid + subwidth >= width:
            #     segment_coord.append((current_wid, width))
            # else:
            #     segment_coord.append((current_wid, current_wid + subwidth))
            segment_coord.append((current_wid, current_wid + subwidth))
            current_wid += subwidth
            subwidth *= ratio
            count += 1
            if verbose:
                print(f"Segment {count}, coordinates: {segment_coord[count - 1]}")

        print(f"Now the fault width is readjusted to {current_wid}!!!")
        # convert to fault coordinate system, negative values
        segment_coord = [(-x, -y) for (x, y) in segment_coord]
        return current_wid, segment_coord




# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
#
# class RectPatch1(Fault):
#     """Planar/rectangle fault grid generation with rectangle patches.
#
#     """
#     def __init__(self, name, lon0, lat0, ellps="WGS84", utmzone=None):
#         super().__init__(name, lon0, lat0, ellps, utmzone)
#
#         # fault parameters
#         self.origin = None   # the origin point you specified
#         self.strike = None
#         self.dip = None
#         self.length = None
#         self.width = None
#
#         self.trans = None
#
#
#     # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
#
#     def initialize_rectangle_patch(self, pointpos: str, lon, lat, verdepth, strike, dip, length, width):
#         """Initialize the fault parameters of the whole fault need to be meshed, and construct the
#         transformation matrix between Cartesian/UTM and local fault coordinate system.
#
#         Args:
#             - pointpos:         point position you specified, "uo", "uc", "ue" ...
#             - lon:            longitude of the specified point, degree
#             - lat:            latitude of the specified point, degree
#             - verdepth:       vertical depth of the specified point, km. Depth should
#                               specified as Negative value.
#             - strike:         strike angle of the fault, degree
#             - dip:            dip angle of the fault, degree
#             - width:          width along the fault dip direction, km
#             - length:         length along the fault strike direction, km
#
#         Return:
#             - None.
#          """
#
#         # initialize the fault parameters
#         self.initialize_fault(pointpos, lon, lat, verdepth, strike, dip, length, width)
#
#         # origin to UTM
#         utm_x, utm_y = self.ll2xy(self.origin[1][0], self.origin[1][1])
#         utm_z = self.origin[1][2]
#
#         # constructing transformation between UTM and fault coordinate systems
#         trans = Transformation()
#         trans.rotation_x(np.radians(self.dip))
#         trans.rotation_z(np.radians(90-self.strike))
#         trans.translation(T=(utm_x, utm_y, utm_z))
#         trans.inverse()
#         self.trans = trans
#
#         # if not hasattr(self.fault, "origin") or getattr(self.fault, "origin") is None:
#         #     raise AttributeError(f"The fault object {fault.name} does not specify attribute 'origin' yet!")
#
#     # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
#
#     def utm2fault(self, points):
#         """UTM to fault coordinate system.
#
#         Args:
#             - points:           point lists in UTM ccordinates, m x 3 list/array
#         Return:
#             - point list in fault coodinates, m x 3 list/array
#         """
#         return self.trans.inverse_trans(points)
#
#     # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
#
#     def fault2utm(self, points):
#         """fault coordinate system to UTM."""
#         return self.trans.forwars_trans(points)
#
#     # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
#
#     def discretize(self, patch_length, patch_width):
#         """Discretize the fault into rectangle patches."""
#         x_num = math.ceil(self.length / patch_length)
#         y_num = math.ceil(self.width / patch_width)
#         patch = []
#
#         if self.origin[0] in ("uo", "UO", "upper origin", "upper_origin"):
#             x = np.linspace(0, self.length, x_num + 1)
#             y = np.linspace(-self.width, 0, y_num + 1)
#         elif self.origin[0] in ("uc", "UC", "upper center", "upper_center"):
#             x = np.linspace(-self.length / 2, self.length / 2, x_num + 1)
#             y = np.linspace(-self.width, 0, y_num + 1)
#         elif self.origin[0] in ("ue",  "UE",  "upper end",       "upper_end"):
#             x = np.linspace(-self.length, 0, x_num + 1)
#             y = np.linspace(-self.width, 0, y_num + 1)
#         elif self.origin[0] in ("bo",  "BO",  "bottom origin",   "bottom_origin"):
#             x = np.linspace(0, self.length, x_num + 1)
#             y = np.linspace(0, self.width, y_num + 1)
#         elif self.origin[0] in ("bc",  "BC",  "bottom center",   "bottom_center"):
#             x = np.linspace(-self.length / 2, self.length / 2, x_num + 1)
#             y = np.linspace(0, self.width, y_num + 1)
#         elif self.origin[0] in ("be",  "BE",  "bottom end",      "bottom_end"):
#             x = np.linspace(-self.length, 0, x_num + 1)
#             y = np.linspace(0, self.width, y_num + 1)
#         elif self.origin[0] in ("cc",  "CC",  "centroid center", "centroid_center"):
#             x = np.linspace(-self.length / 2, self.length / 2, x_num + 1)
#             y = np.linspace(-self.width / 2, self.width / 2, y_num + 1)
#         else:
#             raise ValueError(f"Unknown {self.origin[0]}! Please specify a correct parameter!")
#
#         X, Y = np.meshgrid(x, y)
#         Z = np.zeros((X.shape[0], X.shape[1]))
#
#         for i in range(y_num):
#             for j in range(x_num):
#                 x1, y1, z1 = X[i, j],       Y[i, j],        Z[i, j]
#                 x2, y2, z2 = X[i, j+1],     Y[i, j+1],      Z[i, j+1]
#                 x3, y3, z3 = X[i+1, j+1],   Y[i+1, j+1],    Z[i+1, j+1]
#                 x4, y4, z4 = X[i+1, j],     Y[i+1, j],      Z[i+1, j]
#                 rectangle = [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
#                 rectangle = self.fault2utm(rectangle)
#                 # check if breach to the surface
#                 if any(z > 0 for z in rectangle[:, -1]):
#                     warnings.warn("Warning: The Fault Has Breached To The Surface!")
#                 else:
#                     patch.append(rectangle)
#
#         return patch


# -----------------------------------------------------------------------------------------
if __name__ == "__main__":
    fault = RectPatch({"upper center": (444444, 555555, -4)}, strike=10, dip=30, length=80, width=50)
    y_count, y_coords = fault._get_segments_size_varying(fault.width, 2, ratio=1.5)
    # x_count, x_coords = fault.get_segments(fault.length, 3, ratio=2)
    # pass






