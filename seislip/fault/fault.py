#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Planar fault class for single or multiple faults.

Created on 31.12.23

@author: Zelong Guo
"""

__author__ = "Zelong Guo"

import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import math

# seislip libs
if __name__ == "__main__":
    sys.path.append("../")
    from seislip.crs import GeoTrans
    from seislip.fault.rectpatch import RectPatch
    from seislip.fault.tripatch import TriPatch
else:
    from ..crs import GeoTrans
    from .rectpatch import RectPatch
    from .tripatch import TriPatch


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
class Fault(GeoTrans):
    """Constructing a Fault object with four corner coordinates (and central/centroid point coordinates).

    The object of this class would be used for meshing the fault plane into rectangle or triangle patches.

    Args:
        - name:                 Fault instance name
        - lon0:                 longitude used for specifying the utm zone
        - lat0:                 lattitude used for specifying the utm zone
        - ellps:                Optional, reference ellipsoid, defatult = "WGS84"
        - utmzone:              Optional, if not specify lon0, lat0 and ellps, default = None.

    Return:
        - None.
    """
    def __init__(self, name, lon0, lat0, ellps="WGS84", utmzone=None):
        super().__init__(name, lon0, lat0, ellps, utmzone)

        # fault parameters
        self.ucp = None   # UTM coordinates of central point on upper fault edge
        self.strike = None
        self.dip = None
        self.length = None
        self.width = None
        self.patch_verts = None

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

    def initialize_fault(self, pointpos: str, lon, lat, verdepth, strike, dip, length, width):
        """Initialize a fault plane by specifying a corner or central point of the fault.

        The definition of the fault coordinate system:
        This coordinate system is right-hand:
        X:      along strike
        Y:      along opposite direction of dip
        Z:      normal direction the fault
        Point position on fault plane.
        "uo": upper original point,     "uc": upper center point,       "ue": upper end point
        "bo": bottom original point,    "bc": bottom center point,      "be": bottom end point
        "cc": centroid center point

        The definition of UTM system:
        X:      easting, km
        Y:      northing, km
        Z:      zenith direction, km

        Args:
            - pointpos:       the position of the specified fault point in the fault coordinate system:
                              "uo", "uc", "ue", "bo", "bc", "be", "cc".
            - lon:            longitude of the specified point, degree
            - lat:            latitude of the specified point, degree
            - verdepth:       vertical depth of the specified point, km. Depth should
                              specified as Negative value.
            - strike:         strike angle of the fault, degree
            - dip:            dip angle of the fault, degree
            - width:          width along the fault dip direction, km
            - length:         length along the fault strike direction, km

        Return:
            - None.
        """

        pointpos_list = ("uo",  "UO",  "upper origin",    "upper_origin",
                         "uc",  "UC",  "upper center",    "upper_center",
                         "ue",  "UE",  "upper end",       "upper_end",
                         "bo",  "BO",  "bottom origin",   "bottom_origin",
                         "bc",  "BC",  "bottom center",   "bottom_center",
                         "be",  "BE",  "bottom end",      "bottom_end",
                         "cc",  "CC",  "centroid center", "centroid_center")

        if pointpos not in pointpos_list:
            raise ValueError("Please specify a right point position!")
        else:
            # calculate fault corner coordinates
            fault_corners, _, _ = self.__get_corner_vertices(pointpos, {"LL": (lon, lat, verdepth)}, strike, dip, length, width)
            # self.ccp = ccp
            self.strike = strike
            self.dip = dip
            self.length = length
            # check if the fault expose to surface, and return the upper center point
            utm_xyz_uc, fault_verts, width = self.__check_breach_surface(fault_corners, strike, dip, length, width)
            self.ucp = {"upper center": utm_xyz_uc}
            self.patch_verts = fault_verts  # fault/patches vertices
            self.width = width


    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def extend_to_surface(self):
        x_uc, y_uc, z_uc = self.ucp["upper center"][0], self.ucp["upper center"][1], self.ucp["upper center"][2]

        if z_uc >= 0:
            warnings.warn("The fault has already reached to the surface!")
        else:
            # complementary angle of strike
            strike_comp = np.radians(90 - self.strike)
            wid_required_to_surface = -z_uc / np.tan(np.radians(self.dip))
            cpl_wid_required = (0 - wid_required_to_surface * 1j) * np.exp(strike_comp * 1j)

            # update the fault width
            new_width = -z_uc / np.sin(np.radians(self.dip)) + self.width
            self.width = new_width

            surface_xy_uc = (x_uc + y_uc * 1j) - cpl_wid_required
            surface_utmx_uc, surface_utmy_uc, surface_utmz_uc = surface_xy_uc.real, surface_xy_uc.imag, 0
            _, fault_verts, _ = self.__get_corner_vertices("upper center", {"UTM": (surface_utmx_uc, surface_utmy_uc, surface_utmz_uc)},
                                                        self.strike, self.dip, self.length, self.width)

            self.ucp = {"upper center": (surface_utmx_uc, surface_utmy_uc, surface_utmz_uc)}
            self.patch_verts = fault_verts
            print(f"Now the fault {self.name} has been extended to the surface! The width of it is {self.width}!")
            print("+-" * 50)

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def construct_rect_patches(self, sublength, subwidth, str_vary_fct=1.0, dip_vary_fct=1.0, verbose=False):
        """Construct uniform or strike- and/or dip-varying rectangle patches.

        Args:
            - sublength:                   patch length on upper fault edge, km
            - subwidth:                    patch width on upper fault edge, km
            - str_vary_fct:                strike varying factor of patch length along strike direction,
                                           should >= 1
            - dip_vary_fct:                dip varying factor of patch width along dip direction
                                           should >= 1

        Return:
            - patch:                        a list of discretizing fault patches
        """

        # create a RectPatch object
        rectangle_patches = RectPatch(self.ucp, self.strike, self.dip, self.length, self.width)
        if str_vary_fct == 1 and dip_vary_fct == 1:
            self.patch_verts = rectangle_patches.discretize(sublength, subwidth)
        else:
            self.patch_verts, self.width = rectangle_patches.discretize_depth_varying(sublength, subwidth, str_vary_fct,
                                                                          dip_vary_fct, verbose)

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def construct_tri_patches(self, max_edge_length, refinement=None):
        """Discretize fault into triangular patches using Delaunay triangulation.

        Uses TriPatch for planar fault triangulation. Triangles are created on the
        fault plane and returned as lists of 3 vertices in UTM coordinates.

        Args:
            - max_edge_length: Maximum triangle edge length (km)
            - refinement: Optional refinement function for variable edge sizes.
                         Function signature: (x, y, z) -> float, where
                         x, y, z are fault coordinates and return is
                         local edge length multiplier (default 1.0).

        Return:
            - Number of triangular patches created

        Note:
            This method stores triangles in self.tri_patch_verts, where each triangle
            is a list of 3 vertices: [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)].
            All coordinates are in UTM (km).
        """
        # Create TriPatch object
        tri_patches = TriPatch(self.ucp, self.strike, self.dip, self.length, self.width)
        self.tri_patch_verts = tri_patches.discretize_planar(max_edge_length, refinement)
        return len(self.tri_patch_verts)

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def __get_corner_vertices(self, pointpos: str, coords: dict, strike, dip, length, width):
        """Calculate the 3D UTM coordinates of corners and centroid points on fault, based on one point given.

        Args:
            - pointpos:       the position of the specified fault point in the fault coordinate system:
                              "uo", "uc", "ue", "bo", "bc", "be", "cc".
            - coords:         a dict of point coordinates, either lonlat or utm system
                              e.g., {"UTM", (343434, 5454544, -3)}, {"LL", (45.3, 55.4, -5)}
                              vertical depth of the specified point, km. Depth should specified as Negative value.
            - strike:         strike angle of the fault, degree
            - dip:            dip angle of the fault, degree
            - width:          width along the fault dip direction, km
            - length:         length along the fault strike direction, km

        Return:
            - fault_corners:      a list of fault mian points coordinates
            - fault_verts:        a list of fault 4 corners coordinates
            - fault centroid      the centroid position of the fault
        """
        if "UTM" in coords:
            x, y, verdepth = coords["UTM"][0], coords["UTM"][1], coords["UTM"][2]
        elif "utm" in coords:
            x, y, verdepth = coords["utm"][0], coords["utm"][1], coords["utm"][2]
        elif "LL" in coords:
            lon, lat, verdepth = coords["LL"][0], coords["LL"][1], coords["LL"][2]
            x, y = self.ll2xy(lon, lat)
        elif "ll" in coords:
            lon, lat, verdepth = coords["ll"][0], coords["ll"][1], coords["ll"][2]
            x, y = self.ll2xy(lon, lat)
        else:
            raise ValueError("Invalid coordinate format!")

        # complementary angle of strike
        strike_comp = np.radians(90 - strike)
        dip = np.radians(dip)
        projwidth = width * np.cos(dip)

        cpl_length = (length + 0j) * np.exp(strike_comp * 1j)
        cpl_width = (0 - projwidth * 1j) * np.exp(strike_comp * 1j)

        # if pointpos in ("ul", "UL", "upper left", "upper_left"):
        if pointpos in ("uo", "UO", "upper origin", "upper_origin"):
            x_uo, y_uo, z_uo = x, y, verdepth
        elif pointpos in ("uc", "UC", "upper center", "upper_center"):
            x_uc, y_uc, z_uc = x, y, verdepth
            xy_uo = (x_uc + y_uc * 1j) - cpl_length / 2
            x_uo, y_uo, z_uo = xy_uo.real, xy_uo.imag, verdepth
        elif pointpos in ("ue", "UE", "upper end", "upper_end"):
            x_ue, y_ue, z_ue = x, y, verdepth
            xy_uo = (x_ue + y_ue * 1j) - cpl_length
            x_uo, y_uo, z_uo = xy_uo.real, xy_uo.imag, verdepth
        elif pointpos in ("be", "BE", "bottom end", "bottom_end"):
            x_be, y_be, z_be = x, y, verdepth
            xy_uo = (x_be + y_be * 1j) - cpl_length - cpl_width
            x_uo, y_uo, z_uo = xy_uo.real, xy_uo.imag, verdepth + width * np.sin(dip)
        elif pointpos in ("bc", "BC", "bottom center", "bottom_center"):
            x_bc, y_bc, z_bc = x, y, verdepth
            xy_uo = (x_bc + y_bc * 1j) - cpl_length / 2 - cpl_width
            x_uo, y_uo, z_uo = xy_uo.real, xy_uo.imag, verdepth + width * np.sin(dip)
        elif pointpos in ("bo", "BO", "bottom origin", "bottom_origin"):
            x_bo, y_bo, z_bo = x, y, verdepth
            xy_uo = (x_bo + y_bo * 1j) - cpl_width
            x_uo, y_uo, z_uo = xy_uo.real, xy_uo.imag, verdepth + width * np.sin(dip)
        elif pointpos in ("cc", "CC", "centroid center", "centroid_center"):
            x_cc, y_cc, z_cc = x, y, verdepth
            xy_uo = (x_cc + y_cc * 1j) - cpl_width / 2 - cpl_length / 2
            x_uo, y_uo, z_uo = xy_uo.real, xy_uo.imag, verdepth + width * np.sin(dip) / 2
        else:
            raise ValueError("Please specify a correct point position VALUE!")

        # upper center point
        xy_uc = (x_uo + y_uo * 1j) + cpl_length / 2
        x_uc, y_uc, z_uc = xy_uc.real, xy_uc.imag, z_uo
        # upper right point
        xy_ue = (x_uo + y_uo * 1j) + cpl_length
        x_ue, y_ue, z_ue = xy_ue.real, xy_ue.imag, z_uo
        # bottom right point
        xy_be = (x_ue + y_ue * 1j) + cpl_width
        x_be, y_be, z_be = xy_be.real, xy_be.imag, z_uo - width * np.sin(dip)
        # bottom center point
        xy_bc = (x_be + y_be * 1j) - cpl_length / 2
        x_bc, y_bc, z_bc = xy_bc.real, xy_bc.imag, z_uo - width * np.sin(dip)
        # bottom left point
        xy_bo = (x_bc + y_bc * 1j) - cpl_length / 2
        x_bo, y_bo, z_bo = xy_bo.real, xy_bo.imag, z_uo - width * np.sin(dip)
        # fault center point
        x_cc, y_cc, z_cc = (x_uc + x_bc) / 2, (y_uc + y_bc) / 2, (z_uc + z_bc) / 2

        fault_corners = {
            "upper origin":       (x_uo, y_uo, z_uo),
            "upper center":     (x_uc, y_uc, z_uc),
            "upper end":      (x_ue, y_ue, z_ue),
            "bottom origin":      (x_bo, y_bo, z_bo),
            "bottom center":    (x_bc, y_bc, z_bc),
            "bottom end":     (x_be, y_be, z_be),
            "centroid center":  (x_cc, y_cc, z_cc),
        }

        fault_verts = [
            [(x_uo, y_uo, z_uo), (x_ue, y_ue, z_ue), (x_be, y_be, z_be), (x_bo, y_bo, z_bo)]
        ]
        fault_centroid = (x_cc, y_cc, z_cc)

        return fault_corners, fault_verts, fault_centroid

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def __check_breach_surface(self, fault_corners, strike, dip, length, width):
        """Check the fault breach to the surface or not. If it does, then assign the depth of upper edge
        to 0 (surface) automatically. If it does not, do nothing. Finally, we return the UTM/Cartesian
         coordinates of the upper center point of the fault and for corner coordinates.

        Args:
            - fault_corners:            a dict containing fault corners coordinates
            - strike
            - dip
            - length
            - width

        Return:
            - UTM XYZ coordinates of center point on the fault upper edge
        """

        # uo = fault_corners["upper origin"]
        uc = fault_corners["upper center"]
        # ue = fault_corners["upper end"]

        x_uc, y_uc, z_uc = uc[0], uc[1], uc[2]
        # if the upper fault edge breaches to the surface:
        if z_uc > 0:
            warnings.warn("Warning: The fault has breached to the surface! "
                          "We are now setting the upper edge depth to 0.")
            # complementary angle of strike
            strike_comp = np.radians(90 - strike)
            exposed_surface_wid = z_uc / np.tan(np.radians(dip))
            cpl_exposed_surface_wid = (0 - exposed_surface_wid * 1j) * np.exp(strike_comp * 1j)

            surface_xy_uc = (x_uc + y_uc * 1j) + cpl_exposed_surface_wid
            surface_utmx_uc, surface_utmy_uc, surface_utmz_uc = surface_xy_uc.real, surface_xy_uc.imag, 0
            width = width - z_uc / np.sin(np.radians(dip))
            _, fault_verts, _ = self.__get_corner_vertices("upper center", {"UTM": (surface_utmx_uc, surface_utmy_uc, surface_utmz_uc)},
                                                        strike, dip, length, width)

            return (surface_utmx_uc, surface_utmy_uc, surface_utmz_uc), fault_verts, width

        else:
            _, fault_verts, _ = self.__get_corner_vertices("upper center", {"UTM": (x_uc, y_uc, z_uc)}, strike, dip, length, width)
            print('+-' * 50)
            print("The fault does not reach to the surface yet!")
            print('+-' * 50)
            return (x_uc, y_uc, z_uc), fault_verts, width



    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

    def read_from_trace(self):
        pass


    def plot(self, verts):
        # x, y, z = np.array(x), np.array(y), np.array(z)
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(x, y, z)
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # ax.set_zlim(z.min(), 0)
        #
        # plt.show()

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        poly3d = Poly3DCollection(verts, edgecolor='black', alpha=0.7)

        ax.add_collection3d(poly3d)

        verts = np.array(verts)
        xmin = verts[:, :, 0].min()
        xmax = verts[:, :, 0].max()
        ymin = verts[:, :, 1].min()
        ymax = verts[:, :, 1].max()
        zmin = verts[:, :, 2].min()
        zmax = verts[:, :, 2].max()


        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, max(0, zmax) + 0.0001)

        plt.show()


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
if __name__ == "__main__":

    fault = Fault("fault1", 44.28, 35.47)
    # patch1, patch_corner1 = fault.initialize_planar_fault(lon_uc=44.344, lat_uc=35.603, verdepth_uc=3, strike=10, dip=45, length=80, width=50)
    fault.initialize_fault(pointpos="upper center", lon=44.344, lat=35.603, verdepth=0, strike=50, dip=45, length=180, width=30)
    # fault.plot(fault.patch_verts)

    # fault.extend_to_surface()
    fault.construct_rect_patches(sublength=8, subwidth=6, str_vary_fct=2, dip_vary_fct=1., verbose=False)
    fault.plot(fault.patch_verts)


    # # --------------------------------------------
    # fault = Fault("fault2", 44.28, 35.47)
    # fault.initialize_fault(pointpos="upper center", lon=44.28, lat=35.47, verdepth=-5, strike=280, dip=70, length=10, width=10)
    # fault.plot(fault.patch_verts)

