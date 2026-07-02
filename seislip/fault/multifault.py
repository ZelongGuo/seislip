#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiFault class for curved or chained fault surfaces.

Created on 16.01.2026

@author: Zelong Guo
"""

__author__ = "Zelong Guo"

import sys
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# seislip libs
if __name__ == "__main__":
    sys.path.append("../")
    from seislip.crs import CoordinateTransformer, init_coordinate_transformer_owner, transformer_property
    from seislip.fault.fault import Fault
    from seislip.fault.tripatch import TriPatch
else:
    from ..crs import CoordinateTransformer, init_coordinate_transformer_owner, transformer_property
    from .fault import Fault
    from .tripatch import TriPatch

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
class MultiFault:
    """Container for multiple fault segments or curved fault surfaces.

    This class handles:
    - Curved fault traces (continuous geometry)
    - Chained fault segments (multiple Fault objects)
    - Triangular meshing (no gaps/overlaps)
    - Batch operations on all segments

    Args:
        - name:                 Fault instance name
        - transformer:          CoordinateTransformer instance defining the shared CRS

    Return:
        - None.
    """
    # CRS attributes are stored on ``self.transformer``. These properties let
    # existing code keep using ``obj.utmzone`` / ``obj.lon0`` directly while
    # avoiding duplicated CRS state on the domain object.
    lon0 = transformer_property("lon0")
    lat0 = transformer_property("lat0")
    ellps = transformer_property("ellps")
    wgs = transformer_property("wgs")
    utm = transformer_property("utm")
    proj2utm = transformer_property("proj2utm")
    proj2wgs = transformer_property("proj2wgs")
    utmzone = transformer_property("utmzone")

    def ll2xy(self, lon, lat):
        """Forward lon/lat conversion to the owned coordinate transformer."""
        return self.transformer.ll2xy(lon, lat)

    def xy2ll(self, x, y):
        """Forward UTM-to-lon/lat conversion to the owned coordinate transformer."""
        return self.transformer.xy2ll(x, y)

    def __init__(self, name, transformer: CoordinateTransformer = None):
        init_coordinate_transformer_owner(self, name, transformer)

        # fault segments (list of Fault objects)
        self.segments = []

        # Curved trace data (for triangular meshing)
        self.trace_points = None   # List of (x, y, z) UTM coordinates
        self.trace_dips = None     # Dip at each trace point
        self.trace_widths = None   # Width at each trace point
        self.surface_type = None   # "curved" or "chained"

        # Triangular patches (from gmsh)
        self.tri_patch_verts = None

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def from_trace_triangles(self, trace_points, dips=None, widths=None, coord_type="LL"):
        """Create curved fault from a trace of surface points for triangular meshing.

        This is useful for:
        - Modeling curved or bent fault geometries
        - Using field observations (fault trace + dip measurements)
        - Creating continuous fault surfaces without gaps/overlaps using triangular meshing

        Args:
            - trace_points:   List of (lon, lat, depth) or (x, y, depth) points
                                  defining fault trace (surface exposure)
                                  depth should be negative (km)
            - dips:          Optional, list of dip angles (degrees) at each trace point.
                                  If None, uses 45.0 for all points.
            - widths:        Optional, list of fault widths (km) at each trace point.
                                  If None, uses 10.0 for all points.
            - coord_type:     "LL" for lonlat coordinates (default), "UTM" for UTM

        Return:
            - None.
        """
        if coord_type not in ("LL", "UTM", "ll", "utm"):
            raise ValueError("coord_type must be 'LL' or 'UTM'")

        n_points = len(trace_points)
        if n_points < 2:
            raise ValueError("trace_points must have at least 2 points")

        # Set defaults
        if dips is None:
            dips = [45.0] * n_points
        if widths is None:
            widths = [10.0] * n_points

        if len(dips) != n_points:
            raise ValueError(f"Length of dips ({len(dips)}) must match trace_points ({n_points})")
        if len(widths) != n_points:
            raise ValueError(f"Length of widths ({len(widths)}) must match trace_points ({n_points})")

        # Convert to UTM if needed
        trace_utm = []
        for point in trace_points:
            if coord_type.upper() == "LL":
                lon, lat, depth = point
                x, y = self.transformer.ll2xy(lon, lat)
                trace_utm.append((x, y, depth))
            else:
                trace_utm.append(point)

        # Store trace data
        self.trace_points = trace_utm
        self.trace_dips = dips
        self.trace_widths = widths
        self.surface_type = "curved"

        print(f"MultiFault '{self.name}' created from trace for triangular meshing ({n_points} points):")

        # Calculate statistics
        all_x = [p[0] for p in trace_utm]
        all_y = [p[1] for p in trace_utm]
        all_z = [p[2] for p in trace_utm]

        delta_x = all_x[-1] - all_x[0]
        delta_y = all_y[-1] - all_y[0]
        overall_strike_rad = np.arctan2(delta_x, delta_y)
        overall_strike = np.degrees(overall_strike_rad)
        if overall_strike < 0:
            overall_strike += 360

        total_length = np.sqrt((all_x[-1] - all_x[0])**2 + (all_y[-1] - all_y[0])**2)

        print(f"  Overall strike: {overall_strike:.2f} deg")
        print(f"  Total length: {total_length:.3f} km")
        print(f"  Dip range: {min(dips):.1f} - {max(dips):.1f} deg")
        print(f"  Width range: {min(widths):.2f} - {max(widths):.2f} km")
        print(f"  Depth range: {min(all_z):.2f} - {max(all_z):.2f} km")
        print("+-" * 50)

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def from_trace_rectangles(self, trace_points, dips=None, widths=None, coord_type="LL"):
        """Create chained fault segments from trace points for rectangular patch discretization.

        This method automatically connects consecutive points to form fault segments.
        For n trace points, n-1 segments are created (point[i] to point[i+1]).

        This is useful for:
        - Building multi-segment faults with variable dip
        - Chaining segments where ue of one = uo of next
        - Creating fault surfaces that will be discretized into rectangular patches

        Args:
            - trace_points:   List of (lon, lat, depth) or (x, y, depth) points
                                  defining fault trace. Consecutive points are connected
                                  to form segments.
                                  depth should be negative (km)
            - dips:          Optional, list of dip angles (degrees) for each segment.
                                  If None, uses 45.0 for all segments.
                                  Must have len(dips) == len(trace_points) - 1.
            - widths:        Optional, list of fault widths (km) for each segment.
                                  If None, uses 10.0 for all segments.
                                  Must have len(widths) == len(trace_points) - 1.
            - coord_type:     "LL" for lonlat coordinates (default), "UTM" for UTM

        Return:
            - None.
        """
        n_points = len(trace_points)
        if n_points < 2:
            raise ValueError("trace_points must have at least 2 points")

        n_segments = n_points - 1

        # Set defaults for dips and widths
        if dips is None:
            dips = [45.0] * n_segments
        if widths is None:
            widths = [10.0] * n_segments

        if len(dips) != n_segments:
            raise ValueError(f"Length of dips ({len(dips)}) must equal n_segments ({n_segments})")
        if len(widths) != n_segments:
            raise ValueError(f"Length of widths ({len(widths)}) must equal n_segments ({n_segments})")

        self.segments = []
        self.surface_type = "chained"

        print(f"MultiFault '{self.name}' creating {n_segments} segments from {n_points} trace points for rectangular patch discretization:")
        print("=" * 60)

        for i in range(n_segments):
            # Get upper origin and upper end from consecutive trace points
            uo = trace_points[i]
            ue = trace_points[i + 1]
            dip = dips[i]
            width = widths[i]

            # Convert points to UTM if needed
            if coord_type.upper() == "LL":
                uo_lon, uo_lat, uo_depth = uo
                ue_lon, ue_lat, ue_depth = ue
                uo_x, uo_y = self.transformer.ll2xy(uo_lon, uo_lat)
                ue_x, ue_y = self.transformer.ll2xy(ue_lon, ue_lat)
            else:
                uo_x, uo_y, uo_depth = uo
                ue_x, ue_y, ue_depth = ue
                # For UTM, we need lon/lat for initialize_fault, so convert back
                uo_lon, uo_lat = self.transformer.xy2ll(uo_x, uo_y)

            # Calculate strike (angle from uo to ue, measured from north clockwise)
            delta_x = ue_x - uo_x
            delta_y = ue_y - uo_y
            strike_rad = np.arctan2(delta_x, delta_y)
            strike = np.degrees(strike_rad)
            if strike < 0:
                strike += 360

            # Calculate length (distance between uo and ue)
            length = np.sqrt(delta_x**2 + delta_y**2)

            # Create Fault object for this segment using the same CRS transformer.
            seg = Fault(f"{self.name}_seg{i+1}", transformer=self.transformer)
            seg.initialize_fault(
                pointpos="uo",
                lon=uo_lon,
                lat=uo_lat,
                verdepth=uo_depth,
                strike=strike,
                dip=dip,
                length=length,
                width=width
            )
            self.segments.append(seg)

        print(f"Created {len(self.segments)} fault segments total.")
        print("+-" * 50)

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def discretize_all_rectangles(self, sublength, subwidth, str_vary_fct=1.0, dip_vary_fct=1.0, verbose=False):
        """Discretize all fault segments with rectangular patches.

        Args:
            - sublength:                   Patch length on upper fault edge (km)
            - subwidth:                    Patch width on upper fault edge (km)
            - str_vary_fct:                Strike varying factor (default 1.0)
            - dip_vary_fct:                Dip varying factor (default 1.0)
            - verbose:                     Print patch details (default False)

        Return:
            - Total number of patches across all segments
        """
        if not self.segments:
            warnings.warn("No fault segments to discretize!")
            return 0

        total_patches = 0
        for seg in self.segments:
            seg.construct_rect_patches(sublength, subwidth,
                                     str_vary_fct=str_vary_fct,
                                     dip_vary_fct=dip_vary_fct,
                                     verbose=verbose)
            total_patches += len(seg.patch_verts)

        print(f"Discretized {total_patches} rectangular patches across {len(self.segments)} segments.")
        return total_patches

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def discretize_triangles(self, element_size, refine_near_trace=True):
        """Discretize curved fault surface into triangular patches using gmsh.

        This method now delegates to TriPatch for proper curved surface meshing.

        Args:
            - element_size:          Maximum triangle edge length (km)
            - refine_near_trace:     Use smaller triangles near surface trace (default True)

        Return:
            - List of triangular patches (each = [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)])
        """
        if self.surface_type != "curved" or self.trace_points is None:
            raise ValueError("discretize_triangles() requires from_trace_triangles() to be called first!")

        # Create TriPatch object (fault parameters are not needed for curved surface meshing)
        tri_patch = TriPatch(None, None, None, None, None)
        self.tri_patch_verts = tri_patch.discretize_curved(
            trace_points=self.trace_points,
            dips=self.trace_dips,
            widths=self.trace_widths,
            element_size=element_size,
            refine_near_trace=refine_near_trace
        )

        return self.tri_patch_verts

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def extend_to_surface_all(self):
        """Extend all fault segments to surface."""
        if not self.segments:
            warnings.warn("No fault segments to extend!")
            return

        for i, seg in enumerate(self.segments, 1):
            print(f"Extending segment {i}...")
            seg.extend_to_surface()

        print("All segments extended to surface.")

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def plot_segments(self, title="Multi-Segment Fault"):
        """Plot all fault segments in 3D."""
        if not self.segments:
            print("No fault segments to plot!")
            return

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all segments
        for i, seg in enumerate(self.segments):
            if seg.patch_verts is not None:
                verts = seg.patch_verts
                poly3d = Poly3DCollection(verts, edgecolor='black', alpha=0.7,
                                            facecolors=f'C{i}', label=f"Segment {i+1}")
                ax.add_collection3d(poly3d)

        # Calculate bounds
        all_coords = []
        for seg in self.segments:
            if seg.patch_verts is not None:
                for patch in seg.patch_verts:
                    for corner in patch:
                        all_coords.append(corner)

        all_coords = np.array(all_coords)
        ax.set_xlim(np.min(all_coords[:, 0]), np.max(all_coords[:, 0]))
        ax.set_ylim(np.min(all_coords[:, 1]), np.max(all_coords[:, 1]))
        ax.set_zlim(np.min(all_coords[:, 2]), max(0, np.max(all_coords[:, 2])) + 0.5)

        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title(title)
        ax.legend()

        plt.tight_layout()

        # # Save
        # import os
        # output_dir = "test_output"
        # os.makedirs(output_dir, exist_ok=True)
        # safe_title = title.replace(" ", "_").replace("/", "_")
        # output_path = os.path.join(output_dir, f"{safe_title}.png")
        # plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        # print(f"Plot saved to: {output_path}")

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def plot_trace(self, title="Curved Fault Trace"):
        """Plot curved fault trace with triangular mesh."""
        if self.trace_points is None or self.tri_patch_verts is None:
            print("No trace data to plot!")
            return

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot triangular mesh
        if self.tri_patch_verts is not None:
            poly3d = Poly3DCollection(self.tri_patch_verts, edgecolor='blue', alpha=0.5,
                                            facecolors='lightblue')
            ax.add_collection3d(poly3d)

        # Plot trace line
        trace = np.array(self.trace_points)
        ax.plot(trace[:, 0], trace[:, 1], trace[:, 2],
                'r-', linewidth=3, label='Fault Trace')

        # Plot trace points
        ax.scatter(trace[:, 0], trace[:, 1], trace[:, 2],
                   c='red', s=100, label='Trace Points')

        # Calculate bounds
        all_coords = [p for p in self.trace_points]
        if self.tri_patch_verts is not None:
            for tri in self.tri_patch_verts:
                all_coords.extend(tri)

        all_coords = np.array(all_coords)
        ax.set_xlim(np.min(all_coords[:, 0]) - 1, np.max(all_coords[:, 0]) + 1)
        ax.set_ylim(np.min(all_coords[:, 1]) - 1, np.max(all_coords[:, 1]) + 1)
        ax.set_zlim(np.min(all_coords[:, 2]) - 1, max(0, np.max(all_coords[:, 2])) + 1)

        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title(title)
        ax.legend()

        plt.tight_layout()

        # Save
        import os
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_")
        output_path = os.path.join(output_dir, f"{safe_title}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"Plot saved to: {output_path}")


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
if __name__ == "__main__":
    # Example 1: Curved fault from trace for triangular meshing
    print("=" * 60)
    print("Example 1: Curved fault from trace (for triangular meshing)")
    print("=" * 60)

    trace_points = [
        (44.28, 35.47, -2),
        (44.32, 35.52, -2),
        (44.36, 35.57, -2),
        (44.40, 35.62, -2),
        (44.44, 35.67, -2),
    ]
    dips = [35, 45, 60, 50, 40]
    widths = [12, 12, 12, 12, 12]

    transformer = CoordinateTransformer("multifault_example_crs", lon0=44.0, lat0=35.0)
    mf = MultiFault("curved_example", transformer=transformer)
    mf.from_trace_triangles(trace_points=trace_points, dips=dips, widths=widths)
    # Note: To plot the trace, call mf.discretize_triangles() first
    # mf.discretize_triangles(element_size=2.0)
    # mf.plot_trace()

    # Example 2: Chained fault segments for rectangular patch discretization
    print("\n" + "=" * 60)
    print("Example 2: Chained fault segments (for rectangular patches)")
    print("=" * 60)

    # NEW SIMPLIFIED FORMAT: Just provide trace points, dips and widths will auto-connect
    trace_points_rect = [
        (44.28, 35.47, -3),
        (44.35, 35.55, -3),
        (44.42, 35.63, -3),
        (44.50, 35.70, -3),
    ]
    dips_rect = [45, 60, 30]  # 3 dips for 3 segments (4 points -> 3 segments)
    widths_rect = [12, 12, 12]  # 3 widths for 3 segments

    mf2 = MultiFault("chained_example", transformer=transformer)
    mf2.from_trace_rectangles(trace_points=trace_points_rect, dips=dips_rect, widths=widths_rect)

    # Discretize and plot examples
    print("\n" + "=" * 60)
    print("Discretizing and plotting...")
    print("=" * 60)
    mf2.discretize_all_rectangles(sublength=2.0, subwidth=2.0, str_vary_fct=1.2, dip_vary_fct=1.2, verbose=True)
    mf2.plot_segments(title="Chained Fault Segments")
