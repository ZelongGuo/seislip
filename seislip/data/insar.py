#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InSAR data and related operations.

Author: Zelong Guo 
Email: zelong.guo@outlook.com
Created on Tue May 18 20:08:23 2023

"""

__author__ = "Zelong Guo"
__version__ = "1.0.0"

import os.path
import math
# Standard and third-party libs
import sys
from typing import Optional, Union, Tuple, Dict
import struct
import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt
import warnings


# SlipPy libs
if __name__ == "__main__":
    sys.path.append("../")
    from seislip.crs import CoordinateTransformer, init_coordinate_transformer_owner, transformer_property
    from seislip.utils.quadtree import QTree
else:
    from ..crs import CoordinateTransformer, init_coordinate_transformer_owner, transformer_property
    from ..utils.quadtree import QTree


# Insar Class
class InSAR:
    """Insar class for handling InSAR data.

    Args:
        - name:     instance name
        - transformer: CoordinateTransformer instance defining the shared CRS

    Return:
        None.

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

    def __init__(self, name: str, transformer: CoordinateTransformer = None) -> None:
        init_coordinate_transformer_owner(self, name, transformer)

        print("+-" * 50)
        print(f"Now we initialize the InSAR instance {self.name}...")

        # Internal initialization
        self.data = None
        self.data_dsm = None

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

    def read_from_gamma(self, para_file: str, phase_file: str, azi_file: str, inc_file: str, satellite: str,
                        downsample: int = 3) -> None:
        """Read InSAR files (including phase, incidence, azimuth and DEM files) processed by GAMMA software
        with the parameter files, and preliminary downsampling the data if needed.

        This method will assign values to the instance attribute, self.data.

        Args:
            - para_file:        parameter files like *.utm.dem.par which aligns with the sar images after
            co-registration of dem and mli, resampling or oversampling dem file to proper resolution cell
            - phase_file:       filename of InSAR phase data
            - azi_file:         filename of azimuth file
            - inc_file:         filename of incidence file
            - satellite:        Satellite type, "Sentinel-1", "ALOS" ...
            - downsample:       downsample factor to reduce the number of the points

        Return:
            None.
        """

        # Firstly we read the para_file to get the porameters
        try:
            with open(para_file, 'r') as file:
                print("+-" * 25 + " Original Data " + "+-" * 25)
                print("Now we are reading the parameter file...")
                for line in file:
                    if line.startswith('width:'):
                        range_samples = int(line.strip().split(':')[1])
                    elif line.startswith('nlines:'):
                        azimuth_lines = int(line.strip().split(':')[1])
                    elif line.startswith('corner_lat:'):
                        corner_lat = float(line.strip().split(':')[1].split()[0])
                    elif line.startswith('corner_lon:'):
                        corner_lon = float(line.strip().split(':')[1].split()[0])
                    elif line.startswith('post_lat:'):
                        post_lat = float(line.strip().split(':')[1].split()[0])
                    elif line.startswith('post_lon:'):
                        post_lon = float(line.strip().split(':')[1].split()[0])
                    # Here we also read the geodetic datum, usually it would be WGS 84.
                    elif line.startswith('ellipsoid_name:'):
                        ellps_name = line.strip().split(':')[1].strip()

                    else:
                        pass
            post_arc = post_lon * 3600 # to arcsecond
            post_utm = post_arc * 40075017 / 360 / 3600  # earth circumference to ground resolution, meter
            # post_arc2 = "{:.2f}".format(post_arc)
            # post_utm2 = "{:.2f}".format(post_utm)
            # print("The InSAR pixel resoluton is {} arc-second, ~{} meters." .format(post_arc2, post_utm2))
            print(f"azimuth lines: {azimuth_lines}, Range samples: {range_samples} in {satellite} data.")
            print(f"The InSAR pixel resoluton is {post_arc:.3f} arc-second, ~{post_utm:.3f} meters.")

        except IOError:
            print("Error: cannot open the parameter file, please check the file path!")

        # Then we read the phase, azimuth and the incidence files to get the original data. All the files
        # processed by GAMMA are big-endian (swap_bytes="big-endian").
        try:
            with open(phase_file, 'rb') as f1, open(azi_file, 'rb') as f2, open(inc_file, 'rb') as f3:
                phase = np.zeros([range_samples, azimuth_lines])
                azimuth = np.zeros([range_samples, azimuth_lines])
                incidence = np.zeros([range_samples, azimuth_lines])

                # need read in column
                print("Now we are reading the phase, azimuth and incidence images (binary files)...")

                for i in range(azimuth_lines):
                    if i % 500 == 0:
                        # print(f"{i} ", end = '\r')
                        sys.stdout.write(f"{i} ")
                        sys.stdout.flush()
                    for j in range(range_samples):
                        # >f, big-endian, 4 bytes float
                        chunk = f1.read(4)
                        phase[j][i] = struct.unpack('>f', chunk)[0]
                        chunk = f2.read(4)
                        azimuth[j][i] = struct.unpack('>f', chunk)[0]
                        chunk = f3.read(4)
                        incidence[j][i] = struct.unpack('>f', chunk)[0]

            print(" ")
            print("+-" * 20 + " Primarily Downsampled Data " + "+-" * 20)
            range_samples_downsample = math.ceil(range_samples / downsample)
            azimuth_lines_downsample = math.ceil(azimuth_lines / downsample)
            print(f"There are {azimuth_lines_downsample}x{range_samples_downsample} in data"
                  f" with downsample factor {downsample}.")
            # make 0 values in phases, azi and inc to be Nan
            azimuth = np.where(phase == 0, np.nan, azimuth)
            incidence = np.where(phase == 0, np.nan, incidence)
            phase = np.where(phase == 0, np.nan, phase)
            # primarily downsample
            phase = phase.transpose()[::downsample, ::downsample]
            azimuth = azimuth.transpose()[::downsample, ::downsample]
            incidence = incidence.transpose()[::downsample, ::downsample]
            # los (unit in m)
            los = self._phase2los(phase=phase, satellite=satellite)
            # change to real azimuth and incidence with degree
            azimuth = -180 - np.degrees(azimuth)
            incidence = 90 - np.degrees(incidence)
            # lon and lat
            lats = np.linspace(corner_lat, corner_lat + (azimuth_lines - 1) * post_lat, azimuth_lines)
            lons = np.linspace(corner_lon, corner_lon + (range_samples - 1) * post_lon, range_samples)
            # Lons and Lats are 2-D matrix, Note the origin should be upper left, so Lats should grow small from top to bottom
            # zelong, 08.03.2024
            Lons, Lats = np.meshgrid(lons, lats)
            Lons, Lats = Lons[::downsample, ::downsample], Lats[::downsample, ::downsample]
            # utm
            utm_x, utm_y = self.transformer.ll2xy(Lons, Lats)

            post_arc_after_downsample = post_arc * downsample
            post_utm_after_downsample = post_utm * downsample
            print(f"The InSAR pixel resoluton is {post_arc_after_downsample:.3f} arc-second,"
                  f" ~{post_utm_after_downsample:.3f} meters.")

            # now we assign attributes to the instance
            self.data = {
                "lon":          {"value": Lons, "unit": "degree"},
                "lat":          {"value": Lats, "unit": "degree"},
                "x":            {"value": utm_x, "unit": "km"},
                "y":            {"value": utm_y, "unit": "km"},
                "phase":        {"value": phase, "unit": "radian"},
                "los":          {"value": los, "unit": "m"},
                "azi":          {"value": azimuth, "unit": "degree"},
                "inc":          {"value": incidence, "unit": "degree"},
                "parameters":   {
                    "satellite":                        satellite,
                    "datum_name":                       ellps_name,
                    "downsample_factor":                downsample,
                    "azimuth_lines_after_downsample":   azimuth_lines_downsample,
                    "range_samples_after_downsample":   range_samples_downsample,
                    "post_arc_after_downsample":        post_arc_after_downsample,
                    "post_utm_after_downsample":        post_utm_after_downsample,
                }
            }

        except IOError:
            print("Error: cannot open the image file, please check the file path or the parameters!")

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def read_from_xyz(self, xyz_file: str, interval_fact: int):
        """Read InSAR image from .xyz files.

        Args:
            - xyz_file     :        the InSAR xyz (table) file, should be 3 columns, comments starts with '#',
                                    mind the unit of deformation
            - interval_fact:        the interval factor, default is 1 and should be grater than 1. The new resolution
                                    of the data would be: interval_fact * maximum  resolution of the .xyz file.
        """

        # interval_fact should be grater than 1, that means this function cannot do interpolation.

        # NOTE: it would be better do not read the xyz file directly because it is not easy to get the unit...
        pass

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def read_from_grd(self, **kwargs):
        """Read InSAR image from .grd file, it should be a netCDF file.
        Example: t072.read_from_grd(los="./los.grd", inc="./inc.grd", azi="./azi.grd")

        Args:
            - kwargs:           the key should be: los, azi and/or inc, the value are the corresponding path.
        """
        from netCDF4 import Dataset

        # initialize the self.data to an empty dict
        self.data = {}

        for grd, grd_path in kwargs.items():
            if grd not in ("los", "azi", "azimuth", "inc", "incidence"):
                raise ValueError(f"{grd} in kwargs is not los, inc or azi !!!")

            # open and read the files
            dataset = Dataset(grd_path, "r", format="NETCDF4")
            data = dataset.variables
            # dimensions = dataset.dimensions

            if "lon" in data and "lat" in data:
                print("The dimensions are longitude and latitude.")
                lon = np.array(data['lon'][:])
                lat = np.array(data['lat'][:])
                # Lons should grow big from left to right, because we take the origin of the 2-D matrix to upper left
                # zelong, 08.03.2024
                if lon[0] > lon[-1]:
                    lon = np.flip(lon)
                # Lats should grow small from top to bottom, because we take the origin of the 2-D matrix to upper left
                if lat[0] < lat[-1]:
                    lat = np.flip(lat)
                Lons, Lats = np.meshgrid(lon, lat)   # the unit should be degrees
                utm_x, utm_y = self.transformer.ll2xy(Lons, Lats)  # the unit should be km
                self.data.update({
                    "lon":          {"value": Lons, "unit": "degree"},
                    "lat":          {"value": Lats, "unit": "degree"},
                    "x":            {"value": utm_x, "unit": "km"},
                    "y":            {"value": utm_y, "unit": "km"}
                })
            elif "x" in data and "y" in data:
                print("The dimensions are x and y.")
                x = np.array(data['x'][:])
                y = np.array(data['y'][:])
                x, y = np.meshgrid(x, y)
                # NOTE: do not know whether it's utm or the zone number, so it is an unknown Catersian coordinate
                if hasattr(data["x"], "units") and hasattr(data["y"], "units"):
                    x_unit = data["x"].units
                    y_unit = data["y"].units
                else:
                    warnings.warn("x and y do not have units!")
                    x_unit, y_unit = None, None
                self.data.update({
                    "x":          {"value": x, "unit": x_unit},
                    "y":          {"value": y, "unit": y_unit}
                })
            else:
                raise ValueError(f"The dimensions of the given grd {grd} file are neither lonlat nor xy!")

            # Deal with the Unit of z
            if hasattr(data["z"], "units"):
                z_unit = data["z"].units
            else:
                warnings.warn("z does not have unit!")
                z_unit = None

            if grd == "los":
                los = np.array(data["z"][:])
                # default origin is lower, we change it to upper for plotting
                los = np.flipud(los)
                self.data.update({"los": {"value": los, "unit": z_unit}})
            elif grd == "azi":
                azi = np.array(data["z"][:])
                azi = np.flipud(azi)
                self.data.update({"azi": {"value": azi, "unit": z_unit}})
            else:
                inc = np.array(data["z"][:]).transpose()
                inc = np.flipud(inc)
                self.data.update({"inc": {"value": inc, "unit": z_unit}})

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def _phase2los(self, phase: Union[float, np.ndarray], satellite: str) -> Union[float, np.ndarray]:
        """Converting InSAR phase to InSAR line-of-sight (los) disp.

        Args:
            - phase:            InSAR phase data.
            - satellite:        Satellite type, "Sentinel-1", "ALOS", "ALOS-2/U" ...

        Returns:
             - los:             InSAR LOS (unit in m)
        """
        if satellite in ("sentinel-1", "Sentinel-1", "s1", "S1"):
            radar_freq = 5.40500045433e9  # Hz
        elif satellite in ("ALOS", "alos"):
            radar_freq = 1.27e9  # Hz
        elif satellite == "ALOS-2/U":
            radar_freq = 1.2575e9
        elif satellite == "ALOS-2/{F,W}":
            radar_freq = 1.2365e9
        else:
            raise ValueError("The radar frequency of this satellite is not yet specified!")

        wavelength = c / radar_freq  # m
        los = - (phase / 2 / np.pi * wavelength / 2)

        # the unit is "m"
        return los

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def dsm_quadtree(self, mindim: int, maxdim: int, std_threshold: float, fraction: float = 0.3,
                     key: str = "los",  proj: str = "geo"):
        """Downsampling InSAR images (los deformation) with quadtree method.

        Args:
            - mindim:           minimum number of the image pixels consist of the image block
            - maxdim:           maximum number of the image pixels consist of the image block
            - std_threshold:    the standard deviation above which the image block will be split, unit in m
            - fraction:         the proportion of non-nan elements required in an image block, default is 0.3
            - key:              key value of the data, "los" or "phase"
            - proj:             geographic coordinates ("geo") or UTM projection ("utm")

        Returns:
            - None.
        """
        if proj == "geo":
            qtll = QTree(self.data["lon"]["value"], self.data["lat"]["value"], self.data[key]["value"])
        elif proj == "utm":
            qtll = QTree(self.data["x"]["value"], self.data["y"]["value"], self.data[key]["value"])
        else:
            raise ValueError("Please specify a corrct coordinate system!")
        qtll.subdivide(mindim, maxdim, std_threshold)
        qtll.qtresults(nonzero_fraction=fraction)
        # qtll.show_qtresults(key, self.data[key]["unit"])
        qtll.parameters = {
            "proj": proj,
            "key": key,
            "key_unit": self.data[key]["unit"]
        }
        self.data_dsm = qtll

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def mask(self, mask: list, key: str="los", proj: str="geo"):
        """Masking a specific area of the InSAR image, return the masked area and area after masking.
        Only suit for a rectangle area.

        Args:
            - mask:                         A list or tuple of the AOI, [[start_point_x, start_point_y], [x_width, y_height]],
                                            positive or negative values for x_width and y_width matter.
            - proj:                         Projection, "geo" or "utm"

        Returns:
            - after_mask_area
        """

        import copy
        mask = np.array(mask)

        start_point_x0, start_point_y0 = mask[0][0], mask[0][1]
        x_width, y_width = mask[1][0], mask[1][1]
        end_point_x1, end_point_y1 = start_point_x0 + x_width, start_point_y0 + y_width

        if proj == "geo":
            x0_diff = np.abs(self.data["lon"]["value"] - start_point_x0)
            y0_diff = np.abs(self.data["lat"]["value"] - start_point_y0)
            x1_diff = np.abs(self.data["lon"]["value"] - end_point_x1)
            y1_diff = np.abs(self.data["lat"]["value"] - end_point_y1)
        elif proj == "utm":
            x0_diff = np.abs(self.data["x"]["value"] - start_point_x0)
            y0_diff = np.abs(self.data["y"]["value"] - start_point_y0)
            x1_diff = np.abs(self.data["x"]["value"] - end_point_x1)
            y1_diff = np.abs(self.data["y"]["value"] - end_point_y1)
        else:
            raise ValueError("Wrong projection!")

        index_x0 = np.argmin(x0_diff, axis=1)
        index_y0 = np.argmin(y0_diff, axis=0)
        index_x1 = np.argmin(x1_diff, axis=1)
        index_y1 = np.argmin(y1_diff, axis=0)
        # remove repetition, should only 1 value left
        index_x0, index_y0 = set(index_x0).pop(), set(index_y0).pop()
        index_x1, index_y1 = set(index_x1).pop(), set(index_y1).pop()
        # width = index_x0 - index_x1
        # height = index_y0 - index_y1

        if key == "los":
            data_mask = copy.deepcopy(self.data["los"]["value"])
            if index_x0 < index_x1:
                x0 = index_x0
                x1 = index_x1
            else:
                x0 = index_x1
                x1 = index_x0
            if index_y0 < index_y1:
                y0 = index_y0
                y1 = index_y1
            else:
                y0 = index_y1
                y1 = index_y0
            data_mask[y0:y1, x0:x1] = np.nan
        else:
            raise ValueError("Not support yet.")

        self.data_mask = data_mask

        # return data_mask

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def plot(self, key: str, folder_path: str, fig_name: str) -> None:
        """Plot figure to seislip folder in current directory.

        Args:
            - key:                Key value of data you want plotting, "los", "phase", "azi", "inc", "dsm", "mask"
            - folder_path:        The folder path for saving the figure.
            - fig_name:           Specify a figure name without extension, .png file would be generated
                                  and saved automatically.

        Return:
            None.
        """
        # check and create "seislip" folder under working directory
        # folder_name = self.check_folder()

        # plotting
        match key:
            case "phase" | "los" | "azi" | "inc":
                if key == "phase":
                    # wrapped_phase = np.angle(np.exp(1j * self.data[key]["value"].reshape(-1, 1)))
                    # plt.scatter(self.data["lon"]["value"].reshape(-1, 1), self.data["lat"]["value"].reshape(-1, 1),
                    #             c=wrapped_phase, vmin=-np.pi, vmax=np.pi, cmap="rainbow")
                    wrapped_phase = np.angle(np.exp(1j * self.data[key]["value"]))
                    plt.imshow(wrapped_phase, cmap="rainbow", extent=[self.data["lon"]["value"].min(),
                                                                      self.data["lon"]["value"].max(),
                                                                      self.data["lat"]["value"].min(),
                                                                      self.data["lat"]["value"].max()], vmin=-np.pi,
                               vmax=np.pi, origin="upper")
                else:
                    # plt.scatter(self.data["lon"]["value"].reshape(-1, 1), self.data["lat"]["value"].reshape(-1, 1),
                    #             c=self.data[key]["value"].reshape(-1, 1), cmap="rainbow")
                    plt.imshow(self.data[key]["value"], cmap="rainbow",
                               extent=[self.data["lon"]["value"].min(), self.data["lon"]["value"].max(),
                                       self.data["lat"]["value"].min(), self.data["lat"]["value"].max()], origin="upper")

                plt.xlabel("Longitude (deg)")
                plt.ylabel("Latitude (deg)")
                plt.xlim([self.data["lon"]["value"].min(), self.data["lon"]["value"].max()])
                plt.ylim([self.data["lat"]["value"].min(), self.data["lat"]["value"].max()])
                plt.title(f"{self.name}")
                plt.colorbar(label=f"{key} [{self.data[key]['unit']}]")
                # plt.show()
                plt.savefig(os.path.join(folder_path, fig_name + '.png'), dpi=300)
                plt.close()
                print(f"Now {fig_name} is saved to {os.path.join(folder_path, fig_name + '.png')}")

            case "mask":
                plt.imshow(self.data_mask, cmap="rainbow",
                           extent=[self.data["lon"]["value"].min(), self.data["lon"]["value"].max(),
                                   self.data["lat"]["value"].min(), self.data["lat"]["value"].max()], origin="upper")
                plt.xlabel("Longitude (deg)")
                plt.ylabel("Latitude (deg)")
                plt.xlim([self.data["lon"]["value"].min(), self.data["lon"]["value"].max()])
                plt.ylim([self.data["lat"]["value"].min(), self.data["lat"]["value"].max()])
                plt.title(f"{self.name}_mask")
                # plt.colorbar(label=f"{key} [{self.data[key]['unit']}]")
                # plt.show()
                plt.savefig(os.path.join(folder_path, fig_name + '.png'), dpi=300)
                plt.close()
                print(f"Now {fig_name} is saved to {os.path.join(folder_path, fig_name + '.png')}")

            case "dsm":
                save_name = os.path.join(folder_path, fig_name + '.png')
                if self.data_dsm.parameters["proj"] == "geo":
                    self.data_dsm.show_qtresults(self.data_dsm.parameters["key"], "Lon (deg)", "Lat (deg)",
                                                 self.data_dsm.parameters["key_unit"], "yes", save_name)
                else:
                    self.data_dsm.show_qtresults(self.data_dsm.parameters["key"], "X (km)", "Y (km)",
                                                 self.data_dsm.parameters["key_unit"], "yes", save_name)
                print(f"Now {fig_name} is saved to {save_name}")

            case _:
                raise ValueError(f"Key {key} is not in data!")

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def deramp(self, dem_file):
        pass

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

    def write2grd(self, key: str, folder_path: str, grd_name: str):
        """grd_name without extension."""

        lon = self.data["lon"]["value"][0, :]
        lat = self.data["lat"]["value"][:, 0]
        row, col = lat.shape[0], lon.shape[0]

        if key in ("phase", "los", "azi", "inc"):
            data = self.data[key]["value"]
        elif key == "mask":
            data = self.data_mask
        else:
            raise ValueError(f"Key value ({key}) is not supported yet!")

        from netCDF4 import Dataset
        grd_path_name = folder_path + grd_name + ".nc"
        # TODO: More info should add to the grid file, e.g., unit etc.
        # file writting
        nc = Dataset(grd_path_name, "w", format="NETCDF4")
        # create dimensions
        nc.createDimension("lon", col)
        nc.createDimension("lat", row)
        # create variables
        lon_var = nc.createVariable("lon", "f4", ("lon",))
        lat_var = nc.createVariable("lat", "f4", ("lat",))
        z_var = nc.createVariable("z", "f4", ("lat", "lon"))
        # write into variables
        lon_var[:] = lon
        lat_var[:] = lat
        z_var[:, :] = data
        # add units info
        lon_var.units = "deggree_east"
        lat_var.units = "degree_north"
        # z_var.units = "m"

        nc.close()

        print(f"Now the grid file has been written into: {grd_path_name} !")


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
if __name__ == "__main__":

    transformer = CoordinateTransformer("t079_crs", lon0=44.28, lat0=35.47)
    t079 = InSAR("T079D_wang", transformer=transformer)
    t079_wang = "/misc/zs7/Zelong/2017_Iraq-Iran_EQ/Postseismic_InSAR_WangKang/dLOS_Sentinel-1/DES79/dlos_20181125.grd"
    t079.read_from_grd(los=t079_wang)

    fig_path = "/misc/zs7/Zelong/2017_Iraq-Iran_EQ-2/Response4Reviewers/figs/"
    # t079.plot(key="los", folder_path=fig_path, fig_name="t079_wang")

    t079.mask(mask=[[45.2, 34.4], [1.0, 0.8]])
    t079.plot("mask", fig_path, "data_mask")
    t079.write2grd("los", fig_path, "t079_wang")



    # t079.plot(key="los", folder_path=fig_path, fig_name="t079_wang")
    # t079.dsm_quadtree(16, 32, 0.1)
    # t079.plot("dsm", fig_path, "t072_wang_dsm")
# # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
#     dem_par = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T72A/SIM/20171111_20171117.utm.dem.par"
#     t072_file = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T72A/UNW/mcf/20171111_20171117.unw_utm"
#     inc = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T72A/SIM/lv_theta"
#     azi = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T72A/SIM/lv_phi"
#
#     # t072.read_from_gamma(dem_par, t072_file, azi, inc, satellite="Sentinel-1", downsample=10)
#
#     t072_gamma = InSAR("T072A_gamma", transformer=transformer)
#     t072_gamma.read_from_gamma(dem_par, t072_file, azi, inc, satellite="Sentinel-1", downsample=10)
#     t072_gamma.dsm_quadtree(16, 256, 0.015, 0.3, "los", "utm")
#     t072_gamma.plot("dsm", fig_path, "t072_gamma_dsm")




