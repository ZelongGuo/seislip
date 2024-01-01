#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InSAR data and related operations.

Author: Zelong Guo, @ GFZ, Potsdam
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


# SeiSlip libs
from ..seislip import GeoTrans
from ..utils.quadtree import QTree


# Insar Class
class InSAR(GeoTrans):
    """Insar class for handling InSAR data.

    Args:
        - name:     instance name
        - lon0:     longitude of the UTM zone
        - lat0:     latitude of the UTM zone
        - ellps:    ellipsoid, default = "WGS84"

    Return:
        None.

    """

    def __init__(self, name: str, lon0: Optional[float] = None, lat0: Optional[float] = None, ellps: str = "WGS 84",
                 utmzone: Optional[str] = None) -> None:
        # call init function of the parent class to initialize
        super().__init__(name, lon0, lat0, ellps, utmzone)

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
            Lons, Lats = np.meshgrid(lons, lats)
            Lons, Lats = Lons[::downsample, ::downsample], Lats[::downsample, ::downsample]
            # utm
            utm_x, utm_y = self.ll2xy(Lons, Lats)

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
                     key: str = "los",  proj: str = "utm"):
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
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def plot(self, key: str, fig_name: str) -> None:
        """Plot figure to SlipPy folder in current directory.

        Args:
            - key:                Key value of data you want plotting, "los", "phase", "azi", "inc", "dsm"
            - fig_name:           Specify a figure name without extension, .png file would be generated
                                  and saved automatically.

        Return:
            None.
        """
        # check and create "SlipPy" folder under working directory
        folder_name = self.check_folder()

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
                               vmax=np.pi)

                else:
                    # plt.scatter(self.data["lon"]["value"].reshape(-1, 1), self.data["lat"]["value"].reshape(-1, 1),
                    #             c=self.data[key]["value"].reshape(-1, 1), cmap="rainbow")
                    plt.imshow(self.data[key]["value"], cmap="rainbow",
                               extent=[self.data["lon"]["value"].min(), self.data["lon"]["value"].max(),
                                       self.data["lat"]["value"].min(), self.data["lat"]["value"].max()])

                plt.xlabel("Longitude (deg)")
                plt.ylabel("Latitude (deg)")
                plt.xlim([self.data["lon"]["value"].min(), self.data["lon"]["value"].max()])
                plt.ylim([self.data["lat"]["value"].min(), self.data["lat"]["value"].max()])
                plt.title(f"{self.name}")
                plt.colorbar(label=f"{key} [{self.data[key]['unit']}]")
                # plt.show()
                plt.savefig(os.path.join(folder_name, fig_name + '.png'), dpi=300)
                plt.close()
                print(f"Now {fig_name} is saved to {os.path.join(folder_name, fig_name + '.png')}")

            case "dsm":
                save_name = os.path.join(folder_name, fig_name + '.png')
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

    def write2file(self, dem_file):
        pass


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
if __name__ == "__main__":
    pass

