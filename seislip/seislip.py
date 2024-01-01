#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the base classes for SlipPy.

Created on Tue Nov. 21 2023
@author: Zelong Guo, GFZ, Potsdam
"""
__author__ = "Zelong Guo"
__version__ = "1.0.0"

import os
from typing import Optional, Union, Tuple
import numpy as np
from pyproj import CRS, Geod, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info


class GeoTrans(object):
    """A parent class to perform coordinates transformation between geographic (lon and lat) and projection
    (UTM) coordinates.

    Either geographic coordinates or UTM zone should be specified. However, we do not recommend specifying
    utm zone number since it would give negative easting values if the zone number is not appropriate.

    Args:
        - name:     instance name of this parent class,
        - lon0:     longitude defining the center of the custom UTM zone,
        - lat0:     latitude defining the center of the custom UTM zone,
        - ellps:    (optional, default is "WGS 84") reference ellipsoid of the data
        - utmzone:  (optional, default is None) the number of the UTM zone.

    Return:
        - None.

    """

    def __init__(self, name: str, lon0: Optional[float] = None, lat0: Optional[float] = None, ellps: str = "WGS 84",
                 utmzone: Optional[str] = None):

        self.name = name
        self.lon0 = lon0
        self.lat0 = lat0
        self.ellps = ellps

        self.__set_zone(lon0=lon0, lat0=lat0, ellps=ellps, utmzone=utmzone)

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

    # initialize UTM zone, this private method is called by __init__.
    # the initialization of following CRS referred to csi of Romain.
    def __set_zone(self, lon0: Optional[float] = None, lat0: Optional[float] = None, ellps: str = "WGS 84",
                   utmzone: Optional[str] = None) -> None:

        """Sets the UTM zone in the class.

        You can specify the utm zone NO. directly or give the geographic coordinates of
        longitude (lon0) and latitude (lat0).

        Kwargs:
            - ellps:    Reference Ellipsoid of the data, default is "WGS 84"

            :Method 1:
                - utmzone:      International UTM zone number

            :Method 2:
                - lon0:         Longitude of the center of the custom UTM zone (deg)
                - lat0:         Latitude of the center of the custom UTM zone (deg)

        Return:
            None.
        """

        # if the geodetic datum is WGS84, it equals to self.wgs = pp.CRS.from_epsg(4326)
        self.wgs = CRS(ellps)

        if utmzone is not None:
            self.utm = CRS(proj='utm', zone=utmzone, ellps=ellps)
        else:
            assert lon0 is not None, 'Please specify a longitude (lon0)!'
            assert lat0 is not None, 'Please specify a latitude (lat0)!'
            # Find the best zone. Note, every coordinate system has a unique reference code, the so-called EPSG code,
            # for WGS84, its EPSG code is 4326.
            utm_crs_list = query_utm_crs_info(
                datum_name="WGS 84",  # the name of the datum in the CRS name (‘NAD27’, ‘NAD83’, ‘WGS 84’, …)
                area_of_interest=AreaOfInterest(
                    west_lon_degree=lon0 - 2.,
                    south_lat_degree=lat0 - 2.,
                    east_lon_degree=lon0 + 2,
                    north_lat_degree=lat0 + 2
                ),
            )
            self.utm = CRS.from_epsg(utm_crs_list[0].code)
            # self.code = utm_crs_list[0].code

        # Make the projector
        self.proj2utm = Transformer.from_crs(self.wgs, self.utm, always_xy=True)
        self.proj2wgs = Transformer.from_crs(self.utm, self.wgs, always_xy=True)

        if utmzone is None:
            self.utmzone = self.utm.utm_zone

        # Set Geod
        # self.geod = Geod(ellps=ellps)

        # Set utmzone
        # self.utmzone = utmzone
        # self.lon0 = lon0
        # self.lat0 = lat0
        # self.ellps = ellps

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def ll2xy(self, lon: Union[float, np.ndarray], lat: Union[float, np.ndarray]) \
            -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Converting longitudes and latitudes to UTM X Y coordinates.

        Args:
            lon:        Longitudes (deg)
            lat:        Latitudes (deg)

        Return:
            X:          UTM easting coordinates (km)
            Y:          UTM northing coordinates (km)

        """
        x, y = self.proj2utm.transform(lon, lat)

        # the unit is km
        x, y = x /1000, y/1000

        return x, y

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def xy2ll(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray])\
            -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Converting UTM X Y coordinates to longitudes and latitudes.

        Args:
            x:          UTM Easting (km)
            y:          UTM northing (km)

        Return:
            lon:        Longitudes (deg)
            lat:        Latitudes (deg)

        """
        lon, lat = self.proj2wgs.transform(x*1000, y*1000)

        return lon, lat

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def check_folder(self) -> str:
        """Check and create a folder named "SeiSlip" if it does not exist in current working directory.
        This folder is used for saving files.

        Args:
            None.

        Return:
            folder_path:        Absolute folder path.
        """

        current_path = os.getcwd()
        # check the folder's existence
        folder_path = os.path.join(current_path, "SeiSlip")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        return folder_path


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
if __name__ == "__main__":
    test = GeoTrans('TEST', -93, 43)
    # test.check_folder()

    lonlat = np.array([[-90.2897635, 40.1467463],
                       [-91.4456356, 43.5353664],
                       [-94.7463463, 44.8363636],
                       [-94.9236646, 42.9463463]])

    x, y = test.ll2xy(lonlat[:, 0].reshape(-1,1), lonlat[:, 1])
    x, y = test.ll2xy(lonlat[0, 0], lonlat[0, 1])

    a = np.array([-90.1, -91.2, -92, -93])
    b = np.array([40.3, 41.2, 43, 42])
    x, y = test.ll2xy(a, b)

    z = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])

    m, n = test.xy2ll(z[:, 0], z[:, 1])
    k = np.hstack([m.reshape(-1, 1), n.reshape(-1, 1)])



