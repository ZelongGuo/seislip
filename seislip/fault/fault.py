#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 31.12.23

@author: Zelong Guo
@GFZ, Potsdam
"""

import sys
sys.path.append("../")
from seislip.seislip import GeoTrans

# SeiSip libs
# from ..slippy import GeoTrans

class Fault(GeoTrans):
    def __init__(self, name, lon0, lat0, ellps="WGS84", utmzone=None):
        super().__init__(name, lon0, lat0, ellps, utmzone)

        # fault parameters
        self.flon0 = None
        self.flat0 = None
        self.fdepth = None
        self.fstrike = None
        self.fdip = None
        self.flength = None
        self.fwidth = None


    def get_fault_pameters(self, lon, lat, depth, strike, dip, length, width):
        """Get the fault parameters to construct the fault.

        Args:
            - lon:          longitude of the central point on the upper edge of the fault, degree
            - lat:          latitude of the central point on the upper edge of the fault, degree
            - depth:        depth of the central point on the upper edge of the fault, km
            - strike:       strike angle of the fault, degree
            - dip:          dip angle of the fault, degree
            - width:        width along the fault dip direction, km
            - length:       length along the fault strike direction, km
        Return:
            - None.
        """
        self.flon0, self.lat0, self.fdepth = lon, lat, depth
        self.fstrike, self.fdip = strike, dip
        self.flength, self.fwidth = length, width

        self.futmx, self.futmy = self.ll2xy(lon, lat)

    def construct_fault(self):
        pass


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
if __name__ == "__main__":

    fault = Fault("flt", 44.28, 35.47)
    fault.get_fault_pameters(lon=44.344, lat=35.603, depth=3, strike=350, dip=15, length=80, width=50)