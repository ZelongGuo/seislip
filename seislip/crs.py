#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coordinate reference system helpers for SeiSlip.

This module contains CoordinateTransformer, the shared geographic/UTM transformer used by
InSAR data readers and fault geometry classes etc.

Created on Tue Nov. 21 2023
@author: Zelong Guo
"""

from typing import Optional, Tuple, Union
import numpy as np
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info


class CoordinateTransformer(object):
    """Coordinate transformer between geographic lon/lat and projected UTM coordinates.

    ``CoordinateTransformer`` centralizes CRS selection and pyproj transformer setup for
    SeiSlip objects.  Domain objects hold it through composition; its core
    responsibility is coordinate transformation.

    Either geographic coordinates or an explicit UTM zone should be specified.
    Geographic coordinates are recommended for most workflows because pyproj
    can select the UTM CRS containing the reference point.  If ``utmzone`` is
    specified manually, it must include an explicit hemisphere suffix, e.g.
    ``"38N"`` or ``"36S"``.  Bare zones such as ``38`` or ``"38"`` are
    rejected because the hemisphere is ambiguous.  The ``N``/``S`` suffix here
    means northern/southern hemisphere, not an MGRS latitude band.

    Args:
        - name:     instance name of this parent class,
        - lon0:     longitude defining the center of the custom UTM zone,
        - lat0:     latitude defining the center of the custom UTM zone,
        - ellps:    (optional, default is "WGS84") reference ellipsoid of the data
        - utmzone:  (optional, default is None) explicit UTM zone with hemisphere, e.g. "38N".

    Return:
        - None.

    """

    def __init__(self, name: str, lon0: Optional[float] = None, lat0: Optional[float] = None, ellps: str = "WGS84",
                 utmzone: Optional[str] = None):

        self.name = name
        self.lon0 = lon0
        self.lat0 = lat0
        self.ellps = ellps

        self.__set_zone(lon0=lon0, lat0=lat0, ellps=ellps, utmzone=utmzone)

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

    # initialize UTM zone, this private method is called by __init__.
    # the initialization of following CRS referred to csi of Romain.
    @staticmethod
    def _parse_utmzone(utmzone: str) -> Tuple[int, bool]:
        """Parse a strict explicit UTM zone.

        Manual UTM input must be a string made of a zone number and a
        hemisphere suffix, for example ``"38N"`` or ``"36S"``.  The suffix is
        required so that the code never guesses northern vs. southern
        hemisphere from unrelated inputs.  ``N`` and ``S`` mean hemisphere
        only; MGRS latitude-band letters are intentionally not supported.
        """
        if not isinstance(utmzone, str):
            raise TypeError(
                "UTM zone must be a string with an explicit hemisphere suffix, "
                f"for example '38N' or '36S'; got {utmzone!r}."
            )

        zone_text = utmzone.strip().upper()
        if len(zone_text) < 2:
            raise ValueError(
                "UTM zone must include a zone number and hemisphere suffix, "
                f"for example '38N' or '36S'; got {utmzone!r}."
            )

        zone_digits = zone_text[:-1]
        hemisphere = zone_text[-1]
        if not zone_digits.isdigit():
            raise ValueError(f"UTM zone must start with a zone number, got {utmzone!r}.")
        if hemisphere not in {"N", "S"}:
            raise ValueError(
                "UTM zone must end with hemisphere suffix 'N' or 'S', "
                f"got {utmzone!r}."
            )

        zone_num = int(zone_digits)
        if not 1 <= zone_num <= 60:
            raise ValueError(f"UTM zone number must be between 1 and 60, got {zone_num}.")

        # Convert the hemisphere suffix to a bool flag for southern hemisphere.
        south = hemisphere == "S"
        return zone_num, south

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

    @staticmethod
    def _make_utm_crs(zone_num: int, south: bool, ellps: str, geographic_crs: CRS) -> CRS:
        """Create a UTM CRS for the parsed zone and hemisphere, in order to create a ll2xy and xy2ll
        convertor."""

        # For WGS84, the EPSG code is 4326
        if geographic_crs.to_epsg() == 4326:
            # EPSG:326xx represents WGS84 UTM zones in the northern hemisphere,
            # while EPSG:327xx represents WGS84 UTM zones in the southern hemisphere.
            epsg = (32700 if south else 32600) + zone_num
            return CRS.from_epsg(epsg)

        # For non-WGS84 geographic coordinates, build the UTM CRS manually
        # using PROJ parameters instead of predefined EPSG codes.
        proj_params = {
                "proj": "utm",
                "zone": zone_num,
                "ellps": ellps.replace(" ", "")
                }
        # Add the south flag only for southern-hemisphere UTM zone
        if south:
            proj_params["south"] = True
        return CRS.from_dict(proj_params)

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

    def __set_zone(self, lon0: Optional[float] = None, lat0: Optional[float] = None, ellps: str = "WGS84",
                   utmzone: Optional[str] = None) -> None:

        """Sets the UTM zone in the class.

        You can either give geographic coordinates (``lon0`` and ``lat0``) and
        let pyproj select the containing UTM zone, or specify an explicit UTM
        zone with hemisphere suffix (for example ``"38N"`` or ``"36S"``).

        Kwargs:
            - ellps:    Reference Ellipsoid of the data, default is "WGS84"

            :Method 1:
                - utmzone:      Explicit UTM zone with hemisphere, e.g. "38N" or "36S"

            :Method 2:
                - lon0:         Longitude of the center of the custom UTM zone (deg)
                - lat0:         Latitude of the center of the custom UTM zone (deg)

        Return:
            None.
        """

        self.wgs = CRS.from_user_input(ellps)

        if utmzone is not None:
            zone_num, south = self._parse_utmzone(utmzone)
            self.utm = self._make_utm_crs(zone_num, south, ellps, self.wgs)
        else:
            if lon0 is None or lat0 is None:
                raise ValueError(
                    "CoordinateTransformer requires either an explicit utmzone such as '38N' "
                    "or both lon0 and lat0 for automatic UTM zone selection."
                )
            # Find the zone containing the reference point.  Use a point-sized
            # area of interest to avoid selecting a neighbouring zone when a
            # wider AOI crosses a UTM boundary.
            utm_crs_list = query_utm_crs_info(
                datum_name="WGS 84",  # the name of the datum in the CRS name (‘NAD27’, ‘NAD83’, ‘WGS 84’, …)
                area_of_interest=AreaOfInterest(
                    west_lon_degree=lon0,
                    south_lat_degree=lat0,
                    east_lon_degree=lon0,
                    north_lat_degree=lat0
                ),
            )
            self.utm = CRS.from_epsg(utm_crs_list[0].code)
            # self.code = utm_crs_list[0].code

        # Make the projector
        self.proj2utm = Transformer.from_crs(self.wgs, self.utm, always_xy=True)
        self.proj2wgs = Transformer.from_crs(self.utm, self.wgs, always_xy=True)

        self.utmzone = self.utm.utm_zone

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
def init_coordinate_transformer_owner(owner, name: str, transformer: CoordinateTransformer) -> None:
    """Initialize a domain object with an existing coordinate transformer.

    ``Fault``, ``InSAR`` and ``MultiFault`` are not coordinate transformers
    themselves. They must receive a project-level ``CoordinateTransformer``
    and store it as ``self.transformer``. This keeps CRS setup in exactly one
    place, so all domain objects share the same UTM zone and pyproj settings.
    """
    if transformer is None:
        raise ValueError(
            "A CoordinateTransformer must be provided with transformer=. "
            "Create it once, then pass it to Fault, InSAR, or MultiFault."
        )
    if not isinstance(transformer, CoordinateTransformer):
        raise TypeError(
            "transformer must be a CoordinateTransformer instance, "
            f"got {type(transformer).__name__}."
        )

    owner.name = name
    owner.transformer = transformer

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
def transformer_property(attribute_name: str):
    """Create a read-only property that forwards to ``self.transformer``.

    Example: ``utmzone = transformer_property("utmzone")`` means that
    ``fault.utmzone`` returns ``fault.transformer.utmzone``. This keeps the
    convenient old attribute access without copying CRS state into every
    domain object.
    """
    return property(lambda self: getattr(self.transformer, attribute_name))

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

if __name__ == "__main__":
    test = CoordinateTransformer('TEST', -93, 43)

    lonlat = np.array([[-90.2897635, 40.1467463],
                       [-91.4456356, 43.5353664],
                       [-94.7463463, 44.8363636],
                       [-94.9236646, 42.9463463]])

    x, y = test.ll2xy(lonlat[:, 0], lonlat[:, 1])
    x_scalar, y_scalar = test.ll2xy(lonlat[0, 0], lonlat[0, 1])
    z = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])
    m, n = test.xy2ll(z[:, 0], z[:, 1])
    k = np.hstack([m.reshape(-1, 1), n.reshape(-1, 1)])


