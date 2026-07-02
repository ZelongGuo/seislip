# SeiSlip 
> update: 2023-06-09 23:43  
> Zelong Guo, @ Potsdam, zelong.guo@outlook.com


## Requirments:
- python >= 3.11

## Coordinate Reference Systems

`CoordinateTransformer` can initialize its UTM projection in two ways:

- Recommended: provide `lon0` and `lat0`; pyproj selects the UTM CRS containing that reference point.
- Manual: provide `utmzone` as an explicit hemisphere string such as `"38N"` or `"36S"`. Bare zones like `38` or `"38"` are rejected because the hemisphere is ambiguous. In this project, the `N`/`S` suffix means northern/southern hemisphere, not an MGRS latitude band.

Projected `x`/`y` values returned by `ll2xy()` are in kilometers.

Create one transformer for a project or dataset, then pass it to domain objects:

```python
from seislip.crs import CoordinateTransformer
from seislip import InSAR
from seislip.fault import Fault, MultiFault

transformer = CoordinateTransformer("project_crs", lon0=44.0, lat0=35.0)

fault = Fault("fault1", transformer=transformer)
insar = InSAR("track079", transformer=transformer)
multifault = MultiFault("fault_system", transformer=transformer)
```

`Fault`, `InSAR`, and `MultiFault` do not accept separate `lon0`/`lat0` CRS parameters; CRS setup belongs in `CoordinateTransformer`.

Cartesian and Fault Coordinate System:

<center>
    <img style="border-radius: 0.2125em;" src="./examples/imgs/fault_coordinate.png" width="50%" height="auto">
    <div style="
    display: outline;
    font-style: italic;
    color: #666;
    padding: 2px;"> Figure 1. The definition of cartesian and fault coordinate systems.  </div>
</center>




