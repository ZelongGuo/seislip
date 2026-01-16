# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SeiSlip is a Python package for seismological slip inversion analysis using InSAR data. It provides tools for:

- InSAR data processing (reading, coordinate transformation, downsampling, visualization)
- Fault geometry modeling and discretization
- Coordinate transformations between geographic (lon/lat), UTM, and fault coordinate systems

**Requirements**: Python >= 3.11

## Code Architecture

### Core Coordinate System Classes

The codebase uses a hierarchical class structure for coordinate transformations:

1. **`GeoTrans`** (`seislip/seislip.py:20`) - Base class for geographic to UTM coordinate transformations using `pyproj`. Handles:
   - lon/lat to UTM X/Y (km units)
   - UTM to lon/lat
   - Automatic UTM zone detection from center point or manual specification

2. **`InSAR`** (`seislip/data/insar.py:38`) - Extends `GeoTrans` for InSAR data handling
3. **`Fault`** (`seislip/fault/fault.py:31`) - Extends `GeoTrans` for fault geometry modeling

### InSAR Data Module (`seislip/data/`)

**`InSAR` class** - Main class for InSAR data operations:

- **Data input methods**:
  - `read_from_gamma()` - Read GAMMA-processed InSAR files (binary phase, azimuth, incidence)
  - `read_from_grd()` - Read NetCDF grid files (.grd/.nc format)
  - `read_from_xyz()` - Read xyz table files (TODO)

- **Data processing**:
  - `dsm_quadtree()` - Quadtree downsampling using `QTree` utility
  - `mask()` - Mask rectangular areas of InSAR images
  - `deramp()` - DEM-related error and orbit ramp removal (TODO)
  - `_phase2los()` - Convert phase to line-of-sight displacement

- **Coordinate systems**: Stores data in multiple coordinate formats:
  - `lon`/`lat` - Geographic coordinates (degrees)
  - `x`/`y` - UTM coordinates (km)
  - `phase` - Radian phase
  - `los` - Line-of-sight displacement (meters)
  - `azi`/`inc` - Azimuth and incidence angles (degrees)

### Fault Module (`seislip/fault/`)

**`Fault` class** (`seislip/fault/fault.py`) - Planar fault modeling:

- **Initialization**: `initialize_fault()` defines fault geometry from a reference point (upper/center, upper/origin, centroid, etc.) with strike, dip, length, and width
- **Fault coordinate system** (right-hand):
  - X: along strike
  - Y: along opposite direction of dip
  - Z: normal direction to fault
- **Surface breach handling**: `extend_to_surface()` extends fault to surface if upper edge is below ground, automatically adjusts width
- **Discretization**: `construct_rect_patches()` creates rectangular patches, optionally with depth-varying sizes

**`RectPatch` class** (`seislip/fault/rectpatch.py`) - Rectangle fault patch discretization:
- Uses `Transformation` class for coordinate conversion between UTM and fault coordinate systems
- Supports uniform discretization or strike/dip-varying patch sizes
- Returns patches as lists of 4-corner vertices in UTM coordinates

### Utilities (`seislip/utils/`)

**`Transformation` class** (`seislip/utils/transformation.py`) - Homogeneous coordinate transformations:
- Implements 4x4 transformation matrices using left-multiply convention (fixed coordinate system transformations)
- Operations: `translation()`, `rotation_x()`, `rotation_y()`, `rotation_z()`
- `inverse()` computes inverse transformation matrix
- Used by `RectPatch` for UTM ↔ fault coordinate system conversion

**`QTree` class** (`seislip/utils/quadtree.py`) - Quadtree downsampling for InSAR images:
- Recursively subdivides image based on:
  - `mindim`/`maxdim` - Minimum/maximum block sizes (pixels)
  - `std_threshold` - Standard deviation threshold for splitting
  - `fraction` - Minimum non-zero/valid fraction in each block
- `qtresults()` returns downsampled data as scatter points and rectangle patches (GMT format ready)
- `show_qtresults()` visualizes original, scatter, and rectangle representations

## Important Conventions

### Coordinate System Definitions

**UTM System**: X (easting), Y (northing), Z (zenith), units in **km**

**Fault Coordinate System**:
- Origin: upper center point of fault
- X: along strike direction
- Y: opposite direction of dip (downward along fault plane)
- Z: normal direction from fault plane

**Depth Convention**: Vertical depth should be specified as **negative values** (e.g., -5 km for 5 km depth)

### Point Position Naming

Fault reference points (used in `Fault.initialize_fault()`):
- `uo`/`UO` - Upper origin (upper-left in strike-dip view)
- `uc`/`UC` - Upper center (most commonly used)
- `ue`/`UE` - Upper end (upper-right)
- `bo`/`BO` - Bottom origin
- `bc`/`BC` - Bottom center
- `be`/`BE` - Bottom end
- `cc`/`CC` - Centroid center

### Data Structure Pattern

Data dictionaries in `InSAR` class use the pattern:
```python
self.data = {
    "key": {"value": np.ndarray, "unit": "unit_string"},
    ...
}
```

### Satellite Frequency Constants

In `_phase2los()` method (`seislip/data/insar.py:302`):
- Sentinel-1: 5.40500045433e9 Hz
- ALOS: 1.27e9 Hz
- ALOS-2/U: 1.2575e9 Hz
- ALOS-2/{F,W}: 1.2365e9 Hz

## Module Organization

```
seislip/
├── seislip/
│   ├── __init__.py           # Package exports
│   ├── seislip.py           # GeoTrans base class
│   ├── data/
│   │   ├── insar.py         # InSAR class
│   │   └── ...
│   ├── fault/
│   │   ├── fault.py        # Fault class
│   │   └── rectpatch.py   # RectPatch class
│   └── utils/
│       ├── transformation.py # Transformation class
│       └── quadtree.py     # QTree class
├── tests/
└── examples/
```

## Testing

- Test file: `tests/test.py` (contains experimental code, currently mostly commented)
- No formal test framework is set up yet
