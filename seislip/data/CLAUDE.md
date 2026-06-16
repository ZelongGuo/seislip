# CLAUDE.md

Local guidance for Claude Code working in `seislip/data/`.

## Scope

This directory handles InSAR data ingestion, preprocessing, visualization, masking, and quadtree downsampling.

- `insar.py` is the main class-based API.
- `plot_data.py`, `utility.py`, `deramp_dem.py`, and `downsample.py` contain older function-style or prototype utilities.

## Data Model

- `InSAR.data` uses the nested dictionary pattern:

```python
self.data = {
    "key": {"value": np.ndarray, "unit": "unit_string"},
}
```

- Preserve this structure when adding new fields.
- Common keys are `lon`, `lat`, `x`, `y`, `phase`, `los`, `azi`, `inc`, and `parameters`.
- `lon`/`lat` are degrees; `x`/`y` are UTM kilometers; `phase` is radians; `los` is meters; `azi`/`inc` are degrees.

## Image Orientation

- Matrix origin is treated as upper-left for plotting and gridded image handling.
- Longitude should increase left to right.
- Latitude should decrease top to bottom after orientation normalization.
- Be careful with `np.flip`, `np.flipud`, and `.transpose()` in NetCDF/GAMMA readers; these encode real image-orientation choices.

## Reader Conventions

- `read_from_gamma()` reads GAMMA big-endian float binaries and converts zero phase to `np.nan`.
- `read_from_grd()` accepts NetCDF grids with either `lon`/`lat` or `x`/`y` dimensions.
- Satellite phase-to-LOS constants live in `InSAR._phase2los()`; keep sign convention and units stable unless explicitly changing scientific convention.
- `read_from_xyz()` and `deramp()` are currently placeholders.

## Maintenance Notes

- Prefer package-relative imports for library code. Some legacy modules still import from `data` directly; fix those only as part of a focused compatibility cleanup.
- Avoid adding import-time plotting or file I/O.
- Plotting methods may save figures; keep save paths explicit and avoid hard-coded research paths in reusable functions.
- If changing downsampling or masking logic, test array shape, coordinate orientation, and NaN handling together.
