# UTM Known-Bug Regression Test Design

## Objective

Add executable specifications for two known `GeoTrans` correctness defects without changing production behavior in the same commit. The tests define the scientifically correct CRS independently from the code under test and remain visible in every pytest run.

## Expected-failure policy

Each regression uses `pytest.mark.xfail(strict=True)`. While the defect exists, pytest reports the test as `XFAIL` and the existing passing baseline remains useful. When an implementation change makes the assertion pass, strict mode reports `XPASS` as a suite failure. That forces the fixing change to remove the marker and convert the specification into an ordinary permanent regression test.

Leaving the tests unmarked would make the entire branch permanently red and obscure unrelated regressions. Skipping them would provide no execution coverage. Strict expected failures retain both visibility and signal.

## Independent expectations

The automatic-zone case uses longitude 44 degrees east and latitude 35 degrees north. The containing WGS 84 UTM zone is 38N, represented by EPSG:32638. The test checks both the projected CRS and the public `utmzone` attribute; it does not use a forward/inverse round trip, because that can succeed inside the same wrong zone.

The southern case requests explicit zone 36S. Its correct WGS 84 projected CRS is EPSG:32736. The test checks CRS identity, `utmzone`, and a representative transformation of 30 degrees east, 25 degrees south against an independent pyproj transformer. Expected pyproj meter values are converted to SeiSlip's documented kilometer boundary before comparison.

## Scope boundary

No `GeoTrans` code is changed. Default ellipsoid spelling, validation, and broader CRS refactoring remain separate work.

