
# Change Log
All notable changes to giverny will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
 
## [Unreleased] - 2025-03-30

### Added

- add diurnal and neutral boundary layer windfarm datasets.
- function to handle variable grid axis spacing.
- step-down linear interpolation methods for the windfarm datasets.
- getTurbineData and getBladeData DEMO notebooks to read windfarm parquet files.
- getPosition function.

### Changed

- process (uv) and (w) components of the sabl velocity data together to reduce processing time.
- migrate datasets from SQL to CephFS storage.

### Fixed
 
## [3.1.5] - 2025-03-30

### Added

- made CHANGELOG.md file.
 
### Fixed

- fixed function calls for x-axis and y-axis linear interpolations.
 
## [3.1.4] - 2025-03-28
 
### Added

- added coordinate offset logic for the windfarm datasets.

### Changed

- updated "Compute Image" name in the README from "SciServer Essentials (Test)" to "SciServer Essentials 4.0".

### Fixed

- fixed interpolation boundary conditions for 'stsabl2048low' and 'stsabl2048high' datasets.

## [3.1.3] - 2025-03-24

### Changed

- updated the cutout xarray dataset, hdf5, and xmf files to use the specified variable name rather than
  the display formatted variable name.
- removed unnecessary intermediate functions from the getCutout and getData ThreadPoolExecutor parallelizations.

### Fixed

- fixed bug blocking queries of 'magneticfield' and 'vectorpotential' variables.

## [3.1.2] - 2025-03-18

### Changed

- simplified and updated hyperlinks in the README.

## [3.1.1] - 2025-03-14
 
### Added

- added missing variables in matlab getData.m file.

### Changed

- updated README.
