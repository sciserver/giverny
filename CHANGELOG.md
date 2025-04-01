
# Change Log
All notable changes to giverny will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
 
## [Unreleased] - 2025-03-30

### Added

- diurnal and neutral boundary layer windfarm datasets.
- function to handle variable grid axis spacing.
- step-down linear interpolation methods for the windfarm datasets.
- getTurbineData and getBladeData DEMO notebooks to read windfarm parquet files.
- getPosition function.

### Changed

- process (uv) and (w) components of the sabl velocity data together to reduce processing time.
- migrate datasets from SQL to CephFS storage.

## [3.1.8] - 2025-04-01

### Changed

- updated giverny image and the corresponding reference in the README.
- removed version.py files from "giverny" and "givernylocal" source code as the version number
  is specified in the pyproject.toml files for both libraries.
  
### Fixed

- corrected the LICENSE specification in the pyproject.toml files.

## [3.1.7] - 2025-03-31

### Changed

- updated giverny image and the corresponding reference in the README.

## [3.1.6] - 2025-03-30

### Changed

- updated CHANGELOG.md to reflect that the Fortran code, DEMO_F.tar, was fixed in version 3.1.4.
 
## [3.1.5] - 2025-03-30

### Added

- CHANGELOG.md file.
 
### Fixed

- function calls for x-axis and y-axis linear interpolations.
 
## [3.1.4] - 2025-03-28
 
### Added

- coordinate offset logic for the windfarm datasets.

### Changed

- updated "Compute Image" name in the README from "SciServer Essentials (Test)" to "SciServer Essentials 4.0".

### Fixed

- interpolation boundary conditions for 'stsabl2048low' and 'stsabl2048high' datasets.
- implemented the missing gradient, hessian, and laplacian interpolation functions in the Fortran code, DEMO_F.tar.

## [3.1.3] - 2025-03-24

### Changed

- updated the cutout xarray dataset, hdf5, and xmf files to use the specified variable name rather than
  the display formatted variable name.
- removed unnecessary intermediate functions from the getCutout and getData ThreadPoolExecutor parallelizations.

### Fixed

- bug blocking queries of 'magneticfield' and 'vectorpotential' variables of the 'mhd1024' dataset.

## [3.1.2] - 2025-03-18

### Changed

- simplified and updated hyperlinks in the README.

## [3.1.1] - 2025-03-14
 
### Added

- missing variables in matlab getData.m file.

### Changed

- updated README.
