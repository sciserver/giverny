
# Change Log
All notable changes to giverny will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
 
## [Unreleased] - 2025-03-30

### Added

- function to handle variable grid axis spacing.
- getPosition function.

### Changed

- migrate datasets from SQL to CephFS storage.

## [3.3.1] - 2025-06-04

### Changed
- set giverny to read jhtdb-config.json from CephFS storage, and givernylocal to read
  jhtdb-config.json from within the library.

## [3.2.9] - 2025-06-04

### Fixed
- included the jhtdb-config.json file within the giverny and givernylocal libraries.

## [3.2.8] - 2025-06-03

### Changed
- updated jhtdb-config.json file to be read from local CephFS storage rather than GitHub CDN servers.

## [3.2.7] - 2025-05-27

### Changed
- updated the default time in jhtdb-points-config.json for 'rotstrat4096' and 'channel5200' datasets.
- removed getTurbineData.py and getBladeData.py from the DEMO_local_matlab_notebooks.zip file.

## [3.2.6] - 2025-05-24

### Changed
- updated jhtdb-config.json to reduce the maximum number of allowed threads for multiprocessing.
- modified turbulence_dataset.py to choose min(maximum_processes, cpu_count) for queries.

## [3.2.5] - 2025-05-24

### Changed
- updated README.

## [3.2.4] - 2025-05-23

### Changed
- updated README.

## [3.2.3] - 2025-05-23

### Changed
- updated README.

## [3.2.2] - 2025-05-22

### Changed
- updated the package dependency version control.

## [3.2.1] - 2025-05-21

### Added

- diurnal ('diurnal_windfarm'), and neutral boundary layer ('nbl_windfarm') windfarm datasets.
- Getwindfarmdata DEMO notebooks, including two new functions: getTurbineData, and getBladeData.
- Querywindfarmdata DEMO notebook for submitting SQL queries of the turbine and blade data.
- 'soiltemperature' variable for the diurnal windfarm dataset.
- pyarrow, and duckdb library dependencies.

### Changed
- moved the giverny constants declarations to the jhtdb-config.json file, and removed the constants.py file.
- jhtdb_schema.py to forbid parameters in the JSON file that are not explicitly defined in the model.
- removed the grid offsets parameter, and replaced the logic with coordinate offsets parameter.
- process (uv) and (w) components of the sabl velocity data together to reduce processing time.
  
### Fixed

- query size check to make sure too much data is not queried for time series.
- irregular mesh grid functions to handle the queried variable missing from the dataset map.
- removed step-down interpolation point mapping and implemented non-periodic axis boundary extrapolations to 
  apply the full specified spatial interpolation method to all queried points.

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
