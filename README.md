<div align = "center">
  <img src = "https://raw.githubusercontent.com/sciserver/giverny/refs/heads/main/docs/imgs/monet-water_lilies.png" width = "50%"><br>
</div>

# giverny
[![PyPI](https://img.shields.io/pypi/v/giverny.svg?color=darkgreen)](https://pypi.org/project/giverny/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-582913.svg)](https://opensource.org/license/apache-2-0)
[![giverny PyPI downloads](https://img.shields.io/pypi/dm/giverny.svg?label=giverny%20%E2%A4%93&color=461C6C)](https://pypi.org/project/giverny/)
[![givernylocal PyPI downloads](https://img.shields.io/pypi/dm/givernylocal.svg?label=givernylocal%20%E2%A4%93&color=461C6C)](https://pypi.org/project/givernylocal/)

Library for querying the [Johns Hopkins Turbulence Database](https://turbulence.idies.jhu.edu/home).

DEMO notebooks for the various compute environments are provided at the [Johns Hopkins Turbulence github](https://github.com/sciserver/giverny).

## Python on SciServer (recommended)
`DEMO_SciServer_python_notebooks.zip`\
`DEMO_wind_SciServer_python_notebooks.zip`

The SciServer is a cloud-based data-driven cluster of The Institute for Data Intensive Engineering and Science (IDIES) at Johns Hopkins University. Users get the advantages of more reliable and faster data access since the SciServer is directly connected to the Johns Hopkins Turbulence Database (JHTDB) through a 10 Gigabit ethernet connection. SciServer provides containers with `giverny`, and all dependent libraries, pre-installed.

Please go to [SciServer](https://sciserver.org/) to create an account, and access more information as well as help on SciServer.

To use `giverny` through Sciserver:
1. Login to *SciServer*.
2. Click on *Compute* and then *Create container*.
    * Can also run jobs in batch mode, by selecting *Compute Jobs*.
3. Type in a *Container name*, in *Compute Image* select *SciServer Essentials 4.0*, in *Data volumes* mark *Turbulence (ceph)* and *Turbulence Windfarm (ceph)*, and then click on *Create*.
4. Click on the container you just created to start using *giverny* with Python and JupyterLab.

## Python on local computers
`DEMO_local_python_notebooks.zip`\
`DEMO_wind_local_python_notebooks.zip`

The first cell in the notebook runs the `pip` install command for the `givernylocal` library and all dependencies:
```
pip install --upgrade givernylocal
```
If you do not have `pip` on your system, it is quite easy to get it following the instructions at: [http://pip.readthedocs.org/en/latest/installation](http://pip.readthedocs.org/en/latest/installation).

## Matlab on local computers
`DEMO_local_matlab_notebooks.zip`\
`DEMO_wind_local_matlab_notebooks.zip`

## C on local computers
`DEMO_C.tar`

Please see the README inside the archive.

## Fortran on local computers
`DEMO_F.tar`

Please see the README inside the archive.

## Authorization token

While our service is open to anyone, we would like to keep track of who is using the service, and how. To this end, we would like each user or site to obtain an authorization token from us: [JHTDB authorization token](https://turbulence.idies.jhu.edu/staging/database)

For simple experimentation, the default token included in the package should be valid.
