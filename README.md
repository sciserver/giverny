# giverny
Python (version 3.9+) codebase for querying the [JHU Turbulence Database Cluster](https://turbulence.idies.jhu.edu/home) library.

DEMO notebooks for the various compute environments are provided at the [JHU Turbulence github](https://github.com/sciserver/giverny).

## Use giverny via Python through SciServer (RECOMMENDED)
`DEMO_SciServer_python_notebooks.zip`

The SciServer is a cloud-based data-driven cluster of The Institute for Data Intensive Engineering and Science (IDIES) at Johns Hopkins University. Users get the advantages of more reliable and faster data access since the SciServer is directly connected to JHTDB through a 10 Gigabit ethernet connection. SciServer provides containers with `giverny`, and all dependent libraries, pre-installed.

Please go to [SciServer](https://sciserver.org/) to create an account, and access more information as well as help on SciServer.

To use `giverny` through Sciserver:
1. Login to *SciServer*.
2. Click on *Compute* and then *Create container*.
    * Can also run jobs in batch mode, by selecting *Compute Jobs*.
3. Type in a *Container name*, select *SciServer Essentials (Test)* in *Compute Image*, mark *Turbulence (ceph)* in *Data volumes*, and then click on *Create*.
4. Click on the container you just created to start using *giverny* with Python and JupyterLab.

## Use giverny via Python on local computers
`DEMO_local_python_notebooks.zip`

The first cell in the notebook runs the `pip` install command for the `givernylocal` library and all dependencies:
```
pip install --upgrade givernylocal
```
If you don't have `pip` on your system, it is quite easy to get it following the instructions at: [http://pip.readthedocs.org/en/latest/installation](http://pip.readthedocs.org/en/latest/installation).

## Use giverny via Matlab on local computers
`DEMO_local_matlab_notebooks.zip`

## Use giverny via C on local computers
`DEMO_C.tar`

Please see the README inside the archive.

## Use giverny via Fortran on local computers
`DEMO_F.tar`

Please see the README inside the archive.

## Configuration

While our service is open to anyone, we would like to keep track of who is using the service, and how. To this end, we would like each user or site to obtain an authorization token from us: [JHTDB authorization token](https://turbulence.idies.jhu.edu/staging/database)

For simple experimentation, the default token included in the package should be valid.
