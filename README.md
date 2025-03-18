# giverny
Python (version 3.9+) codebase for querying the [JHU Turbulence Database Cluster](https://turbulence.idies.jhu.edu/home) library.

DEMO notebooks for the various compute environments are provided at the [JHU Turbulence github](https://github.com/sciserver/giverny).

## Use giverny via Python through SciServer (RECOMMENDED)
`DEMO_SciServer_python_notebooks.zip`

The SciServer is a cloud-based data-driven cluster of The Institute for Data Intensive Engineering and Science (IDIES) at Johns Hopkins University. Users get the advantages of more reliable and faster data access since the SciServer is directly connected to JHTDB through a 10 Gigabit ethernet connection. SciServer provides containers with the `giverny` library pre-installed.

To use `giverny` through Sciserver:
```
1.) Login to [SciServer](https://sciserver.org/) (may need to create a new account first).
2.) Click on *Compute* and then *Create container*.
    - Can also run jobs in batch mode, by selecting *Compute Jobs*.
3.) Type in a *Container name*, select *SciServer Essentials (Test)* in *Compute Image*,
    mark *Turbulence (ceph)* in *Data volumes*, and then click on *Create*.
4.) Click on the container you just created to start using *giverny* with Python and JupyterLab.
```
Please go to [SciServer](https://sciserver.org/) for more information on SciServer as well as the help on SciServer.

Prerequisites:
```
numpy>=1.23.4
scipy>=1.9.3
sympy>=1.12
h5py>=3.7.0
matplotlib>=3.6.2
wurlitzer>=3.0.3
pydantic>=2.10.6
dill>=0.3.6
zarr>=2.13.3 
bokeh>=2.4.3
pandas>=1.5.1
requests>=2.31.0
xarray>=2022.11.0
tqdm>=4.64.1
tenacity>=8.1.0
plotly>=5.11.0
attrs>=23.2.0 
jsonschema>=4.23.0
jsonschema-specifications>=2023.12.1
nbformat>=5.10.4
referencing>=0.35.1
rpds-py>=0.19.1 
jupyter-core>=5.7.2
pyJHTDB>=20210108.0
SciServer>=2.1.0
```

## Use giverny via Python on local computers
`DEMO_local_python_notebooks.zip`

The first cell in the notebook includes the `pip` install command for the `givernylocal` library:
```
pip install --upgrade givernylocal
```
If you don't have `pip` on your system, it is quite easy to get it following the instructions at: [http://pip.readthedocs.org/en/latest/installation](http://pip.readthedocs.org/en/latest/installation).

Prerequisites:
```
numpy>=1.23.4
matplotlib>=3.6.2
pydantic>=2.10.6
pandas>=1.5.1
requests>=2.31.0
tenacity>=8.1.0
plotly>=5.11.0
attrs>=23.2.0 
jsonschema>=4.23.0
jsonschema-specifications>=2023.12.1
nbformat>=5.10.4
referencing>=0.35.1
rpds-py>=0.19.1
jupyter-core>=5.7.2
```

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
