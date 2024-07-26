# giverny
Python (version 3.9+) codebase for querying the [JHU Turbulence Database Cluster](http://turbulence.pha.jhu.edu/) library.

## Use giverny through SciServer (RECOMMENDED)
The SciServer is a cloud-based data-driven cluster, of The Institute for Data Intensive Engineering and Science (IDIES) at Johns Hopkins University. Users get the advantages of more reliable and faster data access since the SciServer is directly connected to JHTDB through a 10 Gigabit ethernet connection. SciServer provides containers with the "giverny" library pre-installed.

Demo notebooks for the SciServer compute environment are provided at the [JHU Turbulence github](https://github.com/sciserver/giverny).

To use giverny through Sciserver:
```
Login to [SciServer](https://sciserver.org/) (may need to create a new account first).
Click on *Compute* and then *Create container* (You could also run jobs in batch mode, by selecting Compute Jobs).
Type in *Container name*, select *SciServer Essentials (Test)* in *Compute Image*, mark *Turbulence (filedb)* in *Data volumes*, and then click on *Create*.
Click on the container you just created, then you could start using giverny with Python or IPython Notebook.
```
Please go to [SciServer](https://sciserver.org/) for more information on SciServer as well as the help on SciServer.

Prerequisites: numpy>=1.23.4, scipy>=1.9.3, sympy>=1.12, h5py>=3.7.0, matplotlib>=3.6.2, wurlitzer>=3.0.3, morton-py>=1.3, dill>=0.3.6, zarr>=2.13.3, 
bokeh>=2.4.3, dask>=2022.11.0, pandas>=1.5.1, xarray>=2022.11.0, tqdm>=4.64.1, tenacity>=8.1.0, plotly>=5.11.0, attrs>=23.2.0, jsonschema>=4.23.0, jsonschema-specifications>=2023.12.1, 
nbformat>=5.10.4, referencing>=0.35.1, rpds-py>=0.19.1, jupyter-core>=5.7.2, pyJHTDB>=20210108.0, SciServer>=2.1.0

## Use giverny on local computers

Demo notebooks for the local compute environment are provided at the [JHU Turbulence github](https://github.com/sciserver/giverny).

If you have *pip*, you can simply do this:
```
pip install givernylocal
```
If you're running unix (i.e. some MacOS or GNU/Linux variant), you will probably need to have a `sudo` in front of the `pip` command. If you don't have `pip` on your system, it is quite easy to get it following the instructions at [http://pip.readthedocs.org/en/latest/installation](http://pip.readthedocs.org/en/latest/installation).

Prerequisites: numpy>=1.23.4, matplotlib>=3.6.2, pandas>=1.5.1, requests>=2.31.0, tenacity>=8.1.0, plotly>=5.11.0, attrs>=23.2.0, jsonschema>=4.23.0, jsonschema-specifications>=2023.12.1, 
nbformat>=5.10.4, referencing>=0.35.1, rpds-py>=0.19.1, jupyter-core>=5.7.2

## Configuration

While our service is open to anyone, we would like to keep track of who is using the service, and how. To this end, we would like each user or site to obtain an authorization token from us: [http://turbulence.pha.jhu.edu/help/authtoken.aspx](http://turbulence.pha.jhu.edu/help/authtoken.aspx)

For simple experimentation, the default token included in the package should be valid.
