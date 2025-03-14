########################################################################
#
#  Copyright 2024 Johns Hopkins University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Contact: turbulence@pha.jhu.edu
# Website: http://turbulence.pha.jhu.edu/
#
########################################################################

from typing import Optional, List, Union
from pydantic import BaseModel

"""
the JHTDB data model: pydantic classes.
"""
class Feature(BaseModel):
    """
    general triple basically, used for listing the various featues (methods, operators etc) that are
    available on a Dataset. the code is to be used as identifier and references to a Feature in a certain collection.
    """
    # formatted to match user input.
    code: str
    # formatted for display output.
    name: str
    # general description of the feature.
    description: Optional[str] = None
    
class Variable(Feature):
    """
    variable specific settings.
    """
    # codes for each component of the variable. e.g. [bx, by, bz] for the magneticfield variable.
    component_codes: List[str]
    # cardinality of the variable. e.g. 1 for scalar variables, and 3 for vector variables.
    cardinality: int
    
class SpatialMethod(Feature):
    """
    spatial interpolation method specific settings.
    """
    # number of cells to the left and right of the queried cell along each spatial axis which define the spatial interpolation bucket.
    bucketIndices: List[int]

class Storage(BaseModel):
    """
    storage general settings.
    """
    # filepath to the zarr store for a dataset.
    filepath: str
    # chunk size of each spatial axis (x, y, z) of the zarr store.
    chunks: List[int]

class Dimension(BaseModel):
    """
    dimension general settings.
    """
    # physical lower bound on dimension.
    lower: str
    # physical upper bound on dimension. str because it may have "pi" included.
    upper: str
    # number of cells along the dimension.
    n: int
    # whether the dimension boundary is periodic.
    isPeriodic: bool
    
class SpaceDimension(Dimension):
    """
    space dimension specific settings.
    """
    # grid spacing [dx, dy, dz].
    # irregular grid spacing dimensions are specified as a string and the grid coordinates are stored in python.
    spacing: Union[str, float]

class TimeIndexShift(BaseModel):
    """
    specifies how the user-specified time is shifted based on the query type to read from correct zarr
    time index. pchip interpolation requires a precursor time that is not queryable.
    """
    # time index shift for getcutout queries.
    getcutout: int
    # time index shift for getdata queries.
    getdata: int

class TimeDimension(Dimension):
    """
    time dimension specific settings.
    """
    # whether the dimension is stored as discrete time steps.
    isDiscrete: bool
    # shift applied to the user-specified time index. differs based on query type and whether or not
    # pchip time interpolation is allowed.
    timeIndexShift: TimeIndexShift

class Simulation(BaseModel):
    """
    simulation general settings.
    """
    # time dimension settings.
    tlims: TimeDimension
    # x-spatial dimension settings.
    xlims: SpaceDimension
    # y-spatial dimension settings.
    ylims: SpaceDimension
    # z-spatial dimension settings.
    zlims: SpaceDimension

class Offset(BaseModel):
    """
    offset general settings.
    """
    # variable code.
    code: str
    # staggered offset between spatial grid axes (x, y, z).
    grid: List[float]
    # coordinate offset from 0 along each spatial axis.
    coordinate: List[float]
    
class VariableOperatorMethod(BaseModel):
    """
    which spatial interpolation methods can be applied to the result of which operator applied to which variable.
    all represented by their codes pointing to the variables/operators/spatial methods in the database definition.
    """
    # spatial operator code.
    operator: str
    # valid spatial methods for the variable and operator.
    methods: List[str]

class PhysicalVariable(Feature):
    """
    physical variable specific settings.
    """
    # whether the variable is stored persistently on disk.
    isPersistent: bool
    # optional Storage settings.
    storage: Optional[Storage] = None
    # optional SpaceDimension settings.
    xlims: Optional[SpaceDimension] = None
    ylims: Optional[SpaceDimension] = None
    zlims: Optional[SpaceDimension] = None
    # staggered grid offsets between spatial grid axes (x, y, z), and coordinate offsets from 0 along each spatial axis.
    offsets: List[Offset]
    # valid spatial interpolation methods for each variable and operator.
    spatialOperatorMethods: List[VariableOperatorMethod]
    # valid temporal interpolation methods for the variable.
    temporalMethods: List[str]

class Dataset(Feature):
    """
    dataset general settings.
    """
    # storage settings, e.g. zarr filepath and chunk size of each spatial axis (x, y, z).
    storage: Storage
    # simulation settings.
    simulation: Simulation
    # list of codes of variables, and their associated parameters, available in this dataset.
    physicalVariables: List[PhysicalVariable]

class TurbulenceDB(BaseModel):
    """
    database general settings.
    """
    # name of the database.
    name: str
    # description of the database.
    description: Optional[str] = None
    # filepath where the pickled metadata files are stored, e.g. the spatial interpolation lookup tables.
    pickled_metadata_filepath: str
    # variables that can be queried from the datasets in the database.
    variables: List[Variable]
    # operators are derived fields that can be extracted in addition to the original field.
    # examples are gradient, hessian, laplacian.
    spatial_operators: List[Feature]
    # different spatial interpolation methods that can be applied for point queries.
    # examples are lag4, lag6, lag8, m2q8, fd4lag4, fd6noint.
    spatial_methods: List[SpatialMethod]
    # time interpolation methods.
    # examples are none, pchip.
    temporal_methods: List[Feature]
    # datasets included in the database.
    datasets: List[Dataset]
