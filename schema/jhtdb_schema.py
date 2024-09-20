import json
from enum import Enum
from typing import Union, Dict, List, Tuple, Annotated, Literal

from typing import Annotated, Optional

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from annotated_types import Len

###########################################
## The JHTDB Data Model: pydantic classes
###########################################
class Feature(BaseModel):
    '''
    general triple basically, used for listing the various featues (methods, operators etc) that are
    available on a Dataset.
    The code is to be used as identifier and references to a Feature in a certain collection.
    '''
    code: str
    name: str
    description: Optional[str] = None

class PhysicalVariable(Feature):
    cardinality: int = 1

class VariableOperatorMethod(BaseModel):
    '''
    Which spatial interpolation methods can be applied to the result of which operator applied to which variable.
    All represented by their codes pointing to the variables/operators/interpolation_methods in the database definition.
    '''
    variable: str  # code in a variable definition
    operator: str  # code in an operator definition
    methods: List[str]  # list of codes in method definitions
    
class Dimension(BaseModel):
    lower: str  # physical lower bound on dimension
    upper: str  # physical upper bound on dimension. str because it may have "pi" included.
    n: int = Field(ge=1)  # number of cells along the dimension
    isPeriodic: Optional[bool] = False
    
class Simulation(BaseModel):
    '''
    does not yet support description of irregular grids. should add subclasses for that.
    '''
    tlims: Dimension
    xlims: Dimension
    ylims: Dimension
    zlims: Dimension

class Dataset(BaseModel):
    displayname: str
    name: str
    simulation: Simulation
    description: Optional[str] = None
    
    physical_variables: List[str]   # list of codes of variables available in this dataset
    variable_operator_methods: List[VariableOperatorMethod] 
    
class TurbulenceDB(BaseModel):
    name: str
    description: Optional[str] = None
    variables: List[PhysicalVariable]
#     operators are derived fields that can be extracted in addition to the original field
#     examples are Hessian, Gradient, Laplacian
    spatial_operators: List[Feature]
    
#   Different spatial interpolation methods can be applied for point queries.
#   Examples are 
    spatial_methods: List[Feature]
    temporal_methods: List[Feature]
    datasets: List[Dataset]
    
    
########################################################
# GL these classes can be used server side.
# identify a config file describing the datasets 
# and adds some information relevant for server side actions
########################################################

class StorageDescriptor(BaseModel):
    storageType : Literal['TBD'] = 'TBD'
    
class LegacDBStorage(StorageDescriptor):
    storageType : Literal['LegacyDB'] = 'LegacyDB'
    turbinfoDatabaseURL: str
    
class CephZARRStorage(StorageDescriptor):
    storageType : Literal['ZARR'] = 'ZARR'
    cephParentDirectoryPath: str

class FileDBStorage(StorageDescriptor):
    storageType : Literal['FileDB'] = 'FileDB'
    filedbPickledMDFilePath: str

class DatasetStorageDescriptor(BaseModel):
    datasetName: str
    storageDescriptor: Union[LegacDBStorage, CephZARRStorage, FileDBStorage, StorageDescriptor] = Field(discriminator='storageType')

class JHTDBServerSide(BaseModel):
    jhtdbConfigFileURL: str
    datasets: List[DatasetStorageDescriptor]


    
########################################################    
## client side config
# Config files that add application specific metadata to the datasets.    
########################################################        
class CoordinateEnum(str, Enum):
    T = 't'
    X = 'x'
    Y = 'y'
    Z = 'z'

########################################################        
## cutout service config
########################################################        
class CutoutLimit(BaseModel):
    coordinate: CoordinateEnum
    lower: float
    upper: float
    default_lower: float
    default_upper: float
    
class DatasetCutout(BaseModel):
    datasetName: str
    cutout_variables: List[str]
    coordinate_lims: List[CutoutLimit]

class JHTDBCutout(BaseModel):
    jhtdbConfigFileURL: str
    datasets: List[DatasetCutout]
        
########################################################        
## point queries config
########################################################        
class CoordinateValue(BaseModel):
    coordinate: CoordinateEnum
    value: str  # can conain 'pi'

class DatasetDefaults(BaseModel):
    datasetName: str
    default_coordinates: List[CoordinateValue]
    
class JHTDBPointQuery(BaseModel):
    jhtdbConfigFileURL: str
    datasets: List[DatasetDefaults]
    