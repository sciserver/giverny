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

from enum import Enum
from typing import List
from pydantic import BaseModel

"""
client side config.
config files that add application specific metadata to the datasets.    
"""
class CoordinateEnum(str, Enum):
    T = 't'
    X = 'x'
    Y = 'y'
    Z = 'z'

"""      
point queries default settings config.
"""
class CoordinateValue(BaseModel):
    # space or time coordinate.
    coordinate: CoordinateEnum
    # default value for the coordinate.
    value: float

class PositionTimeInterval(BaseModel):
    # end time for position variable queries.
    endTime: float
    # time step for position variable queries.
    deltaT: float

class DatasetDefaults(BaseModel):
    # dataset name.
    datasetName: str
    # default coordinate values.
    default_coordinates: List[CoordinateValue]
    # position variable query parameters.
    position_time_interval: Optional[PositionTimeInterval] = None
    
class JHTDBPointQuery(BaseModel):
    # jhtdb-config.json config file url.
    jhtdbConfigFileURL: str
    # datasets that are queryable.
    datasets: List[DatasetDefaults]
