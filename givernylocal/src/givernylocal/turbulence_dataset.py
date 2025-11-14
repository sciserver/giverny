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

import pathlib
import numpy as np
from givernylocal.turbulence_gizmos.basic_gizmos import *

class turb_dataset():
    def __init__(self, dataset_title = '', output_path = '', auth_token = '',
                 json_url = 'https://raw.githubusercontent.com/sciserver/giverny/refs/heads/main/metadata/configs/jhtdb_config.json'):
        """
        initialize the class.
        """
        # load the json metadata.
        self.metadata = load_json_metadata(json_url)
        
        # check that dataset_title is a valid dataset title.
        check_dataset_title(self.metadata, dataset_title)
        
        # turbulence dataset name, e.g. "isotropic8192" or "isotropic1024fine".
        self.dataset_title = dataset_title
        
        # set the directory for saving any output files.
        self.output_path = output_path.strip()
        if self.output_path == '':
            raise Exception("output_path cannot be an empty string ('')")
        else:
            self.output_path = pathlib.Path(self.output_path)
        
        # create the output directory if it does not already exist.
        create_output_folder(self.output_path)
        
        # user authorization token for pyJHTDB.
        self.auth_token = auth_token
    
    """
    initialization functions.
    """
    def init_constants(self, query_type, var, var_offsets, timepoint, timepoint_original, sint, tint, option,
                       num_values_per_datapoint, c):
        """
        initialize the constants.
        """
        self.var = var
        self.var_offsets = var_offsets
        # convert the timepoint to [hour, minute, simulation number] for the windfarm datasets.
        if self.dataset_title in ['diurnal_windfarm', 'nbl_windfarm']:
            simulation_num = timepoint % 120
            minute = math.floor(timepoint / 120) % 60
            hour = math.floor((timepoint / 120) / 60)
            self.timepoint = [hour, minute, simulation_num]
        else:
            self.timepoint = timepoint
        self.timepoint_original = timepoint_original
        self.timepoint_end, self.delta_t = option
        # cube size.
        self.N = get_dataset_resolution(self.metadata, self.dataset_title, self.var)
        # cube spacing (dx, dy, dz).
        self.spacing = get_dataset_spacing(self.metadata, self.dataset_title, self.var)
        self.dx, self.dy, self.dz = self.spacing
        self.sint = sint
        self.tint = tint
        self.num_values_per_datapoint = num_values_per_datapoint
        self.bytes_per_datapoint = c['bytes_per_datapoint']
        self.missing_value_placeholder = c['missing_value_placeholder']
        self.max_num_chunks = c['max_num_chunks']
        self.decimals = c['decimals']
        self.chunk_size = get_dataset_chunk_size(self.metadata, self.dataset_title, self.var)
        self.query_type = query_type
        
        # set the byte order for reading the data from the files.
        self.dt = np.dtype(np.float32)
        self.dt = self.dt.newbyteorder('<')
        
        # retrieve the coordinate offsets.
        self.coor_offsets = get_dataset_coordinate_offsets(self.metadata, self.dataset_title, self.var_offsets, self.var)
        
        # set the dataset name to be used in the cutout hdf5 file.
        self.dataset_name = self.var + '_' + str(self.timepoint_original).zfill(4)
    