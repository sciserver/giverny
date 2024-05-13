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
    def __init__(self, dataset_title = '', output_path = '', auth_token = ''):
        """
        initialize the class.
        """
        # check that dataset_title is a valid dataset title.
        check_dataset_title(dataset_title)
        
        # turbulence dataset name, e.g. "isotropic8192" or "isotropic1024fine".
        self.dataset_title = dataset_title
        
        # cube size.
        self.N = get_dataset_resolution(self.dataset_title)
        # cube spacing (dx, dy, dz).
        self.spacing = get_dataset_spacing(self.dataset_title)
        self.dx, self.dy, self.dz = self.spacing
        
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
    def init_constants(self, query_type, var, var_original, var_offsets, timepoint, timepoint_original, sint, sint_specified, tint, option,
                       num_values_per_datapoint, c):
        """
        initialize the constants.
        """
        self.var = var
        self.var_name = var_original
        self.var_offsets = var_offsets
        self.timepoint = timepoint
        self.timepoint_original = timepoint_original
        self.timepoint_end, self.delta_t = option
        # sint and sint_specified are the same except for points near the upper and lower z-axis boundaries in
        # the 'sabl2048*' datasets. for these datasets sint is automatically reduced to an interpolation method
        # that fits within the z-axis boundary since the z-axis is not periodic. sint_specified will be used for
        # reading the proper interpolation lookup table(s) from the metadata files.
        self.sint = sint
        self.sint_specified = sint_specified
        self.tint = tint
        self.num_values_per_datapoint = num_values_per_datapoint
        self.bytes_per_datapoint = c['bytes_per_datapoint']
        self.missing_value_placeholder = c['missing_value_placeholder']
        self.database_file_disk_index = c['database_file_disk_index']
        self.decimals = c['decimals']
        self.chunk_size = c['chunk_size']
        self.file_size = c['file_size']
        self.query_type = query_type
        
        # set the byte order for reading the data from the files.
        self.dt = np.dtype(np.float32)
        self.dt = self.dt.newbyteorder('<')
        
        # get the number of digits in the timepoint part of the filenames, in order to add leading zeros.
        self.timepoint_digits = get_timepoint_digits(self.dataset_title)
        
        # retrieve the dimension offsets.
        self.dimension_offsets = get_dataset_dimension_offsets(self.dataset_title, self.var_offsets)
        
        # retrieve the coor offsets.
        self.coor_offsets = get_dataset_coor_offsets(self.dataset_title, self.var_offsets)
        
        # set the dataset name to be used in the cutout hdf5 file.
        self.dataset_name = get_output_variable_name(self.var_name) + '_' + str(self.timepoint_original).zfill(4)
    