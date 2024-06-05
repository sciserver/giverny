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

"""
data constants.
"""
def get_constants():
    """
    data constants:
        - database_file_disk_index: index that pulls out the parent folder for a database file when the filepath is split on each 
                                    folder separator. the parent folder references the hard disk drive that the database file is stored on.
        - dask_maximum_processes: the maximum number of python processes that dask will be allowed to create when using a local cluster.
        - missing_value_placeholder: placeholder for missing values that will be used to fill the output_data array when it is initialized.
        - bytes_per_datapoint: bytes per value associated with a datapoint.
        - max_cutout_size: maximum data size allowed to be retrieved by getCutout, in gigabytes (GB).
        - max_data_points: maximum number of points allowed to be queried by getData.
        - decimals: number of decimals to round to for precision of xcoor, ycoor, and zcoor in the xarray output.
        - chunk_size: cube size of the zarr chunks. the value is the length of one side of the cube.
        - file_size: cube size of an entire file. the value is the length of one side of the cube.
        - pyJHTDB_testing_token: current testing token for accessing datasets through pyJHTDB.
        - turbulence_email_address: current turbulence group e-mail address for requesting an authorization token.
    """
    return {
        'database_file_disk_index':-4,
        'dask_maximum_processes':64,
        'missing_value_placeholder':-999.9,
        'bytes_per_datapoint':4,
        'max_cutout_size':16.0,
        'max_data_points':2 * 10**6,
        'decimals':7,
        'chunk_size':64,
        'file_size':512,
        'pyJHTDB_testing_token':'edu.jhu.pha.turbulence.testing-201406',
        'turbulence_email_address':'turbulence@lists.johnshopkins.edu'
    }