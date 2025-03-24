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

import os
import sys
import dill
import glob
import math
import time
import zarr
import pathlib
import itertools
import subprocess
import numpy as np
import pandas as pd
from collections import defaultdict
from SciServer import Authentication
from concurrent.futures import ThreadPoolExecutor
from giverny.turbulence_gizmos.basic_gizmos import *
from giverny.turbulence_gizmos.constants import get_constants

class turb_dataset():
    def __init__(self, dataset_title = '', output_path = '', auth_token = '', rewrite_interpolation_metadata = False):
        """
        initialize the class.
        """
        # load the json metadata.
        self.metadata = load_json_metadata()
        
        # check that dataset_title is a valid dataset title.
        check_dataset_title(self.metadata, dataset_title)
        
        # dask maximum processes constant.
        self.dask_maximum_processes = get_constants()['dask_maximum_processes']
        
        # turbulence dataset name, e.g. "isotropic8192" or "isotropic1024fine".
        self.dataset_title = dataset_title
        
        # interpolation lookup table resolution.
        self.lookup_N = 10**5
        
        # set the directory for saving any output files.
        self.output_path = output_path.strip()
        if self.output_path == '':
            # get the SciServer user name. note: these notebooks are designed for use in a SciServer container.
            user = Authentication.getKeystoneUserWithToken(Authentication.getToken()).userName
            
            self.output_path = pathlib.Path(f'/home/idies/workspace/Temporary/{user}/scratch/turbulence_output')
        else:
            self.output_path = pathlib.Path(self.output_path)
        
        # create the output directory if it does not already exist.
        create_output_folder(self.output_path)
        
        # user authorization token for pyJHTDB.
        self.auth_token = auth_token
        
        # set the directory for reading the pickled files.
        self.pickle_dir = pathlib.Path(self.metadata['pickled_metadata_filepath'])
        
        # set the local directory for writing the pickled metadata files if the primary and backup directories are inaccessible.
        self.pickle_dir_local = self.output_path.joinpath('turbulence_metadata')
        
        # create the local pickle directory if it does not already exist.
        create_output_folder(self.pickle_dir_local)
        
        """
        read/write metadata files.
        """
        # retrieve the list of datasets processed by the giverny code.
        giverny_datasets = get_giverny_datasets()
        
        # only read/write the metadata files if the dataset being queried is handled by this code.
        if self.dataset_title in giverny_datasets:
            # rewrite interpolation metadata files if specified.
            if rewrite_interpolation_metadata:
                # initialize the interpolation lookup table.
                self.init_interpolation_lookup_table(read_metadata = False, rewrite_metadata = rewrite_interpolation_metadata)
    
    """
    initialization functions.
    """ 
    def init_interpolation_lookup_table(self, sint = 'none', read_metadata = False, rewrite_metadata = False):
        """
        pickled interpolation lookup table.
        """
        # interpolation method 'none' is omitted because there is no lookup table for 'none' interpolation.
        interp_methods = ['lag4', 'm1q4', 'lag6', 'lag8', 'm2q8',
                          'fd4noint_gradient', 'fd6noint_gradient', 'fd8noint_gradient', 'fd4lag4_gradient', 'm1q4_gradient', 'm2q8_gradient',
                          'fd4noint_laplacian', 'fd6noint_laplacian', 'fd8noint_laplacian', 'fd4lag4_laplacian',
                          'fd4noint_hessian', 'fd6noint_hessian', 'fd8noint_hessian', 'm2q8_hessian']
        
        # create the metadata files for each interpolation method if they do not already exist.
        for interp_method in interp_methods:
            # pickled file for saving the interpolation coefficient lookup table.
            pickle_filename = f'{interp_method}_lookup_table.pickle'
            pickle_file_prod = self.pickle_dir.joinpath(pickle_filename)
            pickle_file_local = self.pickle_dir_local.joinpath(pickle_filename)

            # check if the pickled file is accessible.
            if not (pickle_file_prod.is_file() or pickle_file_local.is_file()) or rewrite_metadata:
                # create the interpolation coefficient lookup table.
                tmp_lookup_table = self.createInterpolationLookupTable(interp_method)

                # save tmp_lookup_table to a pickled file.
                with open(pickle_file_local, 'wb') as pickled_lookup_table:
                    dill.dump(tmp_lookup_table, pickled_lookup_table)
        
        # read in the interpolation lookup table for sint. the interpolation lookup tables are only read from
        # the get_points_getdata function.
        if sint != 'none' and read_metadata:
            # pickled interpolation coefficient lookup table.
            self.lookup_table = self.read_pickle_file(f'{sint}_lookup_table.pickle')
            
            # read in the field interpolation lookup table that is used in the calculation of other interpolation methods.
            if sint in ['fd4lag4_gradient', 'm1q4_gradient', 'm2q8_gradient',
                        'fd4lag4_laplacian',
                        'm2q8_hessian']:
                # convert sint to the needed field interpolation name.
                sint_name = sint.split('_')[0].replace('fd4', '')
                
                # pickled interpolation coefficient lookup table.
                self.field_lookup_table = self.read_pickle_file(f'{sint_name}_lookup_table.pickle')
                    
                # read in the gradient coefficient lookup table that is used in the calculation of the m2q8 spline hessian.
                if sint == 'm2q8_hessian':
                    # convert sint to the needed gradient interpolation name.
                    sint_name = sint.replace('_hessian', '_gradient')
                    
                    # pickled gradient coefficient lookup table.
                    self.gradient_lookup_table = self.read_pickle_file(f'{sint_name}_lookup_table.pickle')
            # read in the laplacian interpolation lookup table that is used in the calculation of other interpolation methods.
            elif sint in ['fd4noint_hessian', 'fd6noint_hessian', 'fd8noint_hessian']:
                # convert sint to the needed gradient interpolation name.
                sint_name = sint.replace('_hessian', '_laplacian')
                
                # pickled laplacian coefficient lookup table.
                self.laplacian_lookup_table = self.read_pickle_file(f'{sint_name}_lookup_table.pickle')
                
    def init_interpolation_cube_size_lookup_table(self, metadata, sint = 'none', sint_specified = 'none'):
        """
        pickled interpolation cube sizes table.
        """
        # the interpolation cube size indices are only read when called from the init_constants function.
        interp_cube_sizes = {spatial_method['code']: spatial_method['bucketIndices'] for spatial_method in metadata['spatial_methods']}

        # use sint_specified when one of the 'z_linear*' step-down methods is being used.
        sint_cube_size = sint_specified if 'z_linear' in sint else sint
        # the bucket size is the same for all spatial operators, so we only use the spatial method portion of 'sint' and 'sint_specified'.
        sint_cube_size = sint_cube_size.split('_')[0]

        # lookup the interpolation cube size indices.
        self.cube_min_index, self.cube_max_index = interp_cube_sizes[sint_cube_size]

        # get the interpolation bucket dimension length for sint_cube_size. used for defining the zeros plane at z = 0 (ground) for calculating
        # the w-component of velocity at z-points queried in the range [dz / 2, dz) of the 'sabl2048*' datasets. 
        self.cube_dim = self.cube_min_index + self.cube_max_index + 1

        # convert self.cube_min_index and self.cube_max_index for defining the buckets of the 'z_linear*' step-down methods.
        if sint == 'z_linear':
            self.cube_min_index = np.array([self.cube_min_index, self.cube_min_index, 0])
            self.cube_max_index = np.array([self.cube_max_index, self.cube_max_index, 1])
        elif sint in ['z_linear_gradient', 'z_linear_laplacian', 'z_linear_hessian']:
            self.cube_min_index = np.array([self.cube_min_index, self.cube_min_index, 0])
            self.cube_max_index = np.array([self.cube_max_index, self.cube_max_index, 2])
    
    def init_constants(self, query_type, var, var_offsets, timepoint, timepoint_original, sint, sint_specified, tint, option,
                       num_values_per_datapoint, c):
        """
        initialize the constants.
        """
        self.var = var
        self.var_offsets = var_offsets
        # convert the timepoint to [hour, minute, simulation number] for the windfarm datasets.
        if self.dataset_title == 'diurnal_windfarm':
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
        self.decimals = c['decimals']
        self.chunk_size = get_dataset_chunk_size(self.metadata, self.dataset_title, self.var)
        self.query_type = query_type
        
        # set the byte order for reading the data from the files.
        self.dt = np.dtype(np.float32)
        self.dt = self.dt.newbyteorder('<')
        
        # retrieve the dimension offsets.
        self.grid_offsets = get_dataset_grid_offsets(self.metadata, self.dataset_title, self.var_offsets, self.var)
        
        # retrieve the coor offsets.
        self.coor_offsets = get_dataset_coordinate_offsets(self.metadata, self.dataset_title, self.var_offsets, self.var)
        
        # set the dataset name to be used in the cutout hdf5 file.
        self.dataset_name = self.var + '_' + str(self.timepoint_original).zfill(4)
        
        # retrieve the list of datasets processed by the giverny code.
        giverny_datasets = get_giverny_datasets()
        
        if self.dataset_title in giverny_datasets:
            if query_type == 'getdata':
                # initialize the interpolation cube size lookup table.
                self.init_interpolation_cube_size_lookup_table(self.metadata, self.sint, self.sint_specified)
                
                # defining the zeros plane at z = 0 (ground) for calculating the w-component of velocity at z-points queried in the range [dz / 2, dz) of the
                # 'sabl2048*' datasets.
                bucket_zero_plane = np.zeros((self.cube_dim, self.cube_dim, self.num_values_per_datapoint), dtype = np.float32)
                # interpolate function variables.
                self.interpolate_vars = [self.cube_min_index, self.cube_max_index, self.sint, self.sint_specified, self.spacing, bucket_zero_plane, self.lookup_N]
                
                # getData variables.
                self.getdata_vars = [self.dataset_title, self.num_values_per_datapoint, self.N, self.chunk_size]
            
            # open the zarr store for reading.
            self.zarr_filepath = get_dataset_filepath(self.metadata, self.dataset_title)
            self.zarr_store = self.open_zarr_file([self.zarr_filepath, self.var, self.dt])
    
    """
    interpolation functions.
    """
    def createInterpolationLookupTable(self, sint):
        """
        generate interpolation lookup table.
        """
        if sint in ['fd4noint_gradient', 'fd6noint_gradient', 'fd8noint_gradient',
                    'fd4noint_laplacian', 'fd6noint_laplacian', 'fd8noint_laplacian',
                    'fd4noint_hessian', 'fd6noint_hessian', 'fd8noint_hessian']:
            lookup_table = self.getInterpolationCoefficients(sint)
        else:
            lookup_table = []
            
            frac = np.linspace(0, 1 - 1 / self.lookup_N, self.lookup_N)
            for fp in frac:
                lookup_table.append(self.getInterpolationCoefficients(sint, fp))

        return np.array(lookup_table)
    
    def getInterpolationCoefficients(self, sint, fr = 0.0):
        """
        get interpolation coefficients.
        """
        if sint == 'fd4noint_hessian':
            g = np.array([-1.0 / 48.0,
                          1.0 / 48.0,
                          -1.0 / 48.0,
                          1.0 / 48.0,
                          1.0 / 3.0,
                          -1.0 / 3.0,
                          1.0 / 3.0,
                          -1.0 / 3.0])
        elif sint == 'fd6noint_hessian':
            g = np.array([1.0 / 360.0,
                          -1.0 / 360.0,
                          1.0 / 360.0,
                          -1.0 / 360.0,
                          -3.0 / 80.0,
                          3.0 / 80.0,
                          -3.0 / 80.0,
                          3.0 / 80.0,
                          3.0 / 8.0,
                          -3.0 / 8.0,
                          3.0 / 8.0,
                          -3.0 / 8.0])
        elif sint == 'fd8noint_hessian':
            g = np.array([-1.0 / 2240.0,
                          1.0 / 2240.0,
                          -1.0 / 2240.0,
                          1.0 / 2240.0,
                          2.0 / 315.0,
                          -2.0 / 315.0,
                          2.0 / 315.0,
                          -2.0 / 315.0,
                          -1.0 / 20.0,
                          1.0 / 20.0,
                          -1.0 / 20.0,
                          1.0 / 20.0,
                          14.0 / 35.0,
                          -14.0 / 35.0,
                          14.0 / 35.0,
                          -14.0 / 35.0])
        elif sint == 'm2q8_hessian':
            g = np.zeros(8)
            g[0] = fr * (fr * ((8.0 / 9.0) * fr - 7.0 / 5.0) + 1.0 / 2.0) + 1.0 / 90.0
            g[1] = fr * (fr * (-115.0 / 18.0 * fr + 61.0 / 6.0) - 217.0 / 60.0) - 3.0 / 20.0
            g[2] = fr * (fr * ((39.0 / 2.0) * fr - 153.0 / 5.0) + 189.0 / 20.0) + 3.0 / 2.0
            g[3] = fr * (fr * (-295.0 / 9.0 * fr + 50) - 13) - 49.0 / 18.0
            g[4] = fr * (fr * ((295.0 / 9.0) * fr - 145.0 / 3.0) + 34.0 / 3.0) + 3.0 / 2.0
            g[5] = fr * (fr * (-39.0 / 2.0 * fr + 279.0 / 10.0) - 27.0 / 4.0) - 3.0 / 20.0
            g[6] = fr * (fr * ((115.0 / 18.0) * fr - 9) + 49.0 / 20.0) + 1.0 / 90.0
            g[7] = fr * (fr * (-8.0 / 9.0 * fr + 19.0 / 15.0) - 11.0 / 30.0)
        elif sint == 'fd4noint_laplacian':
            g = np.array([-1.0 / 12.0,
                          4.0 / 3.0,
                          -15.0 / 6.0,
                          4.0 / 3.0,
                          -1.0 / 12.0])
        elif sint == 'fd6noint_laplacian':
            g = np.array([1.0 / 90.0,
                          -3.0 / 20.0,
                          3.0 / 2.0,
                          -49.0 / 18.0,
                          3.0 / 2.0,
                          -3.0 / 20.0,
                          1.0 / 90.0])
        elif sint == 'fd8noint_laplacian':
            g = np.array([9.0 / 3152.0,
                          -104.0 / 8865.0,
                          -207.0 / 2955.0,
                          792.0 / 591.0,
                          -35777.0 / 14184.0,
                          792.0 / 591.0,
                          -207.0 / 2955.0,
                          -104.0 / 8865.0,
                          9.0 / 3152.0])
        elif sint == 'fd4noint_gradient':
            g = np.array([1.0 / 12.0,
                          -2.0 / 3.0, 
                          0.0, 
                          2.0 / 3.0,
                          -1.0 / 12.0])
        elif sint == 'fd6noint_gradient':
            g = np.array([-1.0 / 60.0,
                          3.0 / 20.0,
                          -3.0 / 4.0, 
                          0.0, 
                          3.0 / 4.0,
                          -3.0 / 20.0,
                          1.0 / 60.0])
        elif sint == 'fd8noint_gradient':
            g = np.array([1.0 / 280.0,
                          -4.0 / 105.0,
                          1.0 / 5.0,
                          -4.0 / 5.0, 
                          0.0, 
                          4.0 / 5.0,
                          -1.0 / 5.0,
                          4.0 / 105.0,
                          -1.0 / 280.0])
        elif sint in ['fd4lag4_gradient', 'fd4lag4_laplacian']:
            wN = [1.,-3.,3.,-1.]
            B = np.array([0,1.,0,0])
            # calculate weights if fr>0, and insert into g.
            if (fr>0):
                s = 0
                for n in range(4):
                    B[n] = wN[n]/(fr-n+1)
                    s += B[n]
                for n in range(4):
                    B[n] = B[n]/s

            if sint == 'fd4lag4_gradient':
                A = [1.0 / 12.0,
                     -2.0 / 3.0, 
                     0.0, 
                     2.0 / 3.0,
                     -1.0 / 12.0]
            elif sint == 'fd4lag4_laplacian':
                A = [-1.0 / 12.0,
                     4.0 / 3.0,
                     -15.0 / 6.0,
                     4.0 / 3.0,
                     -1.0 / 12.0]

            g = np.zeros(8)
            g[0] = B[0]*A[0]
            g[1] = B[0]*A[1] + B[1]*A[0]
            g[2] = B[0]*A[2] + B[1]*A[1] + B[2]*A[0]
            g[3] = B[0]*A[3] + B[1]*A[2] + B[2]*A[1] + B[3]*A[0] 
            g[4] = B[0]*A[4] + B[1]*A[3] + B[2]*A[2] + B[3]*A[1] 
            g[5] = B[1]*A[4] + B[2]*A[3] + B[3]*A[2] 
            g[6] = B[2]*A[4] + B[3]*A[3] 
            g[7] = B[3]*A[4]
        elif sint == 'm1q4_gradient':
            g = np.zeros(4)
            g[0] = fr * (-3.0 / 2.0 * fr + 2) - 1.0 / 2.0
            g[1] = fr * ((9.0 / 2.0) * fr - 5)
            g[2] = fr * (-9.0 / 2.0 * fr + 4) + 1.0 / 2.0
            g[3] = fr * ((3.0 / 2.0) * fr - 1)
        elif sint == 'm2q8_gradient':
            g = np.zeros(8)
            g[0] = fr * (fr * (fr * ((2.0 / 9.0) * fr - 7.0 / 15.0) + 1.0 / 4.0) + 1.0 / 90.0) - 1.0 / 60.0
            g[1] = fr * (fr * (fr * (-115.0 / 72.0 * fr + 61.0 / 18.0) - 217.0 / 120.0) - 3.0 / 20.0) + 3.0 / 20.0
            g[2] = fr * (fr * (fr * ((39.0 / 8.0) * fr - 51.0 / 5.0) + 189.0 / 40.0) + 3.0 / 2.0) - 3.0 / 4.0
            g[3] = fr * (fr * (fr * (-295.0 / 36.0 * fr + 50.0 / 3.0) - 13.0 / 2.0) - 49.0 / 18.0)
            g[4] = fr * (fr * (fr * ((295.0 / 36.0) * fr - 145.0 / 9.0) + 17.0 / 3.0) + 3.0 / 2.0) + 3.0 / 4.0
            g[5] = fr * (fr * (fr * (-39.0 / 8.0 * fr + 93.0 / 10.0) - 27.0 / 8.0) - 3.0 / 20.0) - 3.0 / 20.0
            g[6] = fr * (fr * (fr * ((115.0 / 72.0) * fr - 3) + 49.0 / 40.0) + 1.0 / 90.0) + 1.0 / 60.0
            g[7] = fr**2 * (fr * (-2.0 / 9.0 * fr + 19.0 / 45.0) - 11.0 / 60.0)
        elif sint == 'm1q4':
            # define the weights for m1q4 spline interpolation.
            g = np.zeros(4)
            g[0] = fr * (fr * (-1.0 / 2.0 * fr + 1) - 1.0 / 2.0)
            g[1] = fr**2 * ((3.0 / 2.0) * fr - 5.0 / 2.0) + 1
            g[2] = fr * (fr * (-3.0 / 2.0 * fr + 2) + 1.0 / 2.0)
            g[3] = fr**2 * ((1.0 / 2.0) * fr - 1.0 / 2.0)
        elif sint == 'm2q8':
            # define the weights for m2q8 spline interpolation.
            g = np.zeros(8)  
            g[0] = fr * (fr * (fr * (fr * ((2.0 / 45.0) * fr - 7.0 / 60.0) + 1.0 / 12.0) + 1.0 / 180.0) - 1.0 / 60.0)
            g[1] = fr * (fr * (fr * (fr * (-23.0 / 72.0 * fr + 61.0 / 72.0) - 217.0 / 360.0) - 3.0 / 40.0) + 3.0 / 20.0)
            g[2] = fr * (fr * (fr * (fr * ((39.0 / 40.0) * fr - 51.0 / 20.0) + 63.0 / 40.0) + 3.0 / 4.0) - 3.0 / 4.0)
            g[3] = fr**2 * (fr * (fr * (-59.0 / 36.0 * fr + 25.0 / 6.0) - 13.0 / 6.0) - 49.0 / 36.0) + 1
            g[4] = fr * (fr * (fr * (fr * ((59.0 / 36.0) * fr - 145.0 / 36.0) + 17.0 / 9.0) + 3.0 / 4.0) + 3.0 / 4.0)
            g[5] = fr * (fr * (fr * (fr * (-39.0 / 40.0 * fr + 93.0 / 40.0) - 9.0 / 8.0) - 3.0 / 40.0) - 3.0 / 20.0)
            g[6] = fr * (fr * (fr * (fr * ((23.0 / 72.0) * fr - 3.0 / 4.0) + 49.0 / 120.0) + 1.0 / 180.0) + 1.0 / 60.0)
            g[7] = fr**3 * (fr * (-2.0 / 45.0 * fr + 19.0 / 180.0) - 11.0 / 180.0)
        else:
            # define the weights for the different lagrangian interpolation methods.
            if sint == 'lag4':
                wN = [1.,-3.,3.,-1.]
                g = np.array([0,1.,0,0])
                # weight index.
                w_index = 1
            elif sint == 'lag6':
                wN = [1.,-5.,10.,-10.,5.,-1.]
                g = np.array([0,0,1.,0,0,0])
                # weight index.
                w_index = 2
            elif sint == 'lag8':
                wN = [1.,-7.,21.,-35.,35.,-21.,7.,-1.]
                g = np.array([0,0,0,1.,0,0,0,0])
                # weight index.
                w_index = 3

            # calculate weights if fr>0, and insert into g.
            if (fr>0):
                num_points = len(g)

                s = 0
                for n in range(num_points):
                    g[n] = wN[n] / (fr - n + w_index)
                    s += g[n]

                for n in range(num_points):
                    g[n] = g[n] / s

        return g
    
    def spatial_interpolate(self, p, u, u_info, interpolate_vars):
        """
        spatial interpolating functions to compute the kernel, extract subcube and convolve.
        
        vars:
         - p is an np.array(3) containing the three coordinates.
        """
        # assign the local variables.
        cube_min_index, cube_max_index, sint, sint_specified, spacing, bucket_zero_plane, lookup_N = interpolate_vars
        dx, dy, dz = spacing
        
        """
        'none' field interpolation.
        """
        def none():
            ix = np.floor(p + 0.5).astype(np.int32)
            
            return np.array(u[ix[2], ix[1], ix[0], :])
        
        """
        field interpolations.
        """
        def lag_spline():
            ix = p.astype(np.int32)
            fr = p - ix
            
            # get the coefficients.
            gx = self.lookup_table[int(lookup_N * fr[0])]
            gy = self.lookup_table[int(lookup_N * fr[1])]
            gz = self.lookup_table[int(lookup_N * fr[2])]
            
            # create the 3d kernel from the outer product of the 1d kernels.
            gk = np.einsum('i,j,k', gz, gy, gx)

            return np.einsum('ijk,ijkl->l', gk, u)
        
        """
        field linear interpolation (step-down for 'lag8', 'lag6', 'lag4', 'm2q8', 'm1q4').
        """
        def x_linear():
            ix = p.astype(np.int32)
            fr = p - ix
            
            # the first column of buckets_info is x_bottom_flag: bottom bucket flag.
            x_bottom_flag = u_info[0]
            
            # get the coefficients.
            gy = self.lookup_table[int(lookup_N * fr[1])]
            gz = self.lookup_table[int(lookup_N * fr[2])]
            
            # create the 2d kernel from the outer product of the 1d kernels.
            gk = np.einsum('j,k', gz, gy)
            
            if x_bottom_flag == 'zero_ground':
                # get the bucket x-plane above the point. for this particular case, the "top" boundary is math.floor(p[0]) because this handles 
                # x-points between the ground (x = 0) and the 1st x-grid point (x = dx). 
                x_top = math.floor(p[0])
                ux_top = u[:, :, x_top, :]
                
                # get the bucket x-plane below the point.
                ux_bottom = bucket_zero_plane
            else:
                # get the bucket x-plane above the point.
                x_top = math.ceil(p[0])
                ux_top = u[:, :, x_top, :]
            
                # get the bucket x-plane below the point.
                x_bottom = math.floor(p[0])
                ux_bottom = u[:, :, x_bottom, :]
            
            # 2d interpolation at the x-axis point above p.
            fn_top = np.einsum('jk,jkl->l', gk, ux_top)

            # 2d interpolation at the x-axis point below p.
            fn_bottom  = np.einsum('jk,jkl->l', gk, ux_bottom)
            
            return fn_bottom + (fn_top - fn_bottom) * fr[0]
        
        def y_linear():
            ix = p.astype(np.int32)
            fr = p - ix
            
            # the first column of buckets_info is y_bottom_flag: bottom bucket flag.
            y_bottom_flag = u_info[0]
            
            # get the coefficients.
            gx = self.lookup_table[int(lookup_N * fr[0])]
            gz = self.lookup_table[int(lookup_N * fr[2])]
            
            # create the 2d kernel from the outer product of the 1d kernels.
            gk = np.einsum('i,k', gz, gx)
            
            if y_bottom_flag == 'zero_ground':
                # get the bucket y-plane above the point. for this particular case, the "top" boundary is math.floor(p[1]) because this handles 
                # y-points between the ground (y = 0) and the 1st z-grid point (y = dy). 
                y_top = math.floor(p[1])
                uy_top = u[:, y_top, :, :]
                
                # get the bucket y-plane below the point.
                uy_bottom = bucket_zero_plane
            else:
                # get the bucket y-plane above the point.
                y_top = math.ceil(p[1])
                uy_top = u[:, y_top, :, :]
            
                # get the bucket y-plane below the point.
                y_bottom = math.floor(p[1])
                uy_bottom = u[:, y_bottom, :, :]
            
            # 2d interpolation at the y-axis point above p.
            fn_top = np.einsum('ik,ikl->l', gk, uy_top)

            # 2d interpolation at the y-axis point below p.
            fn_bottom  = np.einsum('ik,ikl->l', gk, uy_bottom)
            
            return fn_bottom + (fn_top - fn_bottom) * fr[1]
        
        def z_linear():
            ix = p.astype(np.int32)
            fr = p - ix
            
            # the first column of buckets_info is z_bottom_flag: bottom bucket flag.
            z_bottom_flag = u_info[0]
            
            # get the coefficients.
            gx = self.lookup_table[int(lookup_N * fr[0])]
            gy = self.lookup_table[int(lookup_N * fr[1])]
            
            # create the 2d kernel from the outer product of the 1d kernels.
            gk = np.einsum('i,j', gy, gx)
            
            if z_bottom_flag == 'zero_ground':
                # get the bucket z-plane above the point. for this particular case, the "top" boundary is math.floor(p[2]) because this handles 
                # z-points between the ground (z = 0) and the 1st z-grid point (z = dz). applies to the 'velocity_w' variable of the 'sabl2048*' datasets. 
                z_top = math.floor(p[2])
                uz_top = u[z_top, :, :, :]
                
                # get the bucket z-plane below the point.
                uz_bottom = bucket_zero_plane
            else:
                # get the bucket z-plane above the point.
                z_top = math.ceil(p[2])
                uz_top = u[z_top, :, :, :]
            
                # get the bucket z-plane below the point.
                z_bottom = math.floor(p[2])
                uz_bottom = u[z_bottom, :, :, :]
            
            # 2d interpolation at the z-axis point above p.
            fn_top = np.einsum('ij,ijl->l', gk, uz_top)

            # 2d interpolation at the z-axis point below p.
            fn_bottom  = np.einsum('ij,ijl->l', gk, uz_bottom)
            
            return fn_bottom + (fn_top - fn_bottom) * fr[2]
        
        """
        gradient linear region finite differences.
        """
        def x_linear_gradient():
            ix = p.astype(np.int32)
            # diagonal coefficients.
            fd_coeff = self.lookup_table
            
            # the 3 columns of buckets_info are x_bottom: bottom bucket index, x_top: top bucket index, and x_divisor: number of grid points to divide by.
            x_bottom, x_top, x_divisor = u_info[:3]
            
            # diagonal components.
            component_y = u[ix[2], ix[1] - cube_min_index[1] : ix[1] + cube_max_index[1], ix[0], :]
            component_z = u[ix[2] - cube_min_index[2] : ix[2] + cube_max_index[2], ix[1], ix[0], :]
            # get the x-gridpoint above the specified point. 
            component_x_top = u[ix[2], ix[1], ix[0] + x_top, :]
            # get the x-gridpoint below the specified point. if x_bottom is 'zero_ground' then the x-gridpoint below the specified point is set to 0.
            component_x_bottom = 0.0 if x_bottom == 'zero_ground' else u[ix[2], ix[1], ix[0] + x_bottom, :]
            # the linear dvdx divisor equals the spacing between the top and bottom x-gridpoints.
            dvdx_divisor = x_divisor * dx

            dvdx = (component_x_top - component_x_bottom) / dvdx_divisor
            dvdy = np.inner(fd_coeff, component_y.T) / dy
            dvdz = np.inner(fd_coeff, component_z.T) / dz
            
            return np.stack((dvdx, dvdy, dvdz), axis = 1).flatten()
        
        def y_linear_gradient():
            ix = p.astype(np.int32)
            # diagonal coefficients.
            fd_coeff = self.lookup_table
            
            # the 3 columns of buckets_info are y_bottom: bottom bucket index, y_top: top bucket index, and y_divisor: number of grid points to divide by.
            y_bottom, y_top, y_divisor = u_info[:3]
            
            # diagonal components.
            component_x = u[ix[2], ix[1], ix[0] - cube_min_index[0] : ix[0] + cube_max_index[0], :]
            component_z = u[ix[2] - cube_min_index[2] : ix[2] + cube_max_index[2], ix[1], ix[0], :]
            # get the y-gridpoint above the specified point. 
            component_y_top = u[ix[2], ix[1] + y_top, ix[0], :]
            # get the y-gridpoint below the specified point. if y_bottom is 'zero_ground' then the y-gridpoint below the specified point is set to 0.
            component_y_bottom = 0.0 if y_bottom == 'zero_ground' else u[ix[2], ix[1] + y_bottom, ix[0], :]
            # the linear dvdy divisor equals the spacing between the top and bottom y-gridpoints.
            dvdy_divisor = y_divisor * dy

            dvdx = np.inner(fd_coeff, component_x.T) / dx
            dvdy = (component_y_top - component_y_bottom) / dvdy_divisor
            dvdz = np.inner(fd_coeff, component_z.T) / dz
            
            return np.stack((dvdx, dvdy, dvdz), axis = 1).flatten()
        
        def z_linear_gradient():
            ix = p.astype(np.int32)
            # diagonal coefficients.
            fd_coeff = self.lookup_table
            
            # the 3 columns of buckets_info are z_bottom: bottom bucket index, z_top: top bucket index, and z_divisor: number of grid points to divide by.
            z_bottom, z_top, z_divisor = u_info[:3]
            
            # diagonal components.
            component_x = u[ix[2], ix[1], ix[0] - cube_min_index[0] : ix[0] + cube_max_index[0], :]
            component_y = u[ix[2], ix[1] - cube_min_index[1] : ix[1] + cube_max_index[1], ix[0], :]
            # get the z-gridpoint above the specified point. 
            component_z_top = u[ix[2] + z_top, ix[1], ix[0], :]
            # get the z-gridpoint below the specified point. if z_bottom is 'zero_ground' then the z-gridpoint below the specified point is set to 0.
            component_z_bottom = 0.0 if z_bottom == 'zero_ground' else u[ix[2] + z_bottom, ix[1], ix[0], :]
            # the linear dvdz divisor equals the spacing between the top and bottom z-gridpoints.
            dvdz_divisor = z_divisor * dz

            dvdx = np.inner(fd_coeff, component_x.T) / dx
            dvdy = np.inner(fd_coeff, component_y.T) / dy
            dvdz = (component_z_top - component_z_bottom) / dvdz_divisor
            
            return np.stack((dvdx, dvdy, dvdz), axis = 1).flatten()
            
        """
        laplacian linear region finite differences.
        """
        def x_linear_laplacian():
            ix = p.astype(np.int32)
            # diagonal coefficients.
            fd_coeff = self.lookup_table
            
            # the 3 columns of buckets_info are x_bottom: bottom bucket index, x_top: top bucket index, and x_divisor: number of grid points to divide by.
            x_bottom, x_top, x_divisor = u_info[:3]
            
            # diagonal components.
            component_y = u[ix[2], ix[1] - cube_min_index[1] : ix[1] + cube_max_index[1], ix[0], :]
            component_z = u[ix[2] - cube_min_index[2] : ix[2] + cube_max_index[2], ix[1], ix[0], :]
            # get the x-gridpoint above the specified point. 
            component_x_top = u[ix[2], ix[1], ix[0] + x_top, :]
            # get the x-gridpoint below the specified point. if x_bottom is 'zero_ground' then the x-gridpoint below the specified point is set to 0.
            component_x_bottom = 0.0 if x_bottom == 'zero_ground' else u[ix[2], ix[1], ix[0] + x_bottom, :]
            # the linear dvdx divisor equals the spacing between the top and bottom x-gridpoints.
            dvdx_divisor = x_divisor * dx

            dvdx = (component_x_top - component_x_bottom) / dvdx_divisor
            dvdy = np.inner(fd_coeff, component_y.T) / dy / dy
            dvdz = np.inner(fd_coeff, component_z.T) / dz / dz
            
            return dvdx + dvdy + dvdz
        
        def y_linear_laplacian():
            ix = p.astype(np.int32)
            # diagonal coefficients.
            fd_coeff = self.lookup_table
            
            # the 3 columns of buckets_info are y_bottom: bottom bucket index, y_top: top bucket index, and y_divisor: number of grid points to divide by.
            y_bottom, y_top, y_divisor = u_info[:3]
            
            # diagonal components.
            component_x = u[ix[2], ix[1], ix[0] - cube_min_index[0] : ix[0] + cube_max_index[0], :]
            component_z = u[ix[2] - cube_min_index[2] : ix[2] + cube_max_index[2], ix[1], ix[0], :]
            # get the y-gridpoint above the specified point. 
            component_y_top = u[ix[2], ix[1] + y_top, ix[0], :]
            # get the y-gridpoint below the specified point. if y_bottom is 'zero_ground' then the y-gridpoint below the specified point is set to 0.
            component_y_bottom = 0.0 if y_bottom == 'zero_ground' else u[ix[2], ix[1] + y_bottom, ix[0], :]
            # the linear dvdy divisor equals the spacing between the top and bottom y-gridpoints.
            dvdy_divisor = y_divisor * dy

            dvdx = np.inner(fd_coeff, component_x.T) / dx / dx
            dvdy = (component_y_top - component_y_bottom) / dvdy_divisor
            dvdz = np.inner(fd_coeff, component_z.T) / dz / dz
            
            return dvdx + dvdy + dvdz
        
        def z_linear_laplacian():
            ix = p.astype(np.int32)
            # diagonal coefficients.
            fd_coeff = self.lookup_table
            
            # the 3 columns of buckets_info are z_bottom: bottom bucket index, z_top: top bucket index, and z_divisor: number of grid points to divide by.
            z_bottom, z_top, z_divisor = u_info[:3]
            
            # diagonal components.
            component_x = u[ix[2], ix[1], ix[0] - cube_min_index[0] : ix[0] + cube_max_index[0], :]
            component_y = u[ix[2], ix[1] - cube_min_index[1] : ix[1] + cube_max_index[1], ix[0], :]
            # get the z-gridpoint above the specified point. 
            component_z_top = u[ix[2] + z_top, ix[1], ix[0], :]
            # get the z-gridpoint below the specified point. if z_bottom is 'zero_ground' then the z-gridpoint below the specified point is set to 0.
            component_z_bottom = 0.0 if z_bottom == 'zero_ground' else u[ix[2] + z_bottom, ix[1], ix[0], :]
            # the linear dvdz divisor equals the spacing between the top and bottom z-gridpoints.
            dvdz_divisor = z_divisor * dz

            dvdx = np.inner(fd_coeff, component_x.T) / dx / dx
            dvdy = np.inner(fd_coeff, component_y.T) / dy / dy
            dvdz = (component_z_top - component_z_bottom) / dvdz_divisor
            
            return dvdx + dvdy + dvdz
        
        """
        hessian linear region finite differences.
        """
        def x_linear_hessian():
            ix = p.astype(np.int32)
            # diagonal coefficients.
            fd_coeff_laplacian = self.laplacian_lookup_table
            # off-diagonal coefficients.
            fd_coeff_hessian = self.lookup_table
            
            # the 3 columns of buckets_info are x_bottom: bottom bucket index, x_middle: middle bucket index, and x_top: top bucket index.
            x_bottom, x_middle, x_top = u_info[:3]
            
            # diagonal components.
            component_y = u[ix[2], ix[1] - cube_min_index[1] : ix[1] + cube_max_index[1], ix[0], :]
            component_z = u[ix[2] - cube_min_index[2] : ix[2] + cube_max_index[2], ix[1], ix[0], :]

            ujj = np.inner(fd_coeff_laplacian, component_y.T) / dy / dy
            ukk = np.inner(fd_coeff_laplacian, component_z.T) / dz / dz

            # off-diagonal components.
            if sint_specified == 'fd4noint_hessian':
                component_yz = np.array([u[ix[2]+2,ix[1]+2,ix[0],:],u[ix[2]+2,ix[1]-2,ix[0],:],u[ix[2]-2,ix[1]-2,ix[0],:],u[ix[2]-2,ix[1]+2,ix[0],:],
                                         u[ix[2]+1,ix[1]+1,ix[0],:],u[ix[2]+1,ix[1]-1,ix[0],:],u[ix[2]-1,ix[1]-1,ix[0],:],u[ix[2]-1,ix[1]+1,ix[0],:]])
            elif sint_specified == 'fd6noint_hessian':
                component_yz = np.array([u[ix[2]+3,ix[1]+3,ix[0],:],u[ix[2]+3,ix[1]-3,ix[0],:],u[ix[2]-3,ix[1]-3,ix[0],:],u[ix[2]-3,ix[1]+3,ix[0],:],
                                         u[ix[2]+2,ix[1]+2,ix[0],:],u[ix[2]+2,ix[1]-2,ix[0],:],u[ix[2]-2,ix[1]-2,ix[0],:],u[ix[2]-2,ix[1]+2,ix[0],:],
                                         u[ix[2]+1,ix[1]+1,ix[0],:],u[ix[2]+1,ix[1]-1,ix[0],:],u[ix[2]-1,ix[1]-1,ix[0],:],u[ix[2]-1,ix[1]+1,ix[0],:]])
            elif sint_specified == 'fd8noint_hessian':
                component_yz = np.array([u[ix[2]+4,ix[1]+4,ix[0],:],u[ix[2]+4,ix[1]-4,ix[0],:],u[ix[2]-4,ix[1]-4,ix[0],:],u[ix[2]-4,ix[1]+4,ix[0],:],
                                         u[ix[2]+3,ix[1]+3,ix[0],:],u[ix[2]+3,ix[1]-3,ix[0],:],u[ix[2]-3,ix[1]-3,ix[0],:],u[ix[2]-3,ix[1]+3,ix[0],:],
                                         u[ix[2]+2,ix[1]+2,ix[0],:],u[ix[2]+2,ix[1]-2,ix[0],:],u[ix[2]-2,ix[1]-2,ix[0],:],u[ix[2]-2,ix[1]+2,ix[0],:],
                                         u[ix[2]+1,ix[1]+1,ix[0],:],u[ix[2]+1,ix[1]-1,ix[0],:],u[ix[2]-1,ix[1]-1,ix[0],:],u[ix[2]-1,ix[1]+1,ix[0],:]])
            
            ujk = np.inner(fd_coeff_hessian, component_yz.T) / dy / dz
            
            if x_bottom == 'zero_ground':
                # sets all values at x_bottom (x = 0) equal to 0 because this handles the boundary condition gridpoint at x = 0.
                uii = (u[ix[2], ix[1], x_top, :] - (2 * u[ix[2], ix[1], x_middle, :])) / (dx * dx)

                uij = (u[ix[2], ix[1] + 1, x_top, :] - u[ix[2], ix[1] - 1, x_top, :]) / (4 * dx * dy)
                uik = (u[ix[2] + 1, ix[1], x_top, :] - u[ix[2] - 1, ix[1], x_top, :]) / (4 * dx * dz)
            else:
                uii = (u[ix[2], ix[1], x_top, :] - (2 * u[ix[2], ix[1], x_middle, :]) + u[ix[2], ix[1], x_bottom, :]) / (dx * dx)

                uij = (u[ix[2], ix[1] + 1, x_top, :] - u[ix[2], ix[1] - 1, x_top, :] - u[ix[2], ix[1] + 1, x_bottom, :] + u[ix[2], ix[1] - 1, x_bottom, :]) / (4 * dx * dy)
                uik = (u[ix[2] + 1, ix[1], x_top, :] - u[ix[2] - 1, ix[1], x_top, :] - u[ix[2] + 1, ix[1], x_bottom, :] + u[ix[2] - 1, ix[1], x_bottom, :]) / (4 * dx * dz)
            
            return np.stack((uii,uij,uik,ujj,ujk,ukk), axis = 1).flatten()
        
        def y_linear_hessian():
            ix = p.astype(np.int32)
            # diagonal coefficients.
            fd_coeff_laplacian = self.laplacian_lookup_table
            # off-diagonal coefficients.
            fd_coeff_hessian = self.lookup_table
            
            # the 3 columns of buckets_info are y_bottom: bottom bucket index, y_middle: middle bucket index, and y_top: top bucket index.
            y_bottom, y_middle, y_top = u_info[:3]
            
            # diagonal components.
            component_x = u[ix[2], ix[1], ix[0] - cube_min_index[0] : ix[0] + cube_max_index[0], :]
            component_z = u[ix[2] - cube_min_index[2] : ix[2] + cube_max_index[2], ix[1], ix[0], :]

            uii = np.inner(fd_coeff_laplacian, component_x.T) / dx / dx
            ukk = np.inner(fd_coeff_laplacian, component_z.T) / dz / dz

            # off-diagonal components.
            if sint_specified == 'fd4noint_hessian':
                component_xz = np.array([u[ix[2]+2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]-2,:],u[ix[2]+2,ix[1],ix[0]-2,:],
                                         u[ix[2]+1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]-1,:],u[ix[2]+1,ix[1],ix[0]-1,:]])
            elif sint_specified == 'fd6noint_hessian':
                component_xz = np.array([u[ix[2]+3,ix[1],ix[0]+3,:],u[ix[2]-3,ix[1],ix[0]+3,:],u[ix[2]-3,ix[1],ix[0]-3,:],u[ix[2]+3,ix[1],ix[0]-3,:],
                                         u[ix[2]+2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]-2,:],u[ix[2]+2,ix[1],ix[0]-2,:],
                                         u[ix[2]+1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]-1,:],u[ix[2]+1,ix[1],ix[0]-1,:]])
            elif sint_specified == 'fd8noint_hessian':
                component_xz = np.array([u[ix[2]+4,ix[1],ix[0]+4,:],u[ix[2]-4,ix[1],ix[0]+4,:],u[ix[2]-4,ix[1],ix[0]-4,:],u[ix[2]+4,ix[1],ix[0]-4,:],
                                         u[ix[2]+3,ix[1],ix[0]+3,:],u[ix[2]-3,ix[1],ix[0]+3,:],u[ix[2]-3,ix[1],ix[0]-3,:],u[ix[2]+3,ix[1],ix[0]-3,:],
                                         u[ix[2]+2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]-2,:],u[ix[2]+2,ix[1],ix[0]-2,:],
                                         u[ix[2]+1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]-1,:],u[ix[2]+1,ix[1],ix[0]-1,:]])
            
            uik = np.inner(fd_coeff_hessian, component_xz.T) / dx / dz
            
            if z_bottom == 'zero_ground':
                # sets all values at y_bottom (y = 0) equal to 0 because this handles the boundary condition gridpoint at y = 0.
                ujj = (u[ix[2], y_top, ix[0], :] - (2 * u[ix[2], y_middle, ix[0], :])) / (dy * dy)

                uij = (u[ix[2], y_top, ix[0] + 1, :] - u[ix[2], y_top, ix[0] - 1, :]) / (4 * dx * dy)
                ujk = (u[ix[2] + 1, y_top, ix[0], :] - u[ix[2] - 1, y_top, ix[0], :]) / (4 * dy * dz)
            else:
                ujj = (u[ix[2], y_top, ix[0], :] - (2 * u[ix[2], y_middle, ix[0], :]) + u[ix[2], y_bottom, ix[0], :]) / (dy * dy)

                uij = (u[ix[2], y_top, ix[0] + 1, :] - u[ix[2], y_top, ix[0] - 1, :] - u[ix[2], y_bottom, ix[0] + 1, :] + u[ix[2], y_bottom, ix[0] - 1, :]) / (4 * dx * dy)
                ujk = (u[ix[2] + 1, y_top, ix[0], :] - u[ix[2] - 1, y_top, ix[0], :] - u[ix[2] + 1, y_bottom, ix[0], :] + u[ix[2] - 1, y_bottom, ix[0], :]) / (4 * dy * dz)
            
            return np.stack((uii,uij,uik,ujj,ujk,ukk), axis = 1).flatten()
        
        def z_linear_hessian():
            ix = p.astype(np.int32)
            # diagonal coefficients.
            fd_coeff_laplacian = self.laplacian_lookup_table
            # off-diagonal coefficients.
            fd_coeff_hessian = self.lookup_table
            
            # the 3 columns of buckets_info are z_bottom: bottom bucket index, z_middle: middle bucket index, and z_top: top bucket index.
            z_bottom, z_middle, z_top = u_info[:3]
            
            # diagonal components.
            component_x = u[ix[2], ix[1], ix[0] - cube_min_index[0] : ix[0] + cube_max_index[0], :]
            component_y = u[ix[2], ix[1] - cube_min_index[1] : ix[1] + cube_max_index[1], ix[0], :]

            uii = np.inner(fd_coeff_laplacian, component_x.T) / dx / dx
            ujj = np.inner(fd_coeff_laplacian, component_y.T) / dy / dy

            # off-diagonal components.
            if sint_specified == 'fd4noint_hessian':
                component_xy = np.array([u[ix[2],ix[1]+2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]-2,:],u[ix[2],ix[1]+2,ix[0]-2,:],
                                         u[ix[2],ix[1]+1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]-1,:],u[ix[2],ix[1]+1,ix[0]-1,:]])
            elif sint_specified == 'fd6noint_hessian':
                component_xy = np.array([u[ix[2],ix[1]+3,ix[0]+3,:],u[ix[2],ix[1]-3,ix[0]+3,:],u[ix[2],ix[1]-3,ix[0]-3,:],u[ix[2],ix[1]+3,ix[0]-3,:],
                                         u[ix[2],ix[1]+2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]-2,:],u[ix[2],ix[1]+2,ix[0]-2,:],
                                         u[ix[2],ix[1]+1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]-1,:],u[ix[2],ix[1]+1,ix[0]-1,:]])
            elif sint_specified == 'fd8noint_hessian':
                component_xy = np.array([u[ix[2],ix[1]+4,ix[0]+4,:],u[ix[2],ix[1]-4,ix[0]+4,:],u[ix[2],ix[1]-4,ix[0]-4,:],u[ix[2],ix[1]+4,ix[0]-4,:],
                                         u[ix[2],ix[1]+3,ix[0]+3,:],u[ix[2],ix[1]-3,ix[0]+3,:],u[ix[2],ix[1]-3,ix[0]-3,:],u[ix[2],ix[1]+3,ix[0]-3,:],
                                         u[ix[2],ix[1]+2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]-2,:],u[ix[2],ix[1]+2,ix[0]-2,:],
                                         u[ix[2],ix[1]+1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]-1,:],u[ix[2],ix[1]+1,ix[0]-1,:]])
            
            uij = np.inner(fd_coeff_hessian, component_xy.T) / dx / dy
            
            if z_bottom == 'zero_ground':
                # sets all values at z_bottom (z = 0) equal to 0 because this handles the boundary condition gridpoint at z = 0. e.g. applies to
                # the 'velocity_w' variable of the 'sabl2048*' datasets.
                ukk = (u[z_top, ix[1], ix[0], :] - (2 * u[z_middle, ix[1], ix[0], :])) / (dz * dz)

                uik = (u[z_top, ix[1], ix[0] + 1, :] - u[z_top, ix[1], ix[0] - 1, :]) / (4 * dx * dz)
                ujk = (u[z_top, ix[1] + 1, ix[0], :] - u[z_top, ix[1] - 1, ix[0], :]) / (4 * dy * dz)
            else:
                ukk = (u[z_top, ix[1], ix[0], :] - (2 * u[z_middle, ix[1], ix[0], :]) + u[z_bottom, ix[1], ix[0], :]) / (dz * dz)

                uik = (u[z_top, ix[1], ix[0] + 1, :] - u[z_top, ix[1], ix[0] - 1, :] - u[z_bottom, ix[1], ix[0] + 1, :] + u[z_bottom, ix[1], ix[0] - 1, :]) / (4 * dx * dz)
                ujk = (u[z_top, ix[1] + 1, ix[0], :] - u[z_top, ix[1] - 1, ix[0], :] - u[z_bottom, ix[1] + 1, ix[0], :] + u[z_bottom, ix[1] - 1, ix[0], :]) / (4 * dy * dz)
            
            return np.stack((uii,uij,uik,ujj,ujk,ukk), axis = 1).flatten()
        
        """
        gradient finite differences.
        """
        def fdnoint_gradient():
            ix = p.astype(np.int32)
            # diagonal coefficients.
            fd_coeff = self.lookup_table
            
            # diagonal components.
            component_x = u[ix[2], ix[1], ix[0] - cube_min_index : ix[0] + cube_max_index, :]
            component_y = u[ix[2], ix[1] - cube_min_index : ix[1] + cube_max_index, ix[0], :]
            component_z = u[ix[2] - cube_min_index : ix[2] + cube_max_index, ix[1], ix[0], :]

            dvdx = np.inner(fd_coeff, component_x.T) / dx
            dvdy = np.inner(fd_coeff, component_y.T) / dy
            dvdz = np.inner(fd_coeff, component_z.T) / dz
            
            return np.stack((dvdx, dvdy, dvdz), axis = 1).flatten()
            
        """
        laplacian finite differences.
        """
        def fdnoint_laplacian():
            ix = p.astype(np.int32)
            # diagonal coefficients.
            fd_coeff = self.lookup_table
            
            # diagonal components.
            component_x = u[ix[2], ix[1], ix[0] - cube_min_index : ix[0] + cube_max_index, :]
            component_y = u[ix[2], ix[1] - cube_min_index : ix[1] + cube_max_index, ix[0], :]
            component_z = u[ix[2] - cube_min_index : ix[2] + cube_max_index, ix[1], ix[0], :]

            dvdx = np.inner(fd_coeff, component_x.T) / dx / dx
            dvdy = np.inner(fd_coeff, component_y.T) / dy / dy
            dvdz = np.inner(fd_coeff, component_z.T) / dz / dz
            
            return dvdx + dvdy + dvdz
        
        """
        hessian finite differences.
        """
        def fdnoint_hessian():
            ix = p.astype(np.int32)
            # diagonal coefficients.
            fd_coeff_laplacian = self.laplacian_lookup_table
            # off-diagonal coefficients.
            fd_coeff_hessian = self.lookup_table
            
            # diagonal components.
            component_x = u[ix[2], ix[1], ix[0] - cube_min_index : ix[0] + cube_max_index, :]
            component_y = u[ix[2], ix[1] - cube_min_index : ix[1] + cube_max_index, ix[0], :]
            component_z = u[ix[2] - cube_min_index : ix[2] + cube_max_index, ix[1], ix[0], :]

            uii = np.inner(fd_coeff_laplacian, component_x.T) / dx / dx
            ujj = np.inner(fd_coeff_laplacian, component_y.T) / dy / dy
            ukk = np.inner(fd_coeff_laplacian, component_z.T) / dz / dz

            # off-diagonal components.
            if sint == 'fd4noint_hessian':
                component_xy = np.array([u[ix[2],ix[1]+2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]-2,:],u[ix[2],ix[1]+2,ix[0]-2,:],
                                         u[ix[2],ix[1]+1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]-1,:],u[ix[2],ix[1]+1,ix[0]-1,:]])
                component_xz = np.array([u[ix[2]+2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]-2,:],u[ix[2]+2,ix[1],ix[0]-2,:],
                                         u[ix[2]+1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]-1,:],u[ix[2]+1,ix[1],ix[0]-1,:]])
                component_yz = np.array([u[ix[2]+2,ix[1]+2,ix[0],:],u[ix[2]-2,ix[1]+2,ix[0],:],u[ix[2]-2,ix[1]-2,ix[0],:],u[ix[2]+2,ix[1]-2,ix[0],:],
                                         u[ix[2]+1,ix[1]+1,ix[0],:],u[ix[2]-1,ix[1]+1,ix[0],:],u[ix[2]-1,ix[1]-1,ix[0],:],u[ix[2]+1,ix[1]-1,ix[0],:]])
            elif sint == 'fd6noint_hessian':
                component_xy = np.array([u[ix[2],ix[1]+3,ix[0]+3,:],u[ix[2],ix[1]-3,ix[0]+3,:],u[ix[2],ix[1]-3,ix[0]-3,:],u[ix[2],ix[1]+3,ix[0]-3,:],
                                         u[ix[2],ix[1]+2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]-2,:],u[ix[2],ix[1]+2,ix[0]-2,:],
                                         u[ix[2],ix[1]+1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]-1,:],u[ix[2],ix[1]+1,ix[0]-1,:]])
                component_xz = np.array([u[ix[2]+3,ix[1],ix[0]+3,:],u[ix[2]-3,ix[1],ix[0]+3,:],u[ix[2]-3,ix[1],ix[0]-3,:],u[ix[2]+3,ix[1],ix[0]-3,:],
                                         u[ix[2]+2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]-2,:],u[ix[2]+2,ix[1],ix[0]-2,:],
                                         u[ix[2]+1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]-1,:],u[ix[2]+1,ix[1],ix[0]-1,:]])
                component_yz = np.array([u[ix[2]+3,ix[1]+3,ix[0],:],u[ix[2]-3,ix[1]+3,ix[0],:],u[ix[2]-3,ix[1]-3,ix[0],:],u[ix[2]+3,ix[1]-3,ix[0],:],
                                         u[ix[2]+2,ix[1]+2,ix[0],:],u[ix[2]-2,ix[1]+2,ix[0],:],u[ix[2]-2,ix[1]-2,ix[0],:],u[ix[2]+2,ix[1]-2,ix[0],:],
                                         u[ix[2]+1,ix[1]+1,ix[0],:],u[ix[2]-1,ix[1]+1,ix[0],:],u[ix[2]-1,ix[1]-1,ix[0],:],u[ix[2]+1,ix[1]-1,ix[0],:]])
            elif sint == 'fd8noint_hessian':
                component_xy = np.array([u[ix[2],ix[1]+4,ix[0]+4,:],u[ix[2],ix[1]-4,ix[0]+4,:],u[ix[2],ix[1]-4,ix[0]-4,:],u[ix[2],ix[1]+4,ix[0]-4,:],
                                         u[ix[2],ix[1]+3,ix[0]+3,:],u[ix[2],ix[1]-3,ix[0]+3,:],u[ix[2],ix[1]-3,ix[0]-3,:],u[ix[2],ix[1]+3,ix[0]-3,:],
                                         u[ix[2],ix[1]+2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]-2,:],u[ix[2],ix[1]+2,ix[0]-2,:],
                                         u[ix[2],ix[1]+1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]-1,:],u[ix[2],ix[1]+1,ix[0]-1,:]])
                component_xz = np.array([u[ix[2]+4,ix[1],ix[0]+4,:],u[ix[2]-4,ix[1],ix[0]+4,:],u[ix[2]-4,ix[1],ix[0]-4,:],u[ix[2]+4,ix[1],ix[0]-4,:],
                                         u[ix[2]+3,ix[1],ix[0]+3,:],u[ix[2]-3,ix[1],ix[0]+3,:],u[ix[2]-3,ix[1],ix[0]-3,:],u[ix[2]+3,ix[1],ix[0]-3,:],
                                         u[ix[2]+2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]-2,:],u[ix[2]+2,ix[1],ix[0]-2,:],
                                         u[ix[2]+1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]-1,:],u[ix[2]+1,ix[1],ix[0]-1,:]])
                component_yz = np.array([u[ix[2]+4,ix[1]+4,ix[0],:],u[ix[2]-4,ix[1]+4,ix[0],:],u[ix[2]-4,ix[1]-4,ix[0],:],u[ix[2]+4,ix[1]-4,ix[0],:],
                                         u[ix[2]+3,ix[1]+3,ix[0],:],u[ix[2]-3,ix[1]+3,ix[0],:],u[ix[2]-3,ix[1]-3,ix[0],:],u[ix[2]+3,ix[1]-3,ix[0],:],
                                         u[ix[2]+2,ix[1]+2,ix[0],:],u[ix[2]-2,ix[1]+2,ix[0],:],u[ix[2]-2,ix[1]-2,ix[0],:],u[ix[2]+2,ix[1]-2,ix[0],:],
                                         u[ix[2]+1,ix[1]+1,ix[0],:],u[ix[2]-1,ix[1]+1,ix[0],:],u[ix[2]-1,ix[1]-1,ix[0],:],u[ix[2]+1,ix[1]-1,ix[0],:]])
            
            uij = np.inner(fd_coeff_hessian, component_xy.T) / dx / dy
            uik = np.inner(fd_coeff_hessian, component_xz.T) / dx / dz
            ujk = np.inner(fd_coeff_hessian, component_yz.T) / dy / dz
            
            return np.stack((uii,uij,uik,ujj,ujk,ukk), axis = 1).flatten()
        
        """
        gradient spline differentiations.
        """
        def spline_gradient():
            ix = p.astype(np.int32)
            fr = p - ix
            
            # field spline coefficients.
            gx = self.field_lookup_table[int(lookup_N * fr[0])]
            gy = self.field_lookup_table[int(lookup_N * fr[1])]
            gz = self.field_lookup_table[int(lookup_N * fr[2])]
            
            # gradient spline coefficients.
            gx_gradient = self.lookup_table[int(lookup_N * fr[0])]
            gy_gradient = self.lookup_table[int(lookup_N * fr[1])]
            gz_gradient = self.lookup_table[int(lookup_N * fr[2])]
            
            gk_x = np.einsum('i,j,k', gz, gy, gx_gradient)
            gk_y = np.einsum('i,j,k', gz, gy_gradient, gx)
            gk_z = np.einsum('i,j,k', gz_gradient, gy, gx)
            
            # dudx, dvdx, dwdx.
            dvdx = np.einsum('ijk,ijkl->l', gk_x, u) / dx
            # dudy, dvdy, dwdy.
            dvdy = np.einsum('ijk,ijkl->l', gk_y, u) / dy
            # dudz, dvdz, dwdz.
            dvdz = np.einsum('ijk,ijkl->l', gk_z, u) / dz
            
            return np.stack((dvdx, dvdy, dvdz), axis = 1).flatten()
        
        """
        hessian spline differentiation.
        """
        def spline_hessian():
            ix = p.astype(np.int32)
            fr = p - ix
            
            # field spline coefficients.
            gx = self.field_lookup_table[int(lookup_N * fr[0])]
            gy = self.field_lookup_table[int(lookup_N * fr[1])]
            gz = self.field_lookup_table[int(lookup_N * fr[2])]
            
            # gradient spline coefficients.
            gx_gradient = self.gradient_lookup_table[int(lookup_N * fr[0])]
            gy_gradient = self.gradient_lookup_table[int(lookup_N * fr[1])]
            gz_gradient = self.gradient_lookup_table[int(lookup_N * fr[2])]
            
            # hessian spline coefficients.
            gx_hessian = self.lookup_table[int(lookup_N * fr[0])]
            gy_hessian = self.lookup_table[int(lookup_N * fr[1])]
            gz_hessian = self.lookup_table[int(lookup_N * fr[2])]

            gk_xx = np.einsum('i,j,k', gz, gy, gx_hessian)
            gk_yy = np.einsum('i,j,k', gz, gy_hessian, gx)
            gk_zz = np.einsum('i,j,k', gz_hessian, gy, gx)
            gk_xy = np.einsum('i,j,k', gz, gy_gradient, gx_gradient)
            gk_xz = np.einsum('i,j,k', gz_gradient, gy, gx_gradient)
            gk_yz = np.einsum('i,j,k', gz_gradient, gy_gradient, gx)

            uii = np.einsum('ijk,ijkl->l', gk_xx, u) / dx / dx
            ujj = np.einsum('ijk,ijkl->l', gk_yy, u) / dy / dy
            ukk = np.einsum('ijk,ijkl->l', gk_zz, u) / dz / dz
            uij = np.einsum('ijk,ijkl->l', gk_xy, u) / dx / dy
            uik = np.einsum('ijk,ijkl->l', gk_xz, u) / dx / dz
            ujk = np.einsum('ijk,ijkl->l', gk_yz, u) / dy / dz                    

            return np.stack((uii, uij, uik, ujj, ujk, ukk), axis = 1).flatten()
        
        """
        gradient finite difference with field interpolation.
        """
        def fd4lag4_gradient():
            ix = p.astype(np.int32)
            fr = p - ix      
            
            # field interpolation coefficients.
            gx = self.field_lookup_table[int(lookup_N * fr[0])]
            gy = self.field_lookup_table[int(lookup_N * fr[1])]
            gz = self.field_lookup_table[int(lookup_N * fr[2])]
            
            # finite difference coefficients.
            gx_gradient = self.lookup_table[int(lookup_N * fr[0])]
            gy_gradient = self.lookup_table[int(lookup_N * fr[1])]
            gz_gradient = self.lookup_table[int(lookup_N * fr[2])]
            
            gk_x = np.einsum('i,j,k', gz, gy, gx_gradient)
            gk_y = np.einsum('i,j,k', gz, gy_gradient, gx)
            gk_z = np.einsum('i,j,k', gz_gradient, gy, gx)

            d_x = u[ix[2] - 1 : ix[2] + 3, ix[1] - 1 : ix[1] + 3, ix[0] - 3 : ix[0] + 5, :]           
            d_y = u[ix[2] - 1 : ix[2] + 3, ix[1] - 3 : ix[1] + 5, ix[0] - 1 : ix[0] + 3, :]           
            d_z = u[ix[2] - 3 : ix[2] + 5, ix[1] - 1 : ix[1] + 3, ix[0] - 1 : ix[0] + 3, :]
            
            # dudx,dvdx,dwdx.
            dvdx = np.einsum('ijk,ijkl->l', gk_x, d_x) / dx
            # dudy,dvdy,dwdy.
            dvdy = np.einsum('ijk,ijkl->l', gk_y, d_y) / dy
            # dudz,dvdz,dwdz.
            dvdz = np.einsum('ijk,ijkl->l', gk_z, d_z) / dz
            
            return np.stack((dvdx, dvdy, dvdz), axis = 1).flatten()
            
        """
        laplacian finite difference with field interpolation.
        """
        def fd4lag4_laplacian():
            ix = p.astype(np.int32)
            fr = p - ix      
            
            # field interpolation coefficients.
            gx = self.field_lookup_table[int(lookup_N * fr[0])]
            gy = self.field_lookup_table[int(lookup_N * fr[1])]
            gz = self.field_lookup_table[int(lookup_N * fr[2])]
            
            # finite difference coefficients.
            gx_laplacian = self.lookup_table[int(lookup_N * fr[0])]
            gy_laplacian = self.lookup_table[int(lookup_N * fr[1])]
            gz_laplacian = self.lookup_table[int(lookup_N * fr[2])]
            
            gk_x = np.einsum('i,j,k', gz, gy, gx_laplacian)           
            gk_y = np.einsum('i,j,k', gz, gy_laplacian, gx)           
            gk_z = np.einsum('i,j,k', gz_laplacian, gy, gx)

            d_x = u[ix[2] - 1 : ix[2] + 3, ix[1] - 1 : ix[1] + 3, ix[0] - 3 : ix[0] + 5, :]           
            d_y = u[ix[2] - 1 : ix[2] + 3, ix[1] - 3 : ix[1] + 5, ix[0] - 1 : ix[0] + 3, :]           
            d_z = u[ix[2] - 3 : ix[2] + 5, ix[1] - 1 : ix[1] + 3, ix[0] - 1 : ix[0] + 3, :]
            
            # dudx,dvdx,dwdx.
            dvdx = np.einsum('ijk,ijkl->l', gk_x, d_x) / dx / dx
            # dudy,dvdy,dwdy.
            dvdy = np.einsum('ijk,ijkl->l', gk_y, d_y) / dy / dy
            # dudz,dvdz,dwdz.
            dvdz = np.einsum('ijk,ijkl->l', gk_z, d_z) / dz / dz
            
            dudxyz = dvdx[0] + dvdy[0] + dvdz[0]
            dvdxyz = dvdx[1] + dvdy[1] + dvdz[1]
            dwdxyz = dvdx[2] + dvdy[2] + dvdz[2]

            return np.array([dudxyz, dvdxyz, dwdxyz])
        
        # interpolate functions map.
        interpolate_functions = {
            'none': none,
            'lag4': lag_spline, 'lag6': lag_spline, 'lag8': lag_spline,
            'm1q4': lag_spline, 'm2q8': lag_spline,
            'x_linear': x_linear,
            'y_linear': y_linear,
            'z_linear': z_linear,
            'z_linear_gradient': z_linear_gradient,
            'z_linear_laplacian': z_linear_laplacian,
            'z_linear_hessian': z_linear_hessian,
            'fd4noint_gradient': fdnoint_gradient, 'fd6noint_gradient': fdnoint_gradient, 'fd8noint_gradient': fdnoint_gradient,
            'fd4noint_laplacian': fdnoint_laplacian, 'fd6noint_laplacian': fdnoint_laplacian, 'fd8noint_laplacian': fdnoint_laplacian,
            'fd4noint_hessian': fdnoint_hessian, 'fd6noint_hessian': fdnoint_hessian, 'fd8noint_hessian': fdnoint_hessian,
            'm1q4_gradient': spline_gradient, 'm2q8_gradient': spline_gradient,
            'm2q8_hessian': spline_hessian,
            'fd4lag4_gradient': fd4lag4_gradient,
            'fd4lag4_laplacian': fd4lag4_laplacian
        }
        
        # interpolation function to call.
        interpolate_function = interpolate_functions[sint]
        
        return interpolate_function()
        
    """
    common functions.
    """
    def read_pickle_file(self, pickle_filename):
        """
        read the pickle metadata file. first, try reading from the production copy. second, try reading from the backup copy.
        """
        try:
            # pickled file production filepath.
            pickle_file = self.pickle_dir.joinpath(pickle_filename)
        
            # try reading the pickled file.
            with open(pickle_file, 'rb') as pickled_filepath:
                return dill.load(pickled_filepath)
        except:
            try:
                # pickled file backup filepath.
                pickle_file = self.pickle_dir_local.joinpath(pickle_filename)

                # try reading the pickled file.
                with open(pickle_file, 'rb') as pickled_filepath:
                    return dill.load(pickled_filepath)
            except:
                raise Exception('metadata files are not accessible.')
    
    def open_zarr_file(self, open_file_vars):
        """
        open the zarr file for reading. first, try reading from the production copy. second, try reading from the backup copy.
        """
        # assign the local variables.
        zarr_filepath, var, dt = open_file_vars
        
        try:
            # try reading from the production file.
            return zarr.open(store = f'{zarr_filepath}{os.sep}{var}', dtype = dt, mode = 'r')
        except:
            raise Exception(f'{zarr_filepath}{os.sep}{var} is not accessible.')
    
    """
    getCutout functions.
    """
    def map_chunks_getcutout(self, axes_ranges):
        """
        split up the cutout box into all constituent chunks for reading.
        """
        chunk_boxes = []
        
        # modulus and periodic axes ranges.
        mod_axes_ranges = (axes_ranges.T % self.N).T
        periodic_axes_ranges = [[axes_chunk_size * math.floor(axes_range[0] / axes_chunk_size), axes_range[1]] for axes_range, axes_chunk_size in zip(axes_ranges, self.chunk_size)]
        
        # split up axes_ranges into the constituent chunks, taking into account periodic boundary conditions.
        for xax in range(periodic_axes_ranges[0][0], periodic_axes_ranges[0][1] + 1, self.chunk_size[0]):
            for yax in range(periodic_axes_ranges[1][0], periodic_axes_ranges[1][1] + 1, self.chunk_size[1]):
                for zax in range(periodic_axes_ranges[2][0], periodic_axes_ranges[2][1] + 1, self.chunk_size[2]):
                    # modulus of xax, yax, and zax values.
                    mod_xax, mod_yax, mod_zax = xax % self.N[0], yax % self.N[1], zax % self.N[2]
                    # append the constituent chunk boxes into chunk_boxes. the axes ranges to read from the zarr store (first x, y, z ranges in chunk_boxes) and
                    # corresponding index ranges (second x, y, z ranges in chunk_boxes) are appended together for filling the result array correctly.
                    chunk_boxes.append([[[mod_xax if xax != periodic_axes_ranges[0][0] else mod_axes_ranges[0][0],
                                          mod_xax + self.chunk_size[0] - 1 if xax + self.chunk_size[0] - 1 <= periodic_axes_ranges[0][1] else mod_axes_ranges[0][1]],
                                         [mod_yax if yax != periodic_axes_ranges[1][0] else mod_axes_ranges[1][0],
                                          mod_yax + self.chunk_size[1] - 1 if yax + self.chunk_size[1] - 1 <= periodic_axes_ranges[1][1] else mod_axes_ranges[1][1]],
                                         [mod_zax if zax != periodic_axes_ranges[2][0] else mod_axes_ranges[2][0],
                                          mod_zax + self.chunk_size[2] - 1 if zax + self.chunk_size[2] - 1 <= periodic_axes_ranges[2][1] else mod_axes_ranges[2][1]]],
                                        [[xax if xax != periodic_axes_ranges[0][0] else axes_ranges[0][0],
                                          xax + self.chunk_size[0] - 1 if xax + self.chunk_size[0] - 1 <= periodic_axes_ranges[0][1] else axes_ranges[0][1]],
                                         [yax if yax != periodic_axes_ranges[1][0] else axes_ranges[1][0],
                                          yax + self.chunk_size[1] - 1 if yax + self.chunk_size[1] - 1 <= periodic_axes_ranges[1][1] else axes_ranges[1][1]],
                                         [zax if zax != periodic_axes_ranges[2][0] else axes_ranges[2][0],
                                          zax + self.chunk_size[2] - 1 if zax + self.chunk_size[2] - 1 <= periodic_axes_ranges[2][1] else axes_ranges[2][1]]]])

        return np.array(chunk_boxes)
    
    def read_database_files_getcutout(self, chunk_boxes):
        """
        submit the chunks for reading.
        """
        num_chunks = len(chunk_boxes)
        num_processes = min(self.dask_maximum_processes, num_chunks)
        
        with ThreadPoolExecutor(max_workers = num_processes) as executor:
            result_output_data = list(executor.map(self.get_points_getcutout,
                chunk_boxes,
                [self.timepoint] * num_chunks,
                [self.dataset_title] * num_chunks,
                [self.zarr_store] * num_chunks,
                chunksize = 1))
        
        # flattens result_output_data.
        return list(itertools.chain.from_iterable(result_output_data))
    
    def get_points_getcutout(self, chunk_data, timepoint, dataset_title, zarr_store):
        """
        retrieve the values for the specified var(iable) in the user-specified box and at the specified timepoint.
        """
        chunk_ranges = chunk_data[0]
        index_ranges = chunk_data[1]

        # retrieve the minimum and maximum (x, y, z) coordinates of the chunk that is going to be read in.
        min_xyz = [axis_range[0] for axis_range in index_ranges]
        max_xyz = [axis_range[1] for axis_range in index_ranges]

        def single_timepoint():
            # cutout data from the specified chunk.
            return [(zarr_store[timepoint,
                                chunk_ranges[2][0] : chunk_ranges[2][1] + 1,
                                chunk_ranges[1][0] : chunk_ranges[1][1] + 1,
                                chunk_ranges[0][0] : chunk_ranges[0][1] + 1],
                                min_xyz, max_xyz)]
        
        def windfarm_timepoint():
            # cutout data from the specified chunk.
            return [(zarr_store[timepoint[0],
                                timepoint[1],
                                timepoint[2],
                                chunk_ranges[2][0] : chunk_ranges[2][1] + 1,
                                chunk_ranges[1][0] : chunk_ranges[1][1] + 1,
                                chunk_ranges[0][0] : chunk_ranges[0][1] + 1],
                                min_xyz, max_xyz)]
        
        # read zarr function map.
        read_zarr_functions = defaultdict(lambda: single_timepoint, {'diurnal_windfarm': windfarm_timepoint})
        
        return read_zarr_functions[dataset_title]()
            
    """
    getData functions.
    """
    def map_chunks_getdata(self, points):
        """
        map each point to a chunk group for reading from the zarr store.
        """
        # chunk cube size.
        chunk_cube_size = np.prod(self.chunk_size)
        # empty array for subdividing chunk groups.
        empty_array = np.array([0, 0, 0])
        # chunk size array for subdividing chunk groups.
        chunk_size_array = self.chunk_size - 1
        
        # convert the points to the center point position within their own bucket.
        center_points = (((points + self.spacing * self.grid_offsets) / self.spacing) % 1) + self.cube_min_index
        # convert the points to gridded datapoints. there is a +0.5 point shift because the finite differencing methods would otherwise add +0.5 to center_points when
        # interpolating the values. shifting the datapoints up by +0.5 adjusts the bucket up one grid point so the center_points do not needed to be shifted up by +0.5.
        sint_datapoint_shift = self.sint_specified if 'z_linear' in self.sint else self.sint
        if sint_datapoint_shift in ['fd4noint_gradient', 'fd6noint_gradient', 'fd8noint_gradient',
                                    'fd4noint_laplacian', 'fd6noint_laplacian', 'fd8noint_laplacian',
                                    'fd4noint_hessian', 'fd6noint_hessian', 'fd8noint_hessian']:
            datapoints = np.floor((points + self.spacing * (self.grid_offsets + 0.5)) / self.spacing).astype(int) % self.N
        else:
            datapoints = np.floor((points + self.spacing * self.grid_offsets) / self.spacing).astype(int) % self.N
        
        # adjust center_points and datapoints for the 'z_linear*' methods to make sure that the buckets do not wrap around the z-axis since it is not periodic. the values
        # in buckets_info are used in the 'z_linear*' methods in the spatial_interpolate function.
        buckets_info = np.full((len(points), 3), None)
        if self.dataset_title in ['sabl2048low', 'sabl2048high']:
            if self.sint == 'z_linear':
                # adjustments for the 'z_linear' step-down method for the interpolation functions (i.e. lag4/6/8, m1q4, m2q8). 
                if self.var_offsets == 'sgsenergy':
                    # ground boundary condition of the 'sgsenergy' variable.
                    # condition to test the points against.
                    where_condition = points[:, 2] < self.dz
                    # update center_points z-axis because the points are below self.dz, which is the first gridpoint for the 'sgsenergy' variable.
                    center_points[:, 2] = np.where(where_condition, np.floor(center_points[:, 2]), center_points[:, 2])
                    # shift datapoints up by 1 gridpoint because points < self.dz are below the first gridpoint. the ground boundary condition will be applied for the
                    # 'sgsenergy' variable, which is that e(z = 0) = e(z = self.dz).
                    datapoints[:, 2] = np.where(where_condition, (datapoints[:, 2] + 1) % self.N[2], datapoints[:, 2])
                elif self.var_offsets == 'velocity_w':
                    # ground boundary condition of the w-component of the 'velocity' variable.
                    where_condition = points[:, 2] < self.dz
                    # update the z_bottom_flag of buckets_info for points that satisfy where_condition. 'zero_ground' is handled inside the spatial_interpolate function.
                    buckets_info[np.where(where_condition), 0] = 'zero_ground'
                    # shift datapoints up by 1 gridpoint because points < self.dz are below the first gridpoint. the ground boundary condition will be applied for the
                    # w-component of the 'velocity' variable, which is that w(z = 0) = 0.
                    datapoints[:, 2] = np.where(where_condition, (datapoints[:, 2] + 1) % self.N[2], datapoints[:, 2])
                elif self.var_offsets in ['pressure', 'temperature', 'velocity_uv']:
                    # sky boundary condition of the 'pressure', 'temperature', and (u,v)-components of the 'velocity variables.
                    where_condition = points[:, 2] == (2047.5 * self.dz)
                    # update center_points z-axis because datapoints will be shifted down by 1 gridpoint.
                    center_points[:, 2] = np.where(where_condition, center_points[:, 2] + 1, center_points[:, 2])
                    # shift datapoints down by 1 gridpoint to make sure that the bucket does not needlessly wrap around the z-axis.
                    datapoints[:, 2] = np.where(where_condition, (datapoints[:, 2] - 1) % self.N[2], datapoints[:, 2])
            elif self.sint in ['z_linear_gradient', 'z_linear_laplacian']:
                # adjustments for the 'z_linear_gradient' and 'z_linear_laplacian' step-down methods for the finite differencing functions (i.e. fd4/6/8noint, fd4lag4).
                if self.var_offsets in ['sgsenergy', 'velocity_w']:
                    # fd2 ground boundary condition.
                    if self.var_offsets == 'sgsenergy':
                        where_condition = points[:, 2] < (1.5 * self.dz)
                        buckets_info[np.where(where_condition), :] = [0, 1, 2]
                    else:
                        where_condition = points[:, 2] < (1.5 * self.dz)
                        buckets_info[np.where(where_condition), :] = ['zero_ground', 2, 2]
                    
                    # fd2 ground and sky regions. the fd2 region is left open to the entire dataset because it is a step-down method for specified methods
                    # that have different bucket sizes. only points assigned to the linear method will be interpolated inside this region.
                    where_condition = np.logical_and(points[:, 2] >= (1.5 * self.dz), points[:, 2] < (2047.5 * self.dz))
                    buckets_info[np.where(where_condition), :] = [0, 2, 2]
                    # shift datapoints down by 1 gridpoint to make sure that the correct bucket is being read and does not needlessly wrap around the z-axis.
                    datapoints[:, 2] = np.where(where_condition, (datapoints[:, 2] - 1) % self.N[2], datapoints[:, 2])
                    
                    # fd1 sky region.
                    where_condition = points[:, 2] == (2047.5 * self.dz)
                    buckets_info[np.where(where_condition), :] = [1, 2, 1]
                    # shift datapoints down by 2 gridpoints to make sure that the bucket does not needlessly wrap around the z-axis.
                    datapoints[:, 2] = np.where(where_condition, (datapoints[:, 2] - 2) % self.N[2], datapoints[:, 2])
                elif self.var_offsets in ['pressure', 'temperature', 'velocity_uv']:
                    # fd1 ground region.
                    where_condition = points[:, 2] < self.dz
                    buckets_info[np.where(where_condition), :] = [0, 1, 1]
                    
                    # fd2 ground and sky regions. the fd2 region is left open to the entire dataset because it is a step-down method for specified methods
                    # that have different bucket sizes. only points assigned to the linear method will be interpolated inside this region.
                    where_condition = np.logical_and(points[:, 2] >= self.dz, points[:, 2] < (2047 * self.dz))
                    buckets_info[np.where(where_condition), :] = [0, 2, 2]
                    # shift datapoints down by 1 gridpoint to make sure that the correct bucket is being read and does not needlessly wrap around the z-axis.
                    datapoints[:, 2] = np.where(where_condition, (datapoints[:, 2] - 1) % self.N[2], datapoints[:, 2])
                    
                    # fd1 sky region.
                    where_condition = points[:, 2] >= (2047 * self.dz)
                    buckets_info[np.where(where_condition), :] = [1, 2, 1]
                    # shift datapoints down by 2 gridpoints to make sure that the bucket does not needlessly wrap around the z-axis.
                    datapoints[:, 2] = np.where(where_condition, (datapoints[:, 2] - 2) % self.N[2], datapoints[:, 2])
            elif self.sint == 'z_linear_hessian':
                # adjustments for the 'z_linear_hessian' step-down methods for the finite differencing functions (i.e. fd4/6/8noint, m2q8).
                if self.var_offsets in ['sgsenergy', 'velocity_w']:
                    # fd2 ground boundary condition.
                    if self.var_offsets == 'sgsenergy':
                        where_condition = points[:, 2] < (1.5 * self.dz)
                        buckets_info[np.where(where_condition), :] = [0, 0, 1]
                    else:
                        where_condition = points[:, 2] < (1.5 * self.dz)
                        buckets_info[np.where(where_condition), :] = ['zero_ground', 0, 1]
                    
                    # fd2 ground and sky regions. the fd2 region is left open to the entire dataset because it is a step-down method for specified methods
                    # that have different bucket sizes. only points assigned to the linear method will be interpolated inside this region.
                    where_condition = np.logical_and(points[:, 2] >= (1.5 * self.dz), points[:, 2] < (2047.5 * self.dz))
                    buckets_info[np.where(where_condition), :] = [0, 1, 2]
                    # shift datapoints down by 1 gridpoint to make sure that the correct bucket is being read and does not needlessly wrap around the z-axis.
                    datapoints[:, 2] = np.where(where_condition, (datapoints[:, 2] - 1) % self.N[2], datapoints[:, 2])
                    
                    # fd2 sky boundary condition.
                    where_condition = points[:, 2] == (2047.5 * self.dz)
                    buckets_info[np.where(where_condition), :] = [0, 1, 2]
                    # shift datapoints down by 2 gridpoints to make sure that the bucket does not needlessly wrap around the z-axis.
                    datapoints[:, 2] = np.where(where_condition, (datapoints[:, 2] - 2) % self.N[2], datapoints[:, 2])
                elif self.var_offsets in ['pressure', 'temperature', 'velocity_uv']:
                    # fd2 ground boundary condition.
                    where_condition = points[:, 2] < self.dz
                    buckets_info[np.where(where_condition), :] = [0, 1, 2]
                    
                    # fd2 ground and sky regions. the fd2 region is left open to the entire dataset because it is a step-down method for specified methods
                    # that have different bucket sizes. only points assigned to the linear method will be interpolated inside this region.
                    where_condition = np.logical_and(points[:, 2] >= self.dz, points[:, 2] < (2046 * self.dz))
                    buckets_info[np.where(where_condition), :] = [0, 1, 2]
                    # # shift datapoints down by 1 gridpoint to make sure that the correct bucket is being read and does not needlessly wrap around the z-axis.
                    datapoints[:, 2] = np.where(where_condition, (datapoints[:, 2] - 1) % self.N[2], datapoints[:, 2])
                    
                    # fd2 sky boundary conditions.
                    where_condition = np.logical_and(points[:, 2] >= (2046 * self.dz), points[:, 2] < (2047 * self.dz))
                    buckets_info[np.where(where_condition), :] = [0, 1, 2]
                    # shift datapoints down by 1 gridpoint because points >= (2046 * self.dz) do not satisfy having one z-gridpoint above and below the datapoint
                    # for calculating the linear fd2 hessian in the boundary region.
                    datapoints[:, 2] = np.where(where_condition, (datapoints[:, 2] - 1) % self.N[2], datapoints[:, 2])
                    
                    where_condition = points[:, 2] >= (2047 * self.dz)
                    buckets_info[np.where(where_condition), :] = [0, 1, 2]
                    # shift datapoints down by 2 gridpoints because points >= (2047 * self.dz) do not satisfy having one z-gridpoint above and below the datapoint
                    # for calculating the linear fd2 hessian in the boundary region.
                    datapoints[:, 2] = np.where(where_condition, (datapoints[:, 2] - 2) % self.N[2], datapoints[:, 2])
        
        # calculate the minimum and maximum chunk (x, y, z) corner point for each point in datapoints.
        chunk_min_xyzs = ((datapoints - self.cube_min_index) - ((datapoints - self.cube_min_index) % self.chunk_size))
        chunk_max_xyzs = ((datapoints + self.cube_max_index) + (self.chunk_size - ((datapoints + self.cube_max_index) % self.chunk_size) - 1))
        chunk_min_mod_xyzs = chunk_min_xyzs % self.N
        chunk_max_mod_xyzs = chunk_max_xyzs % self.N
        # chunk volumes.
        chunk_volumes = np.prod(chunk_max_xyzs - chunk_min_xyzs + 1, axis = 1)        
        # create the chunk keys for each chunk group.
        chunk_keys = [chunk_origin_group.tobytes() for chunk_origin_group in np.stack([chunk_min_xyzs, chunk_max_xyzs], axis = 1)]
        
        # save the original indices for points, which corresponds to the orderering of the user-specified
        # points. these indices will be used for sorting output_data back to the user-specified points ordering.
        original_points_indices = np.arange(len(points))
        # zip the data. sort by volume first so that all fully overlapped chunk groups can be easily found.
        zipped_data = sorted(zip(chunk_volumes, chunk_keys, points, datapoints, center_points,
                                 chunk_min_xyzs, chunk_max_xyzs, chunk_min_mod_xyzs, chunk_max_mod_xyzs,
                                 buckets_info, original_points_indices), key = lambda x: (-1 * x[0], x[1]))
        
        # map the bucket points to their chunks.
        chunk_data_map = defaultdict(list)
        # chunk key map used for storing all subdivided chunk groups to find fully overlapped chunk groups.
        chunk_map = {}
        
        for chunk_volume, chunk_key, point, datapoint, center_point, \
            chunk_min_xyz, chunk_max_xyz, chunk_min_mod_xyz, chunk_max_mod_xyz, \
            bucket_info, original_point_index in zipped_data:
            # update the chunk key if the chunk group is fully contained in another larger chunk group.
            updated_chunk_key = chunk_key
            if chunk_key in chunk_map:
                updated_chunk_key = chunk_map[chunk_key]
            elif chunk_volume != chunk_cube_size:
                chunk_map = self.subdivide_chunk_group(chunk_map, chunk_key, chunk_min_xyz, chunk_max_xyz, chunk_size_array, empty_array)

            # assign to chunk_data_map.
            if updated_chunk_key not in chunk_data_map:
                chunk_data_map[updated_chunk_key].append((chunk_min_xyz, chunk_max_xyz, chunk_min_mod_xyz, chunk_max_mod_xyz))

            chunk_data_map[updated_chunk_key].append((point, datapoint, center_point, bucket_info, original_point_index))
        
        return np.array(list(chunk_data_map.values()), dtype = object)
    
    def subdivide_chunk_group(self, chunk_map, chunk_key, chunk_min_xyz, chunk_max_xyz, chunk_size_array, empty_array):
        """
        map all subset chunk groups to chunk_key.
        """
        chunk_mins = []
        chunk_maxs = []

        # axes that are 2 chunks in length.
        chunk_diffs = np.where(chunk_max_xyz - chunk_min_xyz + 1 == 2 * self.chunk_size)[0]
        num_long_axes = len(chunk_diffs)

        # 1-cubes, which are needed for all chunk groups (2, 4, or 8 chunks).
        # long axis 1, first 1-cube.
        chunk_mins.append(chunk_min_xyz)
        new_max = chunk_min_xyz + chunk_size_array
        chunk_maxs.append(new_max)

        # long axis 1, second 1-cube.
        new_min = chunk_min_xyz + empty_array
        new_min[chunk_diffs[0]] += self.chunk_size[0]
        new_max = chunk_min_xyz + chunk_size_array
        new_max[chunk_diffs[0]] += self.chunk_size[0]
        chunk_mins.append(new_min)
        chunk_maxs.append(new_max)
        
        # add additional sub-chunks chunk group contains 4 or 8 chunks.
        if num_long_axes == 2 or num_long_axes == 3:
            # 1-cubes, additional.
            # long axis 2, first 1-cube.
            new_min = chunk_min_xyz + empty_array
            new_min[chunk_diffs[1]] += self.chunk_size[1]
            new_max = chunk_min_xyz + chunk_size_array
            new_max[chunk_diffs[1]] += self.chunk_size[1]
            chunk_mins.append(new_min)
            chunk_maxs.append(new_max)

            # long axis 2, second 1-cube.
            new_min = chunk_min_xyz + empty_array
            new_min[chunk_diffs[0]] += self.chunk_size[0]
            new_min[chunk_diffs[1]] += self.chunk_size[1]
            new_max = chunk_min_xyz + chunk_size_array
            new_max[chunk_diffs[0]] += self.chunk_size[0]
            new_max[chunk_diffs[1]] += self.chunk_size[1]
            chunk_mins.append(new_min)
            chunk_maxs.append(new_max)

            # 2-cubes.
            # long axis 1, first 2-cube.
            new_max = chunk_min_xyz + chunk_size_array
            new_max[chunk_diffs[0]] += self.chunk_size[0]
            chunk_mins.append(chunk_min_xyz)
            chunk_maxs.append(new_max)

            # long axis 1, second 2-cube.
            new_min = chunk_min_xyz + empty_array
            new_min[chunk_diffs[1]] += self.chunk_size[1]
            new_max = chunk_min_xyz + chunk_size_array
            new_max[chunk_diffs[0]] += self.chunk_size[0]
            new_max[chunk_diffs[1]] += self.chunk_size[1]
            chunk_mins.append(new_min)
            chunk_maxs.append(new_max)
            
            # long axis 2, first 2-cube.
            new_max = chunk_min_xyz + chunk_size_array
            new_max[chunk_diffs[1]] += self.chunk_size[1]
            chunk_mins.append(chunk_min_xyz)
            chunk_maxs.append(new_max)

            # long axis 2, second 2-cube.
            new_min = chunk_min_xyz + empty_array
            new_min[chunk_diffs[0]] += self.chunk_size[0]
            new_max = chunk_min_xyz + chunk_size_array
            new_max[chunk_diffs[0]] += self.chunk_size[0]
            new_max[chunk_diffs[1]] += self.chunk_size[1]
            chunk_mins.append(new_min)
            chunk_maxs.append(new_max)
        
            if num_long_axes == 3:
                # 1-cubes, additional.
                # long axis 3, first 1-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[2]] += self.chunk_size[2]
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[2]] += self.chunk_size[2]
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 3, second 1-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[0]] += self.chunk_size[0]
                new_min[chunk_diffs[2]] += self.chunk_size[2]
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size[0]
                new_max[chunk_diffs[2]] += self.chunk_size[2]
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 3, third 1-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[1]] += self.chunk_size[1]
                new_min[chunk_diffs[2]] += self.chunk_size[2]
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[1]] += self.chunk_size[1]
                new_max[chunk_diffs[2]] += self.chunk_size[2]
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 3, fourth 1-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[0]] += self.chunk_size[0]
                new_min[chunk_diffs[1]] += self.chunk_size[1]
                new_min[chunk_diffs[2]] += self.chunk_size[2]
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size[0]
                new_max[chunk_diffs[1]] += self.chunk_size[1]
                new_max[chunk_diffs[2]] += self.chunk_size[2]
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # 2-cubes, additional.
                # long axis 1, third 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[2]] += self.chunk_size[2]
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size[0]
                new_max[chunk_diffs[2]] += self.chunk_size[2]
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 1, fourth 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[1]] += self.chunk_size[1]
                new_min[chunk_diffs[2]] += self.chunk_size[2]
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size[0]
                new_max[chunk_diffs[1]] += self.chunk_size[1]
                new_max[chunk_diffs[2]] += self.chunk_size[2]
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)
                
                # long axis 2, third 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[2]] += self.chunk_size[2]
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[1]] += self.chunk_size[1]
                new_max[chunk_diffs[2]] += self.chunk_size[2]
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 2, fourth 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[0]] += self.chunk_size[0]
                new_min[chunk_diffs[2]] += self.chunk_size[2]
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size[0]
                new_max[chunk_diffs[1]] += self.chunk_size[1]
                new_max[chunk_diffs[2]] += self.chunk_size[2]
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)
                
                # long axis 3, first 2-cube.
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[2]] += self.chunk_size[2]
                chunk_mins.append(chunk_min_xyz)
                chunk_maxs.append(new_max)

                # long axis 3, second 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[0]] += self.chunk_size[0]
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size[0]
                new_max[chunk_diffs[2]] += self.chunk_size[2]
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 3, third 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[1]] += self.chunk_size[1]
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[1]] += self.chunk_size[1]
                new_max[chunk_diffs[2]] += self.chunk_size[2]
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 3, fourth 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[0]] += self.chunk_size[0]
                new_min[chunk_diffs[1]] += self.chunk_size[1]
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size[0]
                new_max[chunk_diffs[1]] += self.chunk_size[1]
                new_max[chunk_diffs[2]] += self.chunk_size[2]
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # 4-cubes.
                # long axes 1 and 2, first 4-cube.
                new_min = chunk_min_xyz + empty_array
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size[0]
                new_max[chunk_diffs[1]] += self.chunk_size[1]
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axes 1 and 2, second 4-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[2]] += self.chunk_size[2]
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size[0]
                new_max[chunk_diffs[1]] += self.chunk_size[1]
                new_max[chunk_diffs[2]] += self.chunk_size[2]
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axes 1 and 3, first 4-cube.
                new_min = chunk_min_xyz + empty_array
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size[0]
                new_max[chunk_diffs[2]] += self.chunk_size[2]
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axes 1 and 3, second 4-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[1]] += self.chunk_size[1]
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size[0]
                new_max[chunk_diffs[1]] += self.chunk_size[1]
                new_max[chunk_diffs[2]] += self.chunk_size[2]
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axes 2 and 3, first 4-cube.
                new_min = chunk_min_xyz + empty_array
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[1]] += self.chunk_size[1]
                new_max[chunk_diffs[2]] += self.chunk_size[2]
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axes 2 and 3, second 4-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[0]] += self.chunk_size[0]
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size[0]
                new_max[chunk_diffs[1]] += self.chunk_size[1]
                new_max[chunk_diffs[2]] += self.chunk_size[2]
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

        # whole cube.
        chunk_mins.append(chunk_min_xyz)
        chunk_maxs.append(chunk_max_xyz)

        # convert to numpy arrays.
        chunk_mins = np.array(chunk_mins)
        chunk_maxs = np.array(chunk_maxs)

        # update chunk_map with all of the new keys.
        chunk_keys = [chunk_origin_group.tobytes() for chunk_origin_group in np.stack([chunk_mins, chunk_maxs], axis = 1)]
        for key in chunk_keys:
            chunk_map[key] = chunk_key

        return chunk_map
    
    def read_database_files_getdata(self, chunk_data_map):
        """
        submit the points for reading and interpolation.
        """
        num_chunks = len(chunk_data_map)
        num_processes = min(self.dask_maximum_processes, num_chunks)
        
        with ThreadPoolExecutor(max_workers = num_processes) as executor:
            result_output_data = list(executor.map(self.get_points_getdata,
                chunk_data_map,
                [self.timepoint] * num_chunks,
                [self.zarr_store] * num_chunks,
                [self.getdata_vars] * num_chunks,
                [self.interpolate_vars] * num_chunks,
                chunksize = 1))
        
        # flattens result_output_data.
        return list(itertools.chain.from_iterable(result_output_data))
    
    def get_points_getdata(self, map_data, timepoint, zarr_store,
                           getdata_vars, interpolate_vars):
        """
        reads and interpolates the user-requested points in a zarr store.
        """
        # assign the local variables.
        cube_min_index, cube_max_index, sint, sint_specified = interpolate_vars[:4]
        dataset_title, num_values_per_datapoint, N, chunk_size = getdata_vars
        
        def single_timepoint(chunk_min_ranges, chunk_max_ranges, chunk_step = (1, 1, 1)):
            # cutout data from the specified chunk.
            return zarr_store[timepoint,
                              chunk_min_ranges[2] : chunk_max_ranges[2] + chunk_step[2],
                              chunk_min_ranges[1] : chunk_max_ranges[1] + chunk_step[1],
                              chunk_min_ranges[0] : chunk_max_ranges[0] + chunk_step[0]]
        
        def windfarm_timepoint(chunk_min_ranges, chunk_max_ranges, chunk_step = (1, 1, 1)):
            # cutout data from the specified chunk.
            return zarr_store[timepoint[0],
                              timepoint[1],
                              timepoint[2],
                              chunk_min_ranges[2] : chunk_max_ranges[2] + chunk_step[2],
                              chunk_min_ranges[1] : chunk_max_ranges[1] + chunk_step[1],
                              chunk_min_ranges[0] : chunk_max_ranges[0] + chunk_step[0]]
        
        # read zarr function map.
        read_zarr_functions = defaultdict(lambda: single_timepoint, {'diurnal_windfarm': windfarm_timepoint})
        read_zarr_function = read_zarr_functions[dataset_title]
        
        # initialize the interpolation lookup table. use sint_specified when one of the 'z_linear*' step-down methods is being used.
        sint_lookup_table = sint_specified if 'z_linear' in sint else sint
        self.init_interpolation_lookup_table(sint = sint_lookup_table, read_metadata = True)
        
        # empty chunk group array (up to eight 64-cube chunks).
        zarr_matrix = np.zeros((chunk_size[2] * 2, chunk_size[1] * 2, chunk_size[0] * 2, num_values_per_datapoint))
        
        # the collection of local output data that will be returned to fill the complete output_data array.
        local_output_data = []
        
        chunk_min_xyz = map_data[0][0]
        chunk_max_xyz = map_data[0][1]
        chunk_min_mod_xyz = map_data[0][2]
        chunk_max_mod_xyz = map_data[0][3]
        chunk_min_x, chunk_min_y, chunk_min_z = chunk_min_xyz[0], chunk_min_xyz[1], chunk_min_xyz[2]
        chunk_max_x, chunk_max_y, chunk_max_z = chunk_max_xyz[0], chunk_max_xyz[1], chunk_max_xyz[2]

        # read in the chunks separately if they wrap around periodic boundaries.
        if any(chunk_min_xyz < 0) or any(chunk_max_xyz >= N):
            # get the origin points for each chunk in the bucket.
            chunk_origin_groups = np.array([[x, y, z]
                                            for z in range(chunk_min_z, chunk_max_z + 1, chunk_size[2])
                                            for y in range(chunk_min_y, chunk_max_y + 1, chunk_size[1])
                                            for x in range(chunk_min_x, chunk_max_x + 1, chunk_size[0])])
            
            # adjust the chunk origin points to the chunk domain size for filling the empty chunk group array.
            chunk_origin_points = chunk_origin_groups - chunk_origin_groups[0]

            # get the chunk origin group inside the dataset domain.
            chunk_origin_groups = chunk_origin_groups % N

            for chunk_origin_point, chunk_origin_group in zip(chunk_origin_points, chunk_origin_groups):
                zarr_matrix[chunk_origin_point[2] : chunk_origin_point[2] + chunk_size[2],
                            chunk_origin_point[1] : chunk_origin_point[1] + chunk_size[1],
                            chunk_origin_point[0] : chunk_origin_point[0] + chunk_size[0]] = read_zarr_function(chunk_min_ranges = chunk_origin_group,
                                                                                                                chunk_max_ranges = chunk_origin_group,
                                                                                                                chunk_step = chunk_size)
        else:
            # read in all chunks at once, and use default chunk_step (1, 1, 1).
            zarr_matrix[:chunk_max_mod_xyz[2] - chunk_min_mod_xyz[2] + 1,
                        :chunk_max_mod_xyz[1] - chunk_min_mod_xyz[1] + 1,
                        :chunk_max_mod_xyz[0] - chunk_min_mod_xyz[0] + 1] = read_zarr_function(chunk_min_ranges = chunk_min_mod_xyz,
                                                                                               chunk_max_ranges = chunk_max_mod_xyz)

        # iterate over the points to interpolate.            
        for point, datapoint, center_point, bucket_info, original_point_index in map_data[1:]:
            bucket_min_xyz = (datapoint - chunk_min_xyz - cube_min_index) % N
            bucket_max_xyz = (datapoint - chunk_min_xyz + cube_max_index + 1) % N
            # update bucket_max_xyz any dimension is less than the corresponding dimension in bucket_min_xyz. this is
            # necessary to handle points on the boundary of single-chunk dimensions in the zarr store.
            mask = bucket_max_xyz < bucket_min_xyz
            bucket_max_xyz = bucket_max_xyz + (chunk_size * mask)

            bucket = zarr_matrix[bucket_min_xyz[2] : bucket_max_xyz[2],
                                 bucket_min_xyz[1] : bucket_max_xyz[1],
                                 bucket_min_xyz[0] : bucket_max_xyz[0]]

            # interpolate the points and use a lookup table for faster interpolations.
            local_output_data.append((original_point_index, (point, self.spatial_interpolate(center_point, bucket, bucket_info, interpolate_vars))))
        
        return local_output_data
    