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
import duckdb
import pathlib
import itertools
import subprocess
import numpy as np
import pandas as pd
from collections import defaultdict
from SciServer import Authentication
from concurrent.futures import ThreadPoolExecutor
from giverny.turbulence_gizmos.basic_gizmos import *

class turb_dataset():
    def __init__(self, dataset_title = '', output_path = '', auth_token = '', rewrite_interpolation_metadata = False,
                 json_url = 'https://raw.githubusercontent.com/sciserver/giverny/refs/heads/main/metadata/configs/jhtdb-config.json'):
        """
        initialize the class.
        """
        # load the json metadata.
        self.metadata = load_json_metadata(json_url)
        
        # check that dataset_title is a valid dataset title.
        check_dataset_title(self.metadata, dataset_title)
        
        # dask maximum processes constant.
        self.maximum_processes = self.metadata['constants']['maximum_processes']
        
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
                
    def init_interpolation_cube_size_lookup_table(self, metadata, sint = 'none'):
        """
        pickled interpolation cube sizes table.
        """
        # the interpolation cube size indices are only read when called from the init_constants function.
        interp_cube_sizes = {spatial_method['code']: spatial_method['bucketIndices'] for spatial_method in metadata['spatial_methods']}
        
        # the bucket size is the same for all spatial operators, so we only use the spatial method portion of 'sint'.
        sint_cube_size = sint.split('_')[0]

        # lookup the interpolation cube size indices.
        self.cube_min_index, self.cube_max_index = interp_cube_sizes[sint_cube_size]
    
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
        
        # retrieve the non-periodic and regularly spaced spatial axes.
        self.nonperiodic_regular_axes = get_nonperiodic_and_regular_spacing_spatial_axes(self.metadata, self.dataset_title, self.var)
        
        # set the dataset name to be used in the cutout hdf5 file.
        self.dataset_name = self.var + '_' + str(self.timepoint_original).zfill(4)
        
        # retrieve the list of datasets processed by the giverny code.
        giverny_datasets = get_giverny_datasets()
        
        if self.dataset_title in giverny_datasets:
            if query_type == 'getdata':
                # initialize the interpolation cube size lookup table.
                self.init_interpolation_cube_size_lookup_table(self.metadata, self.sint)
                
                # interpolate function variables.
                self.interpolate_vars = [self.cube_min_index, self.cube_max_index, self.sint, self.spacing, self.lookup_N]
                
                # getData variables.
                self.getdata_vars = [self.dataset_title, self.num_values_per_datapoint, self.N, self.chunk_size, self.nonperiodic_regular_axes]
            
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
    
    def spatial_interpolate(self, p, u, interpolate_vars):
        """
        spatial interpolating functions to compute the kernel, extract subcube and convolve.
        
        vars:
         - p is an np.array(3) containing the three coordinates.
        """
        # assign the local variables.
        cube_min_index, cube_max_index, sint, spacing, lookup_N = interpolate_vars
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
        num_processes = min(self.maximum_processes, num_chunks)
        
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
        read_zarr_functions = defaultdict(lambda: single_timepoint,
                                          {'diurnal_windfarm': windfarm_timepoint, 'nbl_windfarm': windfarm_timepoint})
        
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
        
        # handle diurnal windfarm soiltemperature variable because this variable has non-uniform z-axis grid spacing. currently only "none" interpolation
        # is allowed. the "center_points" are found between the z-gridpoints, and "datapoints" are mapped to the floor z-gridpoints.
        if self.dataset_title == 'diurnal_windfarm' and self.var == 'soiltemperature':
            # convert the points to their center points position between grid points.
            x_center = ((points[:, 0] / self.spacing[0]) % 1) + self.cube_min_index
            y_center = ((points[:, 1] / self.spacing[1]) % 1) + self.cube_min_index
            
            z_points = points[:, 2]
            z_grid = self.spacing[2]
            
            # find the index in the z-gridpoint list where each of the z-points would be inserted.
            z_lower_indices = np.searchsorted(z_grid, -z_points, side = 'right') - 1
            # handles the bottom and top boundary gridpoints. shifts the index for z_point == z_grid[-1] down by 2 to account for needing an
            # index before and after the specified z-point (the after gridpoint in this case is the specified z-point == z_grid[-1]).
            z_lower_indices = np.clip(z_lower_indices, 0, len(z_grid) - 2)
            
            z_low = z_grid[z_lower_indices]
            z_high = z_grid[z_lower_indices + 1]

            # calculate z-points centered position within each grid cell.
            z_center = ((-z_points - z_low) / (z_high - z_low)) % 1
            # center points.
            center_points = np.column_stack([x_center, y_center, z_center])
            
            # convert the points to gridded datapoints.
            x_datapoints = np.floor(points[:, 0] / self.spacing[0]).astype(int) % self.N[0]
            y_datapoints = np.floor(points[:, 1] / self.spacing[1]).astype(int) % self.N[1]

            # find the index in the z-gridpoint list where each of the z-points would be inserted.
            z_datapoints = np.searchsorted(z_grid, -z_points, side = 'right') - 1
            # handles the bottom and top boundary gridpoints. shifts the index for z_point == z_grid[-1] down by 1 to account for needing a
            # index before the specified z-point (equivalent to np.floor() for the x- and y-points).
            # no modulo needed since the z-axis is non-periodic and all queried points are restricted to the z-domain.
            z_datapoints = np.clip(z_datapoints, 0, len(z_grid) - 1)
            
            # datapoints.
            datapoints = np.column_stack([x_datapoints, y_datapoints, z_datapoints])
        else:
            # handles all other variables of the "diurnal_windfarm" dataset as well as all other datasets.
            # convert the points to the center point position within their own bucket.
            center_points = ((points / self.spacing) % 1) + self.cube_min_index
            # convert the points to gridded datapoints. there is a +0.5 point shift because the finite differencing methods would otherwise add +0.5 to center_points when
            # interpolating the values. shifting the datapoints up by +0.5 adjusts the bucket up one grid point so the center_points do not needed to be shifted up by +0.5.
            if self.sint in ['fd4noint_gradient', 'fd6noint_gradient', 'fd8noint_gradient',
                             'fd4noint_laplacian', 'fd6noint_laplacian', 'fd8noint_laplacian',
                             'fd4noint_hessian', 'fd6noint_hessian', 'fd8noint_hessian']:
                datapoints = np.floor((points + self.spacing * 0.5) / self.spacing).astype(int) % self.N
            else:
                datapoints = np.floor(points / self.spacing).astype(int) % self.N
        
        # determine if we need to insert a 0-plane of gridpoints to handle the velocity-w and sgsenergy boundary condition at z = 0.
        z_min_boundary_flags = np.full(len(datapoints), fill_value = False)
        if self.dataset_title in ['sabl2048low', 'sabl2048high', 'stsabl2048low', 'stsabl2048high'] and self.var in ['velocity', 'sgsenergy']:
            if self.var == 'velocity':
                # the points array is duplicated to query the velocity-uv and velocity-w components together. we only want to apply the
                # boundary condition, w = 0 at z = 0, to the copy of points that correspond to the velocity-w component query. 
                num_unique_points = int(len(points) / 2)
                z_min_boundary_flags[num_unique_points:][points[num_unique_points:, 2] < ((self.cube_min_index + 1) * self.dz)] = True
            elif self.var == 'sgsenergy':
                # apply the boundary condition, e = 0 at z = 0.
                z_min_boundary_flags[points[:, 2] < ((self.cube_min_index + 1) * self.dz)] = True
            
            # adjust the datapoints +1 along the z-axis to account for inserting a zero-plane of gridpoints at z = 0.
            datapoints[z_min_boundary_flags, 2] = (datapoints[z_min_boundary_flags, 2] + 1) % self.N[2]
        
        # calculate the minimum and maximum chunk (x, y, z) corner point for each point in datapoints.
        chunk_min_xyzs = ((datapoints - self.cube_min_index) - ((datapoints - self.cube_min_index) % self.chunk_size))
        chunk_max_xyzs = ((datapoints + self.cube_max_index) + (self.chunk_size - ((datapoints + self.cube_max_index) % self.chunk_size) - 1))
        chunk_min_mod_xyzs = chunk_min_xyzs % self.N
        chunk_max_mod_xyzs = chunk_max_xyzs % self.N
        # chunk volumes.
        chunk_volumes = np.prod(chunk_max_xyzs - chunk_min_xyzs + 1, axis = 1)        
        # create the chunk keys for each chunk group.
        chunk_keys = [f'{chunk_origin_group[0][0]}_{chunk_origin_group[0][1]}_{chunk_origin_group[0][2]}_' + \
                      f'{chunk_origin_group[1][0]}_{chunk_origin_group[1][1]}_{chunk_origin_group[1][2]}_{z_min_boundary_flag}'
                      for chunk_origin_group, z_min_boundary_flag in zip(np.stack([chunk_min_xyzs, chunk_max_xyzs], axis = 1), z_min_boundary_flags)]
        
        # save the original indices for points, which corresponds to the orderering of the user-specified
        # points. these indices will be used for sorting output_data back to the user-specified points ordering.
        original_points_indices = np.arange(len(points))
        # zip the data. sort by volume first so that all fully overlapped chunk groups can be easily found.
        zipped_data = sorted(zip(chunk_volumes, chunk_keys, points, datapoints, center_points, z_min_boundary_flags,
                                 chunk_min_xyzs, chunk_max_xyzs, chunk_min_mod_xyzs, chunk_max_mod_xyzs,
                                 original_points_indices), key = lambda x: (-1 * x[0], x[1]))
        
        # map the bucket points to their chunks.
        chunk_data_map = defaultdict(list)
        # chunk key map used for storing all subdivided chunk groups to find fully overlapped chunk groups.
        chunk_map = {}
        
        for chunk_volume, chunk_key, point, datapoint, center_point, z_min_boundary_flag, \
            chunk_min_xyz, chunk_max_xyz, chunk_min_mod_xyz, chunk_max_mod_xyz, \
            original_point_index in zipped_data:
            # update the chunk key if the chunk group is fully contained in another larger chunk group.
            updated_chunk_key = chunk_key
            if chunk_key in chunk_map:
                updated_chunk_key = chunk_map[chunk_key]
            elif chunk_volume != chunk_cube_size:
                chunk_map = self.subdivide_chunk_group(chunk_map, chunk_key, chunk_min_xyz, chunk_max_xyz, z_min_boundary_flag, chunk_size_array, empty_array)

            # assign to chunk_data_map.
            if updated_chunk_key not in chunk_data_map:
                chunk_data_map[updated_chunk_key].append((chunk_min_xyz, chunk_max_xyz, chunk_min_mod_xyz, chunk_max_mod_xyz, z_min_boundary_flag))

            chunk_data_map[updated_chunk_key].append((point, datapoint, center_point, original_point_index))
        
        return np.array(list(chunk_data_map.values()), dtype = object)
    
    def subdivide_chunk_group(self, chunk_map, chunk_key, chunk_min_xyz, chunk_max_xyz, z_min_boundary_flag, chunk_size_array, empty_array):
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
        chunk_keys = [f'{chunk_origin_group[0][0]}_{chunk_origin_group[0][1]}_{chunk_origin_group[0][2]}_' + \
                      f'{chunk_origin_group[1][0]}_{chunk_origin_group[1][1]}_{chunk_origin_group[1][2]}_{z_min_boundary_flag}'
                      for chunk_origin_group in np.stack([chunk_mins, chunk_maxs], axis = 1)]
        for key in chunk_keys:
            chunk_map[key] = chunk_key

        return chunk_map
    
    def read_database_files_getdata(self, chunk_data_map):
        """
        submit the points for reading and interpolation.
        """
        num_chunks = len(chunk_data_map)
        num_processes = min(self.maximum_processes, num_chunks)
        
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
        cube_min_index, cube_max_index, sint = interpolate_vars[:3]
        dataset_title, num_values_per_datapoint, N, chunk_size, nonperiodic_regular_axes = getdata_vars
        
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
        read_zarr_functions = defaultdict(lambda: single_timepoint,
                                          {'diurnal_windfarm': windfarm_timepoint, 'nbl_windfarm': windfarm_timepoint})
        read_zarr_function = read_zarr_functions[dataset_title]
        
        self.init_interpolation_lookup_table(sint = sint, read_metadata = True)
        
        # empty chunk group array (up to eight 64-cube chunks).
        zarr_matrix = np.zeros((chunk_size[2] * 2, chunk_size[1] * 2, chunk_size[0] * 2, num_values_per_datapoint))
        
        # the collection of local output data that will be returned to fill the complete output_data array.
        local_output_data = []
        
        chunk_min_xyz = map_data[0][0]
        chunk_max_xyz = map_data[0][1]
        chunk_min_mod_xyz = map_data[0][2]
        chunk_max_mod_xyz = map_data[0][3]
        z_min_boundary_flag = map_data[0][4]
        
        chunk_min_x, chunk_min_y, chunk_min_z = chunk_min_xyz[0], chunk_min_xyz[1], chunk_min_xyz[2]
        chunk_max_x, chunk_max_y, chunk_max_z = chunk_max_xyz[0], chunk_max_xyz[1], chunk_max_xyz[2]

        # read in the chunks separately if they wrap around dataset boundaries.
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
            
            # insert 0-plane values at the z = 0 boundary to handle the velocity-w and sgsenergy boundary condition of the sabl datasets.
            if z_min_boundary_flag:
                zarr_matrix = np.insert(zarr_matrix, chunk_size[2], 0, axis = 0)
            
            # extrapolate points near the boundary of the non-periodic axes, so that the full spatial interpolation method can be applied correctly.
            for nonperiodic_axis in nonperiodic_regular_axes:
                if chunk_min_xyz[nonperiodic_axis] < 0:
                    # handle extrapolations at the lower axis boundary.
                    boundary_index = chunk_size[nonperiodic_axis]
                    
                    # create empty slices for each axis.
                    first_gridpoint = [slice(None)] * 4
                    second_gridpoint = [slice(None)] * 4
                    # update the slices to point to the first and second gridpoints.
                    first_gridpoint[2 - nonperiodic_axis] = boundary_index
                    second_gridpoint[2 - nonperiodic_axis] = boundary_index + 1
                    # convert to tuples for array slicing.
                    first_gridpoint = tuple(first_gridpoint)
                    second_gridpoint = tuple(second_gridpoint)
                    
                    for cube_index in range(cube_min_index):
                        extrapolated_gridpoint = [slice(None)] * 4
                        extrapolated_gridpoint[2 - nonperiodic_axis] = boundary_index - (cube_index + 1)
                        extrapolated_gridpoint = tuple(extrapolated_gridpoint)
                        
                        zarr_matrix[extrapolated_gridpoint] = zarr_matrix[first_gridpoint] - (cube_index + 1) * (zarr_matrix[second_gridpoint] - zarr_matrix[first_gridpoint])
                elif chunk_max_xyz[nonperiodic_axis] > N[nonperiodic_axis]:
                    # handle extrapolations at the upper axis boundary.
                    boundary_index = chunk_size[nonperiodic_axis] - 1
                    
                    # create empty slices for each axis.
                    last_gridpoint = [slice(None)] * 4
                    second_gridpoint = [slice(None)] * 4
                    # update the slices to point to the last and second-to-last gridpoints.
                    last_gridpoint[2 - nonperiodic_axis] = boundary_index
                    second_gridpoint[2 - nonperiodic_axis] = boundary_index - 1
                    # convert to tuples for array slicing.
                    last_gridpoint = tuple(last_gridpoint)
                    second_gridpoint = tuple(second_gridpoint)
                    
                    for cube_index in range(cube_max_index):
                        extrapolated_gridpoint = [slice(None)] * 4
                        extrapolated_gridpoint[2 - nonperiodic_axis] = boundary_index + (cube_index + 1)
                        extrapolated_gridpoint = tuple(extrapolated_gridpoint)
                        
                        zarr_matrix[extrapolated_gridpoint] = zarr_matrix[last_gridpoint] + (cube_index + 1) * (zarr_matrix[last_gridpoint] - zarr_matrix[second_gridpoint])
        else:
            # read in all chunks at once, and use default chunk_step (1, 1, 1).
            zarr_matrix[:chunk_max_mod_xyz[2] - chunk_min_mod_xyz[2] + 1,
                        :chunk_max_mod_xyz[1] - chunk_min_mod_xyz[1] + 1,
                        :chunk_max_mod_xyz[0] - chunk_min_mod_xyz[0] + 1] = read_zarr_function(chunk_min_ranges = chunk_min_mod_xyz,
                                                                                               chunk_max_ranges = chunk_max_mod_xyz)

        # iterate over the points to interpolate.
        for point, datapoint, center_point, original_point_index in map_data[1:]:
            bucket_min_xyz = (datapoint - chunk_min_xyz - cube_min_index) % N
            bucket_max_xyz = (datapoint - chunk_min_xyz + cube_max_index + 1) % N
            # update bucket_max_xyz for any dimension that is less than the corresponding dimension in bucket_min_xyz. this is
            # necessary to handle points on the boundary of single-chunk dimensions in the zarr store.
            mask = bucket_max_xyz < bucket_min_xyz
            bucket_max_xyz = bucket_max_xyz + (chunk_size * mask)
            
            bucket = zarr_matrix[bucket_min_xyz[2] : bucket_max_xyz[2],
                                 bucket_min_xyz[1] : bucket_max_xyz[1],
                                 bucket_min_xyz[0] : bucket_max_xyz[0]]

            # interpolate the points and use a lookup table for faster interpolations.
            local_output_data.append((original_point_index, (point, self.spatial_interpolate(center_point, bucket, interpolate_vars))))
        
        return local_output_data
    