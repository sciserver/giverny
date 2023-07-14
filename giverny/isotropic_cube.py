import os
import sys
import dill
import glob
import math
import zarr
import shutil
import pathlib
import warnings
import subprocess
import numpy as np
import SciServer.CasJobs as cj
from threading import Thread
from collections import defaultdict
from SciServer import Authentication
from dask.distributed import Client, LocalCluster
from giverny.turbulence_gizmos.basic_gizmos import *

# installs morton-py if necessary.
try:
    import morton
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'morton-py'])
finally:
    import morton

class iso_cube:
    def __init__(self, dataset_title = '', output_path = '', cube_dimensions = 3):
        """
        initialize the class.
        """
        # turn off the dask warning for scattering large objects to the workers.
        warnings.filterwarnings("ignore", message = ".*Large object of size.*detected in task graph")
        
        # check that dataset_title is a valid dataset title.
        check_dataset_title(dataset_title)
        
        # cube size.
        self.N = get_dataset_resolution(dataset_title)
        # conversion factor between the cube size and a domain on [0, 2pi].
        self.dx = 2 * np.pi / self.N
        
        # setting up Morton curve.
        bits = int(math.log(self.N, 2))
        self.mortoncurve = morton.Morton(dimensions = cube_dimensions, bits = bits)
        
        # turbulence dataset name, e.g. "isotropic8192" or "isotropic1024fine".
        self.dataset_title = dataset_title
        
        # set the directory for saving any output files.
        self.output_path = output_path.strip()
        if self.output_path == '':
            # get the SciServer user name.
            user = Authentication.getKeystoneUserWithToken(Authentication.getToken()).userName
        
            self.output_path = pathlib.Path(f'/home/idies/workspace/Temporary/{user}/scratch/turbulence_output')
        else:
            self.output_path = pathlib.Path(self.output_path).joinpath('turbulence_output')
        
        # create the output directory if it does not already exist.
        create_output_folder(self.output_path)
        
        # set the directory for reading the pickled files.
        self.pickle_dir = pathlib.Path(f'/home/idies/workspace/turb/data01_01/zarr/turbulence_pickled')
        
        # set the backup directory for reading the pickled files.
        self.pickle_dir_backup = pathlib.Path(f'/home/idies/workspace/turb/data02_01/zarr/turbulence_pickled_back')
        
        # get a cache of the metadata for the database files.
        self.init_cache()
    
    """
    initialization functions.
    """
    def init_cache(self):
        """
        pickled SQL metadata.
        """
        self.cache = self.read_pickle_file(self.dataset_title + '_metadata.pickle')
        
    def init_filepaths(self):
        """
        pickled filepaths.
        """
        # pickled production filepaths.
        self.filepaths = self.read_pickle_file(self.dataset_title + '_database_filepaths.pickle')
        
        # pickled backup filepaths.
        self.filepaths_backup = self.read_pickle_file(self.dataset_title + '_database_filepaths_backup.pickle')
                
    def init_cornercode_file_map(self):
        """
        pickled db file cornercodes to filenames map.
        """
        # pickled db file cornercodes to production filenames map.
        self.cornercode_file_map = self.read_pickle_file(self.dataset_title + f'_cornercode_file_map.pickle')
        
        # pickled db file cornercodes to backup filenames map.
        self.cornercode_file_map_backup = self.read_pickle_file(self.dataset_title + f'_cornercode_file_map_backup.pickle')
    
    def init_interpolation_lookup_table(self):
        """
        pickled interpolation lookup table.
        """
        # lookup table resolution.
        self.NB = 10**5
        
        # read in the interpolation lookup table for self.sint.
        if self.sint != 'none':
            # pickled interpolation coefficient lookup table.
            self.LW = self.read_pickle_file(f'{self.sint}_lookup_table.pickle')
            
            # read in the spatial interpolation lookup table that is used in the calculation of other interpolation methods.
            if self.sint in ['fd4lag4_g', 'm1q4_g', 'm2q8_g',
                             'fd4lag4_l',
                             'm2q8_h']:
                # convert self.sint to the needed spatial interpolation name.
                sint_name = self.sint.split('_')[0].replace('fd4', '')
                
                # pickled interpolation coefficient lookup table.
                self.interpolation_LW = self.read_pickle_file(f'{sint_name}_lookup_table.pickle')
                    
                # read in the gradient coefficient lookup table that is used in the calculation of the m2q8 spline hessian.
                if self.sint == 'm2q8_h':
                    # convert self.sint to the needed gradient interpolation name.
                    sint_name = self.sint.replace('_h', '_g')
                    
                    # pickled gradient coefficient lookup table.
                    self.gradient_LW = self.read_pickle_file(f'{sint_name}_lookup_table.pickle')
            # read in the laplacian interpolation lookup table that is used in the calculation of other interpolation methods.
            elif self.sint in ['fd4noint_h', 'fd6noint_h', 'fd8noint_h']:
                # convert self.sint to the needed gradient interpolation name.
                sint_name = self.sint.replace('_h', '_l')
                
                # pickled laplacian coefficient lookup table.
                self.laplacian_LW = self.read_pickle_file(f'{sint_name}_lookup_table.pickle')
                
    def init_interpolation_cube_size_lookup_table(self):
        """
        pickled interpolation cube sizes table.
        """
        # pickled interpolation cube sizes lookup table.
        interp_cube_sizes = self.read_pickle_file('interpolation_cube_size_lookup_table.pickle')
                
        # lookup the interpolation cube size indices.
        self.cube_min_index, self.cube_max_index = interp_cube_sizes[self.sint]
    
    def init_constants(self, var, var_original, timepoint, sint, num_values_per_datapoint, c):
        """
        initialize the constants.
        """
        self.var = var
        self.var_name = var_original
        self.timepoint = timepoint
        self.sint = sint
        self.num_values_per_datapoint = num_values_per_datapoint
        self.bytes_per_datapoint = c['bytes_per_datapoint']
        self.voxel_side_length = c['voxel_side_length']
        self.missing_value_placeholder = c['missing_value_placeholder']
        self.database_file_disk_index = c['database_file_disk_index']
        self.dask_maximum_processes = c['dask_maximum_processes']
        self.decimals = c['decimals']
        self.chunk_size = c['chunk_size']
        self.file_size = c['file_size']
        
        # set the byte order for reading the data from the files.
        self.dt = np.dtype(np.float32)
        self.dt = self.dt.newbyteorder('<')
        
        # set the dataset name to be used in the hdf5 file. 1 is added to timepoint because the original timepoint was converted to a 0-based index.
        self.dataset_name = get_output_variable_name(var_original) + '_' + str(timepoint + 1).zfill(4)
        
        # get map of the filepaths for all of the dataset files.
        self.init_filepaths()
        
        # get a map of the files to cornercodes for all of the dataset files.
        self.init_cornercode_file_map()
    
    """
    interpolation functions.
    """
    def interpolate(self, p, u):
        """
        interpolating functions to compute the kernel, extract subcube and convolve.
        
        vars:
         - p is an np.array(3) containing the three coordinates.
        """
        if self.sint in ['lag4', 'm1q4', 'lag6', 'lag8', 'm2q8']:
            # spatial interpolations.
            ix = p.astype(np.int32)
            fr = p - ix
            
            # get the coefficients.
            gx = self.LW[int(self.NB * fr[0])]
            gy = self.LW[int(self.NB * fr[1])]
            gz = self.LW[int(self.NB * fr[2])]
            
            # create the 3D kernel from the outer product of the 1d kernels.
            gk = np.einsum('i,j,k', gz, gy, gx)

            return np.einsum('ijk,ijkl->l', gk, u)
        elif self.sint == 'none':
            # 'none' spatial interpolation.
            ix = np.floor(p + 0.5).astype(np.int32)
            
            return np.array(u[ix[2], ix[1], ix[0], :])
        elif self.sint in ['fd4noint_g', 'fd6noint_g', 'fd8noint_g',
                           'fd4noint_l', 'fd6noint_l', 'fd8noint_l']:
            # gradient and laplacian finite differences.
            ix = np.floor(p + 0.5).astype(int)
            # diagonal coefficients.
            fd_coeff = self.LW
            
            # diagnoal components.
            component_x = u[ix[2], ix[1], ix[0] - self.cube_min_index : ix[0] + self.cube_max_index, :]
            component_y = u[ix[2], ix[1] - self.cube_min_index : ix[1] + self.cube_max_index, ix[0], :]
            component_z = u[ix[2] - self.cube_min_index : ix[2] + self.cube_max_index, ix[1], ix[0], :]

            dvdx = np.inner(fd_coeff,component_x.T)  
            dvdy = np.inner(fd_coeff,component_y.T)
            dvdz = np.inner(fd_coeff,component_z.T)
            
            # return gradient values.
            if '_g' in self.sint:
                return np.stack((dvdx, dvdy, dvdz), axis = 1)
            # return laplacian values.
            elif '_l' in self.sint:
                return dvdx + dvdy + dvdz
        elif self.sint in ['fd4noint_h', 'fd6noint_h', 'fd8noint_h']:
            # hessian finite differences.
            ix = np.floor(p + 0.5).astype(int)
            # diagonal coefficients.
            fd_coeff_l = self.laplacian_LW
            # off-diagonal coefficients.
            fd_coeff_h = self.LW
            
            # diagnoal components.
            component_x = u[ix[2], ix[1], ix[0] - self.cube_min_index : ix[0] + self.cube_max_index, :]
            component_y = u[ix[2], ix[1] - self.cube_min_index : ix[1] + self.cube_max_index, ix[0], :]
            component_z = u[ix[2] - self.cube_min_index : ix[2] + self.cube_max_index, ix[1], ix[0], :]

            uii = np.inner(fd_coeff_l, component_x.T)  
            ujj = np.inner(fd_coeff_l, component_y.T)
            ukk = np.inner(fd_coeff_l, component_z.T)

            # off-diagonal components.
            if self.sint == 'fd4noint_h':
                component_xy = np.array([u[ix[2],ix[1]+2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]-2,:],u[ix[2],ix[1]+2,ix[0]-2,:],
                                         u[ix[2],ix[1]+1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]-1,:],u[ix[2],ix[1]+1,ix[0]-1,:]])
                component_xz = np.array([u[ix[2]+2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]-2,:],u[ix[2]+2,ix[1],ix[0]-2,:],
                                         u[ix[2]+1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]-1,:],u[ix[2]+1,ix[1],ix[0]-1,:]])
                component_yz = np.array([u[ix[2]+2,ix[1]+2,ix[0],:],u[ix[2]-2,ix[1]+2,ix[0],:],u[ix[2]-2,ix[1]-2,ix[0],:],u[ix[2]+2,ix[1]-2,ix[0],:],
                                         u[ix[2]+1,ix[1]+1,ix[0],:],u[ix[2]-1,ix[1]+1,ix[0],:],u[ix[2]-1,ix[1]-1,ix[0],:],u[ix[2]+1,ix[1]-1,ix[0],:]])
            elif self.sint == 'fd6noint_h':
                component_xy = np.array([u[ix[2],ix[1]+3,ix[0]+3,:],u[ix[2],ix[1]-3,ix[0]+3,:],u[ix[2],ix[1]-3,ix[0]-3,:],u[ix[2],ix[1]+3,ix[0]-3,:],
                                         u[ix[2],ix[1]+2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]-2,:],u[ix[2],ix[1]+2,ix[0]-2,:],
                                         u[ix[2],ix[1]+1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]-1,:],u[ix[2],ix[1]+1,ix[0]-1,:]])
                component_xz = np.array([u[ix[2]+3,ix[1],ix[0]+3,:],u[ix[2]-3,ix[1],ix[0]+3,:],u[ix[2]-3,ix[1],ix[0]-3,:],u[ix[2]+3,ix[1],ix[0]-3,:],
                                         u[ix[2]+2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]-2,:],u[ix[2]+2,ix[1],ix[0]-2,:],
                                         u[ix[2]+1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]-1,:],u[ix[2]+1,ix[1],ix[0]-1,:]])
                component_yz = np.array([u[ix[2]+3,ix[1]+3,ix[0],:],u[ix[2]-3,ix[1]+3,ix[0],:],u[ix[2]-3,ix[1]-3,ix[0],:],u[ix[2]+3,ix[1]-3,ix[0],:],
                                         u[ix[2]+2,ix[1]+2,ix[0],:],u[ix[2]-2,ix[1]+2,ix[0],:],u[ix[2]-2,ix[1]-2,ix[0],:],u[ix[2]+2,ix[1]-2,ix[0],:],
                                         u[ix[2]+1,ix[1]+1,ix[0],:],u[ix[2]-1,ix[1]+1,ix[0],:],u[ix[2]-1,ix[1]-1,ix[0],:],u[ix[2]+1,ix[1]-1,ix[0],:]])
            elif self.sint == 'fd8noint_h':
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

            uij = np.inner(fd_coeff_h, component_xy.T) 
            uik = np.inner(fd_coeff_h, component_xz.T) 
            ujk = np.inner(fd_coeff_h, component_yz.T)
            
            return np.array([uii,uij,uik,ujj,ujk,ukk])
        elif self.sint in ['m1q4_g', 'm2q8_g']:
            # gradient spline differentiations.
            ix = p.astype(int) 
            fr = p - ix
            
            # spatial spline coefficients.
            gx = self.interpolation_LW[int(self.NB * fr[0])]
            gy = self.interpolation_LW[int(self.NB * fr[1])]
            gz = self.interpolation_LW[int(self.NB * fr[2])]
            
            # gradient spline coefficients.
            gx_G = self.LW[int(self.NB * fr[0])]
            gy_G = self.LW[int(self.NB * fr[1])]
            gz_G = self.LW[int(self.NB * fr[2])]
            
            gk_x = np.einsum('i,j,k', gz, gy, gx_G)
            gk_y = np.einsum('i,j,k', gz, gy_G, gx)
            gk_z = np.einsum('i,j,k', gz_G, gy, gx)
            
            d = u[ix[2] - self.cube_min_index : ix[2] + self.cube_max_index + 1,
                  ix[1] - self.cube_min_index : ix[1] + self.cube_max_index + 1,
                  ix[0] - self.cube_min_index : ix[0] + self.cube_max_index + 1, :] / self.dx
            
            # dudx,dvdx,dwdx.
            dvdx = np.einsum('ijk,ijkl->l', gk_x, d)
            # dudy,dvdy,dwdy.
            dvdy = np.einsum('ijk,ijkl->l', gk_y, d)
            # dudz,dvdz,dwdz.
            dvdz = np.einsum('ijk,ijkl->l', gk_z, d)
            
            return np.stack((dvdx, dvdy, dvdz), axis = 1)
        elif self.sint == 'm2q8_h':
            # hessian spline differentiation.
            ix = p.astype('int')
            fr = p - ix
            
            # spatial spline coefficients.
            gx = self.interpolation_LW[int(self.NB * fr[0])]
            gy = self.interpolation_LW[int(self.NB * fr[1])]
            gz = self.interpolation_LW[int(self.NB * fr[2])]
            
            # gradient spline coefficients.
            gx_G = self.gradient_LW[int(self.NB * fr[0])]
            gy_G = self.gradient_LW[int(self.NB * fr[1])]
            gz_G = self.gradient_LW[int(self.NB * fr[2])]
            
            # hessian spline coefficients.
            gx_GG = self.LW[int(self.NB * fr[0])]
            gy_GG = self.LW[int(self.NB * fr[1])]
            gz_GG = self.LW[int(self.NB * fr[2])]

            gk_xx = np.einsum('i,j,k', gz, gy, gx_GG)
            gk_yy = np.einsum('i,j,k', gz, gy_GG, gx)
            gk_zz = np.einsum('i,j,k', gz_GG, gy, gx)
            gk_xy = np.einsum('i,j,k', gz, gy_G, gx_G)
            gk_xz = np.einsum('i,j,k', gz_G, gy, gx_G)
            gk_yz = np.einsum('i,j,k', gz_G, gy_G, gx)     

            d = u[ix[2] - self.cube_min_index : ix[2] + self.cube_max_index + 1,
                  ix[1] - self.cube_min_index : ix[1] + self.cube_max_index + 1,
                  ix[0] - self.cube_min_index : ix[0] + self.cube_max_index + 1, :] / self.dx / self.dx

            uii = np.einsum('ijk,ijkl->l', gk_xx, d)
            ujj = np.einsum('ijk,ijkl->l', gk_yy, d)
            ukk = np.einsum('ijk,ijkl->l', gk_zz, d)
            uij = np.einsum('ijk,ijkl->l', gk_xy, d)
            uik = np.einsum('ijk,ijkl->l', gk_xz, d)
            ujk = np.einsum('ijk,ijkl->l', gk_yz, d)                              

            return np.stack((uii,uij,uik,ujj,ujk,ukk), axis = 1)
        elif self.sint in ['fd4lag4_g', 'fd4lag4_l']:
            # gradient and laplacian finite difference with spatial interpolation.
            ix = p.astype(int) 
            fr = p - ix      
            
            # spatial interpolation coefficients.
            gx = self.interpolation_LW[int(self.NB * fr[0])]
            gy = self.interpolation_LW[int(self.NB * fr[1])]
            gz = self.interpolation_LW[int(self.NB * fr[2])]
            
            # finite difference coefficients.
            gx_F = self.LW[int(self.NB * fr[0])]
            gy_F = self.LW[int(self.NB * fr[1])]
            gz_F = self.LW[int(self.NB * fr[2])]
            
            gk_x = np.einsum('i,j,k', gz, gy, gx_F)           
            gk_y = np.einsum('i,j,k', gz, gy_F, gx)           
            gk_z = np.einsum('i,j,k', gz_F, gy, gx)

            d_x = u[ix[2] - 1 : ix[2] + 3, ix[1] - 1 : ix[1] + 3, ix[0] - 3 : ix[0] + 5, :]           
            d_y = u[ix[2] - 1 : ix[2] + 3, ix[1] - 3 : ix[1] + 5, ix[0] - 1 : ix[0] + 3, :]           
            d_z = u[ix[2] - 3 : ix[2] + 5, ix[1] - 1 : ix[1] + 3, ix[0] - 1 : ix[0] + 3, :]
            
            # dudx,dvdx,dwdx.
            dvdx = np.einsum('ijk,ijkl->l', gk_x, d_x)
            # dudy,dvdy,dwdy.
            dvdy = np.einsum('ijk,ijkl->l', gk_y, d_y)
            # dudz,dvdz,dwdz.
            dvdz = np.einsum('ijk,ijkl->l', gk_z, d_z)
            
            if self.sint == 'fd4lag4_g':
                return np.stack((dvdx, dvdy, dvdz), axis = 1)
            elif self.sint == 'fd4lag4_l':
                dudxyz = dvdx[0] + dvdy[0] + dvdz[0]
                dvdxyz = dvdx[1] + dvdy[1] + dvdz[1]
                dwdxyz = dvdx[2] + dvdy[2] + dvdz[2]

                return np.array([dudxyz, dvdxyz, dwdxyz])
        
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
                pickle_file = self.pickle_dir_backup.joinpath(pickle_filename)

                # try reading the pickled file.
                with open(pickle_file, 'rb') as pickled_filepath:
                    return dill.load(pickled_filepath)
            except:
                raise Exception('metadata files are not accessible.')
    
    def get_file_for_mortoncode(self, cornercode):
        """
        querying the cached SQL metadata for the file for the specified morton code.
        """
        # query the cached SQL metadata for the user-specified grid point.
        t = self.cache[(self.cache['minLim'] <= cornercode) & (self.cache['maxLim'] >= cornercode)]
        t = t.iloc[0]
        f = self.filepaths[f'{t.ProductionDatabaseName}']
        return f, t.minLim, t.maxLim
    
    def open_zarr_file(self, db_file):
        """
        open the zarr file for reading. first, try reading from the production copy. second, try reading from the backup copy.
        """
        try:
            return zarr.open(f'{db_file}_{self.timepoint}.zarr{os.sep}{self.var_name}', dtype = self.dt, mode = 'r')
        except:
            try:
                db_file_backup = self.filepaths_backup[os.path.basename(db_file)]
                return zarr.open(f'{db_file_backup}_{self.timepoint}.zarr{os.sep}{self.var_name}', dtype = self.dt, mode = 'r')
            except:
                raise Exception(f'{db_file}_{self.timepoint}.zarr{os.sep}{self.var_name} and the corresponding backup file are not accessible.')
    
    """
    getCutout functions.
    """
    def identify_single_database_file_sub_boxes(self, box):
        # initially assumes the user specified box contains points in different files. the box will be split up until all the points
        # in each sub-box are from a single database file.
        box_min_xyz = [axis_range[0] for axis_range in box]
        box_max_xyz = [axis_range[1] for axis_range in box]
        
        # map of the parts of the user-specified box that are found in each database file.
        single_file_boxes = defaultdict(lambda: defaultdict(list))

        # z-value of the origin point (bottom left corner) of the box.
        current_z = box_min_xyz[2]

        # kick out of the while loops when there are no more database files along an axis.
        while current_z <= box_max_xyz[2]:
            # y-value of the origin point of the box.
            current_y = box_min_xyz[1]

            while current_y <= box_max_xyz[1]:
                # x-value of the origin point of the box.
                current_x = box_min_xyz[0]

                while current_x <= box_max_xyz[0]:
                    # database file name and corresponding minimum morton limit for the origin point of the box.  
                    min_corner_xyz = [current_x, current_y, current_z]
                    min_corner_info = self.get_file_for_mortoncode(self.mortoncurve.pack(min_corner_xyz[0] % self.N, min_corner_xyz[1] % self.N, min_corner_xyz[2] % self.N))
                    min_corner_db_file = min_corner_info[0]
                    database_file_disk = min_corner_db_file.split(os.sep)[self.database_file_disk_index]
                    box_minLim = min_corner_info[1]
                    max_corner_xyz = self.mortoncurve.unpack(min_corner_info[2])
                    
                    # calculate the number of periodic cubes that the user-specified box expands into beyond the boundary along each axis.
                    cube_ms = [math.floor(float(min_corner_xyz[q]) / float(self.N)) * self.N for q in range(3)]
                    
                    # specify the box that is fully inside a database file.
                    box = [[min_corner_xyz[i], min(max_corner_xyz[i] + cube_ms[i], box_max_xyz[i])] for i in range(3)]
                    
                    # add the box axes ranges to the map.
                    single_file_boxes[database_file_disk][min_corner_db_file].append(box)

                    # move to the next database file origin point.
                    current_x = max_corner_xyz[0] + cube_ms[0] + 1

                current_y = max_corner_xyz[1] + cube_ms[1] + 1

            current_z = max_corner_xyz[2] + cube_ms[2] + 1
    
        return single_file_boxes
        
    def read_database_files_sequentially(self, user_single_db_boxes):
        result_output_data = []
        # iterate over the hard disk drives that the database files are stored on.
        for database_file_disk in user_single_db_boxes:
            # read in the voxel data from all of the database files on this disk.
            result_output_data += self.get_iso_points(user_single_db_boxes[database_file_disk],
                                                      verbose = False)
        
        return result_output_data
    
    def read_database_files_in_parallel_with_dask(self, user_single_db_boxes):
        # start the dask client for parallel processing.
        # -----
        # flag specifying if the cluster is a premade distributed cluster. assumed to be True to start.
        distributed_cluster = True
        try:
            # using a premade distributed cluster.
            import SciServer.Dask
            
            # attached cluster (when a cluster is created together with a new container).
            client = SciServer.Dask.getClient()
            # deletes data on the network and restarts the workers.
            client.restart()
            
            # get the current working directory for saving the zip file of turbulence processing functions to.
            data_dir = os.getcwd() + os.sep
            
            # upload the turbulence processing functions in the giverny folder to the workers.
            shutil.make_archive(data_dir + 'giverny', 'zip', root_dir = data_dir, base_dir = 'giverny' + os.sep)
            client.upload_file(data_dir + 'giverny.zip')
        except:
            print(f'Starting a local dask cluster...')
            sys.stdout.flush()
            
            # update the distributed_cluster flag to False.
            distributed_cluster = False
            
            # using a local cluster if there is no premade distributed cluster.
            cluster = LocalCluster(n_workers = self.dask_maximum_processes, processes = True)
            client = Client(cluster)
        
        # available workers.
        workers = list(client.scheduler_info()['workers'])
        num_workers = len(workers)
        
        print(f'Database files are being read in parallel...')
        sys.stdout.flush()
        
        result_output_data = []
        # iterate over the hard disk drives that the database files are stored on.
        for database_file_disk in user_single_db_boxes:
            # read in the voxel data from all of the database files on this disk.
            result_output_data.append(client.submit(self.get_iso_points, user_single_db_boxes[database_file_disk],
                                                    verbose = False, workers = workers, pure = False))
        
        # gather all of the results once they are finished being run in parallel by dask.
        result_output_data = client.gather(result_output_data)        
        # flattens result_output_data to match the formatting as when the data is processed sequentially.
        result_output_data = [element for result in result_output_data for element in result]
        
        # close the dask client.
        client.close()
        
        if distributed_cluster:
            # delete the giverny.zip file if using a premade distributed cluster.
            if os.path.exists(data_dir + 'giverny.zip'):
                os.remove(data_dir + 'giverny.zip')
        else:
            # close the cluster if a local cluster was created.
            cluster.close()
        
        return result_output_data
    
    def get_iso_points(self, user_single_db_boxes_disk_data,
                       verbose = False):
        """
        retrieve the values for the specified var(iable) in the user-specified box and at the specified timepoint.
        """
        # the collection of local output data that will be returned to fill the complete output_data array.
        local_output_data = []
        # iterate over the database files to read the data from.
        for db_file in user_single_db_boxes_disk_data:
            zm = self.open_zarr_file(db_file)
            
            # iterate over the user box ranges corresponding to the morton voxels that will be read from this database file.
            for user_box_ranges in user_single_db_boxes_disk_data[db_file]:
                # retrieve the minimum and maximum (x, y, z) coordinates of the database file box that is going to be read in.
                min_xyz = [axis_range[0] for axis_range in user_box_ranges]
                max_xyz = [axis_range[1] for axis_range in user_box_ranges]
                # adjust the user box ranges to file size indices.
                user_box_ranges = np.array(user_box_ranges) % self.file_size
                
                # append the cutout into local_output_data.
                local_output_data.append((zm[user_box_ranges[2][0] : user_box_ranges[2][1] + 1,
                                             user_box_ranges[1][0] : user_box_ranges[1][1] + 1,
                                             user_box_ranges[0][0] : user_box_ranges[0][1] + 1],
                                             min_xyz, max_xyz))
        
        return local_output_data
    
    def write_output_matrix_to_hdf5(self, output_data, output_filename):
        # write output_data to a hdf5 file.
        output_data.to_netcdf(self.output_path.joinpath(output_filename + '.h5'),
                              format = "NETCDF4", mode = "w")
        
    def write_xmf(self, shape, h5_var_name, h5_attribute_type, h5_dataset_name, output_filename):
        # write the xmf file that corresponds to the hdf5 file.
        output_str = f"""<?xml version=\"1.0\" ?>
        <!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>
        <Xdmf Version=\"2.0\">
          <Domain>
              <Grid Name=\"Structured Grid\" GridType=\"Uniform\">
                <Time Value=\"{self.timepoint + 1}\"/>
                <Topology TopologyType=\"3DRectMesh\" NumberOfElements=\"{shape[2]} {shape[1]} {shape[0]}\"/>
                <Geometry GeometryType=\"VXVYVZ\">
                  <DataItem Name=\"Xcoor\" Dimensions=\"{shape[0]}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">
                    {output_filename}.h5:/xcoor
                  </DataItem>
                  <DataItem Name=\"Ycoor\" Dimensions=\"{shape[1]}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">
                    {output_filename}.h5:/ycoor
                  </DataItem>
                  <DataItem Name=\"Zcoor\" Dimensions=\"{shape[2]}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">
                    {output_filename}.h5:/zcoor
                  </DataItem>
                </Geometry>
                <Attribute Name=\"{h5_var_name}\" AttributeType=\"{h5_attribute_type}\" Center=\"Node\">
                  <DataItem Dimensions=\"{shape[2]} {shape[1]} {shape[0]} {self.num_values_per_datapoint}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">
                    {output_filename}.h5:/{h5_dataset_name}
                  </DataItem>
                </Attribute>
              </Grid>
          </Domain>
        </Xdmf>"""
        
        with open(self.output_path.joinpath(output_filename + '.xmf'), 'w') as tf:
            tf.write(output_str)
            
    """
    getVariable functions.
    """
    def identify_database_file_points(self, points):
        # vectorize the mortoncurve.pack function.
        v_morton_pack = np.vectorize(self.mortoncurve.pack, otypes = [int])
        
        # chunk cube size.
        chunk_cube_size = 64**3
        # empty array for subdividing chunk groups.
        empty_array = np.array([0, 0, 0])
        # chunk size array for subdividing chunk groups.
        chunk_size_array = np.array([self.chunk_size, self.chunk_size, self.chunk_size]) - 1
        
        # convert the points to the center point position within their own bucket.
        center_points = (points / self.dx % 1) + self.cube_min_index
        # convert the points to gridded datapoints.
        datapoints = np.floor(points / self.dx).astype(int) % self.N
        # calculate the minimum and maximum chunk (x, y, z) corner point for each point in datapoints.
        chunk_min_xyzs = ((datapoints - self.cube_min_index) - ((datapoints - self.cube_min_index) % self.chunk_size)) % self.N
        chunk_max_xyzs = ((datapoints + self.cube_max_index) + (self.chunk_size - ((datapoints + self.cube_max_index) % self.chunk_size) - 1)) % self.N
        # chunk volumes.
        chunk_volumes = np.prod(chunk_max_xyzs - chunk_min_xyzs + 1, axis = 1)
        # create the chunk keys for each chunk group.
        chunk_keys = [chunk_origin_group.tobytes() for chunk_origin_group in np.stack([chunk_min_xyzs, chunk_max_xyzs], axis = 1)]
        # convert chunk_min_xyzs and chunk_max_xyzs to indices in a single database file.
        chunk_min_mods = chunk_min_xyzs % self.file_size
        chunk_max_mods = chunk_max_xyzs % self.file_size
        # calculate the minimum and maximum chunk morton codes for each point in chunk_min_xyzs and chunk_max_xyzs.
        chunk_min_mortons = v_morton_pack(chunk_min_xyzs[:, 0], chunk_min_xyzs[:, 1], chunk_min_xyzs[:, 2])
        chunk_max_mortons = v_morton_pack(chunk_max_xyzs[:, 0], chunk_max_xyzs[:, 1], chunk_max_xyzs[:, 2])
        # calculate the db file cornercodes for each morton code.
        db_min_cornercodes = (chunk_min_mortons >> 27) << 27
        db_max_cornercodes = (chunk_max_mortons >> 27) << 27
        # identify the database files that will need to be read for each chunk.
        db_min_files = [self.cornercode_file_map[morton_code] for morton_code in db_min_cornercodes]
        db_max_files = [self.cornercode_file_map[morton_code] for morton_code in db_max_cornercodes]
        
        # save the original indices for points, which corresponds to the orderering of the user-specified
        # points. these indices will be used for sorting output_data back to the user-specified points ordering.
        original_points_indices = [q for q in range(len(points))]
        # zip the data. sort by volume first so that all fully overlapped chunk groups can be easily found.
        zipped_data = sorted(zip(chunk_volumes, chunk_min_mortons, chunk_keys, db_min_files, db_max_files, points, datapoints, center_points,
                                 chunk_min_xyzs, chunk_max_xyzs, chunk_min_mods, chunk_max_mods, original_points_indices), key = lambda x: (-1 * x[0], x[1], x[2]))
        
        # map the native bucket points to their respective db files and chunks.
        db_native_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # store an array of visitor bucket points.
        db_visitor_map = []
        # chunk key map used for storing all subdivided chunk groups to find fully overlapped chunk groups.
        chunk_map = {}
        
        for chunk_volume, chunk_min_morton, chunk_key, db_min_file, db_max_file, point, datapoint, center_point, \
            chunk_min_xyz, chunk_max_xyz, chunk_min_mod, chunk_max_mod, original_point_index in zipped_data:
            # update the database file info if the morton code is outside of the previous database fil maximum morton limit.
            db_disk = db_min_file.split(os.sep)[self.database_file_disk_index]
            
            if db_min_file == db_max_file:
                # update the chunk key if the chunk group is fully contained in another larger chunk group.
                updated_chunk_key = chunk_key
                if chunk_key in chunk_map:
                    updated_chunk_key = chunk_map[chunk_key]
                elif chunk_volume != chunk_cube_size:
                    chunk_map = self.subdivide_chunk_group(chunk_map, chunk_key, chunk_min_xyz, chunk_max_xyz, chunk_size_array, empty_array)
                
                # assign to native map.
                if updated_chunk_key not in db_native_map[db_disk][db_min_file]:
                    db_native_map[db_disk][db_min_file][updated_chunk_key].append((chunk_min_xyz, chunk_max_xyz, chunk_min_mod, chunk_max_mod))
    
                db_native_map[db_disk][db_min_file][updated_chunk_key].append((point, datapoint, center_point, original_point_index))
            else:
                # assign to the visitor map.
                db_visitor_map.append((point, datapoint, center_point, original_point_index))
        
        return db_native_map, np.array(db_visitor_map, dtype = 'object')
    
    def subdivide_chunk_group(self, chunk_map, chunk_key, chunk_min_xyz, chunk_max_xyz, chunk_size_array, empty_array): 
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
        new_min[chunk_diffs[0]] += self.chunk_size
        new_max = chunk_min_xyz + chunk_size_array
        new_max[chunk_diffs[0]] += self.chunk_size
        chunk_mins.append(new_min)
        chunk_maxs.append(new_max)
        
        # add additional sub-chunks chunk group contains 4 or 8 chunks.
        if num_long_axes == 2 or num_long_axes == 3:
            # 1-cubes, additional.
            # long axis 2, first 1-cube.
            new_min = chunk_min_xyz + empty_array
            new_min[chunk_diffs[1]] += self.chunk_size
            new_max = chunk_min_xyz + chunk_size_array
            new_max[chunk_diffs[1]] += self.chunk_size
            chunk_mins.append(new_min)
            chunk_maxs.append(new_max)

            # long axis 2, second 1-cube.
            new_min = chunk_min_xyz + empty_array
            new_min[chunk_diffs[0]] += self.chunk_size
            new_min[chunk_diffs[1]] += self.chunk_size
            new_max = chunk_min_xyz + chunk_size_array
            new_max[chunk_diffs[0]] += self.chunk_size
            new_max[chunk_diffs[1]] += self.chunk_size
            chunk_mins.append(new_min)
            chunk_maxs.append(new_max)

            # 2-cubes.
            # long axis 1, first 2-cube.
            new_max = chunk_min_xyz + chunk_size_array
            new_max[chunk_diffs[0]] += self.chunk_size
            chunk_mins.append(chunk_min_xyz)
            chunk_maxs.append(new_max)

            # long axis 1, second 2-cube.
            new_min = chunk_min_xyz + empty_array
            new_min[chunk_diffs[1]] += self.chunk_size
            new_max = chunk_min_xyz + chunk_size_array
            new_max[chunk_diffs[0]] += self.chunk_size
            new_max[chunk_diffs[1]] += self.chunk_size
            chunk_mins.append(new_min)
            chunk_maxs.append(new_max)
            
            # long axis 2, first 2-cube.
            new_max = chunk_min_xyz + chunk_size_array
            new_max[chunk_diffs[1]] += self.chunk_size
            chunk_mins.append(chunk_min_xyz)
            chunk_maxs.append(new_max)

            # long axis 2, second 2-cube.
            new_min = chunk_min_xyz + empty_array
            new_min[chunk_diffs[0]] += self.chunk_size
            new_max = chunk_min_xyz + chunk_size_array
            new_max[chunk_diffs[0]] += self.chunk_size
            new_max[chunk_diffs[1]] += self.chunk_size
            chunk_mins.append(new_min)
            chunk_maxs.append(new_max)
        
            if num_long_axes == 3:
                # 1-cubes, additional.
                # long axis 3, first 1-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[2]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 3, second 1-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[0]] += self.chunk_size
                new_min[chunk_diffs[2]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 3, third 1-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[1]] += self.chunk_size
                new_min[chunk_diffs[2]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 3, fourth 1-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[0]] += self.chunk_size
                new_min[chunk_diffs[1]] += self.chunk_size
                new_min[chunk_diffs[2]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # 2-cubes, additional.
                # long axis 1, third 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[2]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 1, fourth 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[1]] += self.chunk_size
                new_min[chunk_diffs[2]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)
                
                # long axis 2, third 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[2]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 2, fourth 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[0]] += self.chunk_size
                new_min[chunk_diffs[2]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)
                
                # long axis 3, first 2-cube.
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(chunk_min_xyz)
                chunk_maxs.append(new_max)

                # long axis 3, second 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[0]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 3, third 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[1]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 3, fourth 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[0]] += self.chunk_size
                new_min[chunk_diffs[1]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # 4-cubes.
                # long axes 1 and 2, first 4-cube.
                new_min = chunk_min_xyz + empty_array
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[1]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axes 1 and 2, second 4-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[2]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axes 1 and 3, first 4-cube.
                new_min = chunk_min_xyz + empty_array
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axes 1 and 3, second 4-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[1]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axes 2 and 3, first 4-cube.
                new_min = chunk_min_xyz + empty_array
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axes 2 and 3, second 4-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[0]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
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
    
    def get_chunk_origin_groups(self, chunk_min_x, chunk_min_y, chunk_min_z, chunk_max_x, chunk_max_y, chunk_max_z):
        # get arrays of the chunk origin points for each bucket.
        return np.array([[x, y, z]
                         for z in range(chunk_min_z, (chunk_max_z if chunk_min_z <= chunk_max_z else self.N + chunk_max_z) + 1, self.chunk_size)
                         for y in range(chunk_min_y, (chunk_max_y if chunk_min_y <= chunk_max_y else self.N + chunk_max_y) + 1, self.chunk_size)
                         for x in range(chunk_min_x, (chunk_max_x if chunk_min_x <= chunk_max_x else self.N + chunk_max_x) + 1, self.chunk_size)])
    
    def read_natives_sequentially_variable(self, db_native_map, native_output_data):
        # native data.
        # iterate over the data volumes that the database files are stored on.
        for database_file_disk in db_native_map:
            # read in the voxel data from all of the database files on this disk.
            native_output_data += self.get_iso_points_variable(db_native_map[database_file_disk], verbose = False)
            
    def read_visitors_in_parallel_variable(self, db_visitor_map, visitor_output_data):
        # visitor data.
        if len(db_visitor_map) != 0:
            distributed_cluster = True
            try:
                # using a premade distributed cluster.
                import SciServer.Dask

                # attached cluster (when a cluster is created together with a new container).
                client = SciServer.Dask.getClient()
                # deletes data on the network and restarts the workers.
                client.restart()

                # get the current working directory for saving the zip file of turbulence processing functions to.
                data_dir = os.getcwd() + os.sep

                # upload the turbulence processing functions in the giverny folder to the workers.
                shutil.make_archive(data_dir + 'giverny', 'zip', root_dir = data_dir, base_dir = 'giverny' + os.sep)
                client.upload_file(data_dir + 'giverny.zip')
            except:
                print(f'Starting a local dask cluster...')
                sys.stdout.flush()
            
                # update the distributed_cluster flag to False.
                distributed_cluster = False

                # using a local cluster if there is no premade distributed cluster.
                cluster = LocalCluster(n_workers = self.dask_maximum_processes, processes = True)
                client = Client(cluster)
    
            # available workers.
            workers = list(client.scheduler_info()['workers'])
            num_workers = len(workers)
            
            print(f'Database files are being read in parallel...')
            sys.stdout.flush()
            
            # calculate how many chunks to use for splitting up the visitor map data.
            num_visitor_points = len(db_visitor_map)
            num_chunks = num_workers
            if num_visitor_points < num_workers:
                num_chunks = num_visitor_points

            # chunk db_visitor_map.
            db_visitor_map_split = np.array(np.array_split(db_visitor_map, num_chunks), dtype = object)

            temp_visitor_output_data = []
            # scatter the chunks to their own worker and submit each chunk for parallel processing.
            for db_visitor_map_chunk, worker in zip(db_visitor_map_split, workers):
                # submit the chunk for parallel processing.
                temp_visitor_output_data.append(client.submit(self.get_iso_points_variable_visitor, db_visitor_map_chunk, verbose = False, workers = worker, pure = False))
                
            # gather all of the results once they are finished being run in parallel by dask.
            temp_visitor_output_data = client.gather(temp_visitor_output_data)
            # flattens result_output_data to match the formatting as when the data is processed sequentially.
            temp_visitor_output_data = [element for result in temp_visitor_output_data for element in result]

            # update visitor_output_data.
            visitor_output_data += temp_visitor_output_data
            
            # close the dask client.
            client.close()

            if distributed_cluster:
                # delete the giverny.zip file if using a premade distributed cluster.
                if os.path.exists(data_dir + 'giverny.zip'):
                    os.remove(data_dir + 'giverny.zip')
            else:
                # close the cluster if a local cluster was created.
                cluster.close() 
            
    def read_database_files_sequentially_variable(self, db_native_map, db_visitor_map):
        if len(db_visitor_map) == 0:
            print('Database files are being read sequentially...')
            sys.stdout.flush()
        
        # create empty lists for filling the output data.
        native_output_data = []
        visitor_output_data = []
        
        # create threads for parallel processing of the native and visitor data.
        native_thread = Thread(target = self.read_natives_sequentially_variable, args = (db_native_map, native_output_data))
        visitor_thread = Thread(target = self.read_visitors_in_parallel_variable, args = (db_visitor_map, visitor_output_data))
            
        # start the threads.
        native_thread.start()
        visitor_thread.start()

        # wait for the threads to complete.
        native_thread.join()
        visitor_thread.join()
            
        result_output_data = native_output_data + visitor_output_data
            
        return result_output_data
    
    def read_database_files_in_parallel_with_dask_variable(self, db_native_map, db_visitor_map):
        # start the dask client for parallel processing.
        # -----
        # flag specifying if the cluster is a premade distributed cluster. assumed to be True to start.
        distributed_cluster = True
        try:
            # using a premade distributed cluster.
            import SciServer.Dask
            
            # attached cluster (when a cluster is created together with a new container).
            client = SciServer.Dask.getClient()
            # deletes data on the network and restarts the workers.
            client.restart()
            
            # get the current working directory for saving the zip file of turbulence processing functions to.
            data_dir = os.getcwd() + os.sep
            
            # upload the turbulence processing functions in the giverny folder to the workers.
            shutil.make_archive(data_dir + 'giverny', 'zip', root_dir = data_dir, base_dir = 'giverny' + os.sep)
            client.upload_file(data_dir + 'giverny.zip')
        except:
            print(f'Starting a local dask cluster...')
            sys.stdout.flush()
            
            # update the distributed_cluster flag to False.
            distributed_cluster = False
            
            # using a local cluster if there is no premade distributed cluster.
            cluster = LocalCluster(n_workers = self.dask_maximum_processes, processes = True)
            client = Client(cluster)
        
        # available workers.
        workers = list(client.scheduler_info()['workers'])
        num_workers = len(workers)
        
        print(f'Database files are being read in parallel...')
        sys.stdout.flush()
        
        result_output_data = []
        # native buckets.
        # -----
        if len(db_native_map) != 0:
            # iterate over the db volumes.
            for disk_index, database_file_disk in enumerate(db_native_map):
                worker = workers[disk_index % num_workers] 
                
                # submit the data for parallel processing.
                result_output_data.append(client.submit(self.get_iso_points_variable, db_native_map[database_file_disk], verbose = False,
                                                            workers = worker, pure = False))
        
        # visitor buckets.
        # -----
        if len(db_visitor_map) != 0:
            # calculate how many chunks to use for splitting up the visitor map data.
            num_visitor_points = len(db_visitor_map)
            num_chunks = num_workers
            if num_visitor_points < num_workers:
                num_chunks = num_visitor_points

            # chunk db_visitor_map.
            db_visitor_map_split = np.array(np.array_split(db_visitor_map, num_chunks), dtype = object)

            # scatter the chunks to their own worker and submit the chunk for parallel processing.
            for db_visitor_map_chunk, worker in zip(db_visitor_map_split, workers):
                # submit the data for parallel processing.
                result_output_data.append(client.submit(self.get_iso_points_variable_visitor, db_visitor_map_chunk, verbose = False, workers = worker, pure = False))
        
        # gather all of the results once they are finished being run in parallel by dask.
        result_output_data = client.gather(result_output_data)        
        # flattens result_output_data to match the formatting as when the data is processed sequentially.
        result_output_data = [element for result in result_output_data for element in result]
             
        # close the dask client.
        client.close()
        
        if distributed_cluster:
            # delete the giverny.zip file if using a premade distributed cluster.
            if os.path.exists(data_dir + 'giverny.zip'):
                os.remove(data_dir + 'giverny.zip')
        else:
            # close the cluster if a local cluster was created.
            cluster.close()
                
        return result_output_data
    
    def get_iso_points_variable(self, db_native_map_data, verbose = False):
        """
        reads and interpolates the user-requested native points in a single database volume.
        """
        # the collection of local output data that will be returned to fill the complete output_data array.
        local_output_data = []
        # iterate over the database files and morton sub-boxes to read the data from.
        for db_file in db_native_map_data:
            zs = self.open_zarr_file(db_file)
            
            db_file_data = db_native_map_data[db_file]
            for chunk_key in db_file_data:
                chunk_data = db_file_data[chunk_key]
                chunk_min_xyz = chunk_data[0][0]
                chunk_max_xyz = chunk_data[0][1]

                # read in the necessary chunks.
                zm = zs[chunk_data[0][2][2] : chunk_data[0][3][2] + 1,
                        chunk_data[0][2][1] : chunk_data[0][3][1] + 1,
                        chunk_data[0][2][0] : chunk_data[0][3][0] + 1]
                
                # iterate over the points to interpolate.
                for point, datapoint, center_point, original_point_index in chunk_data[1:]:
                    bucket_min_xyz = datapoint - chunk_min_xyz - self.cube_min_index
                    bucket_max_xyz = datapoint - chunk_min_xyz + self.cube_max_index + 1

                    bucket = zm[bucket_min_xyz[2] : bucket_max_xyz[2],
                                bucket_min_xyz[1] : bucket_max_xyz[1],
                                bucket_min_xyz[0] : bucket_max_xyz[0]]
            
                    # interpolate the points and use a lookup table for faster interpolations.
                    local_output_data.append((original_point_index, (point, self.interpolate(center_point, bucket))))
        
        return local_output_data
    
    def get_iso_points_variable_visitor(self, visitor_data, verbose = False):
            """
            reads and interpolates the user-requested visitor points.
            """
            # vectorize the mortoncurve.pack function.
            v_morton_pack = np.vectorize(self.mortoncurve.pack, otypes = [int])

            # the collection of local output data that will be returned to fill the complete output_data array.
            local_output_data = []

            # empty chunk group array (up to eight 64-cube chunks).
            zm = np.zeros((128, 128, 128, self.num_values_per_datapoint))

            datapoints = np.array([datapoint for datapoint in visitor_data[:, 1]])
            # calculate the minimum and maximum chunk group corner point (x, y, z) for each datapoint.
            chunk_min_xyzs = ((datapoints - self.cube_min_index) - ((datapoints - self.cube_min_index) % self.chunk_size)) % self.N
            chunk_max_xyzs = ((datapoints + self.cube_max_index) + (self.chunk_size - ((datapoints + self.cube_max_index) % self.chunk_size) - 1)) % self.N
            # calculate the morton codes for the minimum (x, y, z) point of each chunk group.
            chunk_min_mortons = v_morton_pack(chunk_min_xyzs[:, 0], chunk_min_xyzs[:, 1], chunk_min_xyzs[:, 2])
            # create the chunk keys for each chunk group.
            chunk_keys = [chunk_origin_group.tobytes() for chunk_origin_group in np.stack([chunk_min_xyzs, chunk_max_xyzs], axis = 1)]
            # calculate the minimum and maximum bucket corner point (x, y, z) for each datapoint.
            bucket_min_xyzs = (datapoints - chunk_min_xyzs - self.cube_min_index) % self.N
            bucket_max_xyzs = (datapoints - chunk_min_xyzs + self.cube_max_index + 1) % self.N
            # create the bucket keys for each interpolation bucket.
            bucket_keys = [bucket_origin_group.tobytes() for bucket_origin_group in np.stack([bucket_min_xyzs, bucket_max_xyzs], axis = 1)]

            current_chunk = ''
            current_bucket = ''
            for point_data, chunk_min_morton, chunk_min_xyz, chunk_max_xyz, chunk_key, bucket_min_xyz, bucket_max_xyz, bucket_key in \
                sorted(zip(visitor_data, chunk_min_mortons, chunk_min_xyzs, chunk_max_xyzs, chunk_keys, bucket_min_xyzs, bucket_max_xyzs, bucket_keys),
                       key = lambda x: (x[1], x[4], x[7])):
                if current_chunk != chunk_key:
                    # get the origin points for each voxel in the bucket.
                    chunk_origin_groups = self.get_chunk_origin_groups(chunk_min_xyz[0], chunk_min_xyz[1], chunk_min_xyz[2],
                                                                       chunk_max_xyz[0], chunk_max_xyz[1], chunk_max_xyz[2])
                    # adjust the chunk origin points to the chunk domain size for filling the empty chunk group array.
                    chunk_origin_points = chunk_origin_groups - chunk_origin_groups[0]

                    # get the chunk origin group inside the dataset domain.
                    chunk_origin_groups = chunk_origin_groups % self.N
                    # calculate the morton codes for the minimum point in each chunk of the chunk groups.
                    morton_mins = v_morton_pack(chunk_origin_groups[:, 0], chunk_origin_groups[:, 1], chunk_origin_groups[:, 2])
                    # get the chunk origin group inside the file domain.
                    chunk_origin_groups = chunk_origin_groups % self.file_size

                    # calculate the db file cornercodes for each morton code.
                    db_cornercodes = (morton_mins >> 27) << 27
                    # identify the database files that will need to be read for this bucket.
                    db_files = [self.cornercode_file_map[morton_code] for morton_code in db_cornercodes]

                    current_file = ''
                    # iterate of the db files.
                    for db_file, chunk_origin_point, chunk_origin_group in sorted(zip(db_files, chunk_origin_points, chunk_origin_groups), key = lambda x: x[0]):
                        if db_file != current_file:
                            # create an open file object of the database file.
                            zs = self.open_zarr_file(db_file)

                            # update current_file.
                            current_file = db_file

                        zm[chunk_origin_point[2] : chunk_origin_point[2] + self.chunk_size,
                           chunk_origin_point[1] : chunk_origin_point[1] + self.chunk_size,
                           chunk_origin_point[0] : chunk_origin_point[0] + self.chunk_size] = zs[chunk_origin_group[2] : chunk_origin_group[2] + self.chunk_size,
                                                                                                 chunk_origin_group[1] : chunk_origin_group[1] + self.chunk_size,
                                                                                                 chunk_origin_group[0] : chunk_origin_group[0] + self.chunk_size]

                    # update current_chunk.
                    current_chunk = chunk_key

                if current_bucket != bucket_key:
                    bucket = zm[bucket_min_xyz[2] : bucket_max_xyz[2],
                                bucket_min_xyz[1] : bucket_max_xyz[1],
                                bucket_min_xyz[0] : bucket_max_xyz[0]]

                    # update current_bucket.
                    current_bucket = bucket_key

                # interpolate the point and use a lookup table for faster interpolation.
                local_output_data.append((point_data[3], (point_data[0], self.interpolate(point_data[2], bucket))))

            return local_output_data
    