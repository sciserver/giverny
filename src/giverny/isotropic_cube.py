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
        
        # get the SciServer user name.
        user = Authentication.getKeystoneUserWithToken(Authentication.getToken()).userName
        
        # set the directory for saving any output files.
        self.output_path = output_path.strip()
        if self.output_path == '':
            self.output_path = pathlib.Path(f'/home/idies/workspace/Temporary/{user}/scratch/turbulence_output')
        else:
            self.output_path = pathlib.Path(self.output_path).joinpath('turbulence_output')
        
        # create the output directory if it does not already exist.
        create_output_folder(self.output_path)
        
        # set the directory for saving and reading the pickled files.
        self.pickle_dir = pathlib.Path(f'/home/idies/workspace/turb/data01_01/zarr/turbulence_pickled')
        # create the pickled directory if it does not already exist.
        create_output_folder(self.pickle_dir)
        
        # get a cache of the metadata for the database files.
        self.init_cache()
    
    """
    initialization functions.
    """
    def init_cache(self):
        # pickled file for saving the globbed filepaths.
        pickle_file = self.pickle_dir.joinpath(self.dataset_title + '_metadata.pickle')
        
        try:
            # try reading the pickled file.
            with open(pickle_file, 'rb') as pickled_filepath:
                self.cache = dill.load(pickled_filepath)
        except FileNotFoundError:
            # read SQL metadata for all of the turbulence data files into the cache.
            sql = f"""
            select dbm.ProductionDatabaseName
            , dbm.minLim
            , dbm.maxLim
            from databasemap dbm
            where dbm.datasetname = '{self.dataset_title}'
            order by minlim
            """
            df = cj.executeQuery(sql, "turbinfo")

            x, y, z = self.mortoncurve.unpack(df['minLim'].values)
            df['x_min'] = x
            df['y_min'] = y
            df['z_min'] = z

            x, y, z = self.mortoncurve.unpack(df['maxLim'].values)
            df['x_max'] = x
            df['y_max'] = y 
            df['z_max'] = z

            self.cache = df
            
            # save self.cache to a pickled file.
            with open(pickle_file, 'wb') as pickled_filepath:
                dill.dump(self.cache, pickled_filepath)
    
    def get_turb_folders(self):
        # specifies the folders on fileDB that should be searched for the primary copies of the zarr files.
        folder_base = '/home/idies/workspace/turb/'
        folder_paths = ['data01_01/zarr/', 'data01_02/zarr/', 'data01_03/zarr/',
                        'data02_01/zarr/', 'data02_02/zarr/', 'data02_03/zarr/',
                        'data03_01/zarr/', 'data03_02/zarr/', 'data03_03/zarr/',
                        'data04_01/zarr/', 'data04_02/zarr/', 'data04_03/zarr/',
                        'data05_01/zarr/', 'data05_02/zarr/', 'data05_03/zarr/',
                        'data06_01/zarr/', 'data06_02/zarr/', 'data06_03/zarr/',
                        'data07_01/zarr/', 'data07_02/zarr/', 'data07_03/zarr/',
                        'data08_01/zarr/', 'data08_02/zarr/', 'data08_03/zarr/',
                        'data09_01/zarr/', 'data09_02/zarr/', 'data09_03/zarr/',
                        'data10_01/zarr/', 'data10_02/zarr/', 'data10_03/zarr/',
                        'data11_01/zarr/', 'data11_02/zarr/', 'data11_03/zarr/',
                        'data12_01/zarr/', 'data12_02/zarr/', 'data12_03/zarr/']
        
        return [folder_base + folder_path for folder_path in folder_paths]
        
    def init_filepaths(self):
        # pickled file for saving the globbed filepaths.
        pickle_file = self.pickle_dir.joinpath(self.dataset_title + '_database_filepaths.pickle')
        
        try:
            if self.rewrite_metadata:
                raise FileNotFoundError
            
            # try reading the pickled file.
            with open(pickle_file, 'rb') as pickled_filepath:
                self.filepaths = dill.load(pickled_filepath)
        except FileNotFoundError:
            # map the filepaths to the part of each filename that matches "ProductionDatabaseName" in the SQL metadata for this dataset.
            self.filepaths = {}
            
            # get the common filename prefix for all files in this dataset, e.g. "iso8192" for the isotropic8192 dataset.
            dataset_filename_prefix = get_filename_prefix(self.dataset_title)
            
            # recursively search all sub-directories in the turbulence fileDB system for the dataset zarr files.
            turb_folders = self.get_turb_folders()
            filepaths = []
            for turb_folder in turb_folders:
                data_filepaths = glob.glob(turb_folder + dataset_filename_prefix + '*_prod/*.zarr')

                filepaths += data_filepaths

            # map the filepaths to the filenames so that they can be easily retrieved.
            for filepath in filepaths:
                # part of the filenames that exactly matches the "ProductionDatabaseName" column stored in the SQL metadata.
                filepath_split = filepath.split(os.sep)
                folderpath = os.sep.join(filepath_split[:-1]) + os.sep
                filename = filepath.split(os.sep)[-1].split('_')[0].strip()
                # only add the filepath to the dictionary once since there could be backup copies of the files.
                if filename not in self.filepaths:
                    self.filepaths[filename] = folderpath + filename
            
            # save self.filepaths to a pickled file.
            with open(pickle_file, 'wb') as pickled_filepath:
                dill.dump(self.filepaths, pickled_filepath)
                
    def init_cornercode_file_map(self):
        # pickled file for saving the db file cornercodes to filenames map.
        pickle_file = self.pickle_dir.joinpath(self.dataset_title + f'_cornercode_file_map.pickle')
        
        try:
            if self.rewrite_metadata:
                raise FileNotFoundError
            
            # try reading the pickled file.
            with open(pickle_file, 'rb') as pickled_file_map:
                self.cornercode_file_map = dill.load(pickled_file_map)
        except FileNotFoundError:
            # create a map of the db file cornercodes to filenames for the whole dataset.
            self.cornercode_file_map = {}
            
            cornercode = 0
            while cornercode < self.N ** 3:
                # get the file info for the db file cornercode.
                f, db_minLim, db_maxLim = self.get_file_for_mortoncode(cornercode)
                
                self.cornercode_file_map[db_minLim] = f
                
                cornercode = db_maxLim + 1
                
            # save self.cornercode_file_map to a pickled file.
            with open(pickle_file, 'wb') as pickled_file_map:
                dill.dump(self.cornercode_file_map, pickled_file_map)
    
    def init_interpolation_lookup_table(self):
        # interpolation method 'none' is omitted because there is no lookup table for 'none' interpolation.
        interp_methods = ['lag4', 'm1q4', 'lag6', 'lag8', 'm2q8']
        
        # lookup table resolution.
        self.NB = 10**5
        
        # create the metadata files for each interpolation method if they do not already exist.
        for interp_method in interp_methods:
            # pickled file for saving the interpolation coefficient lookup table.
            pickle_file = self.pickle_dir.joinpath(f'{interp_method}_lookup_table.pickle')

            # check if the pickled file exists.
            if not pickle_file.is_file():
                # create the interpolation coefficient lookup table.
                tmp_LW = self.getLagL(interp_method)

                # save self.LW to a pickled file.
                with open(pickle_file, 'wb') as pickled_lookup_table:
                    dill.dump(tmp_LW, pickled_lookup_table)
        
        # read in the interpolation lookup table for self.sint.
        if self.sint != 'none':
            # pickled file for saving the interpolation coefficient lookup table.
            pickle_file = self.pickle_dir.joinpath(f'{self.sint}_lookup_table.pickle')

            with open(pickle_file, 'rb') as pickled_lookup_table:
                self.LW = dill.load(pickled_lookup_table)
                
    def init_interpolation_cube_size_lookup_table(self):
        # pickled file for saving the interpolation cube sizes lookup table.
        pickle_file = self.pickle_dir.joinpath(f'interpolation_cube_size_lookup_table.pickle')
        
        try:
            # try reading the pickled file.
            with open(pickle_file, 'rb') as pickled_lookup_table:
                interp_cube_sizes = dill.load(pickled_lookup_table)
        except FileNotFoundError:
            # create the interpolation cube size lookup table. the first number is the number of points on the left of the integer 
            # interpolation point, and the second number is the number of points on the right.
            interp_cube_sizes = {}
            interp_cube_sizes['lag4'] = [1, 2]
            interp_cube_sizes['m1q4'] = [1, 2]
            interp_cube_sizes['lag6'] = [2, 3]
            interp_cube_sizes['lag8'] = [3, 4]
            interp_cube_sizes['m2q8'] = [3, 4]
            interp_cube_sizes['none'] = [0, 0]
            
            # save interp_cube_sizes to a pickled file.
            with open(pickle_file, 'wb') as pickled_lookup_table:
                dill.dump(interp_cube_sizes, pickled_lookup_table)
                
        # lookup the interpolation cube size indices.
        self.cube_min_index, self.cube_max_index = interp_cube_sizes[self.sint]
    
    def init_constants(self, var, var_original, timepoint, sint, num_values_per_datapoint, c,
                       rewrite_metadata = False):
        # create the constants.
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
        self.rewrite_metadata = rewrite_metadata
        
        # set the dataset name to be used in the hdf5 file. 1 is added to timepoint because the original timepoint was converted to a 0-based index.
        self.dataset_name = get_output_variable_name(var_original) + '_' + str(timepoint + 1).zfill(4)
        
        # get map of the filepaths for all of the dataset files.
        self.init_filepaths()
        
        # get a map of the files to cornercodes for all of the dataset files.
        self.init_cornercode_file_map()
    
    """
    interpolation functions.
    """
    def getLagL(self, sint):
        frac = np.linspace(0, 1 - 1 / self.NB, self.NB)
        LW = []
        for fp in frac:
            LW.append(self.getLagC(sint, fp))

        return LW
    
    #===============================================================================
    # Interpolating functions to compute the kernel, extract subcube and convolve
    #===============================================================================
    def getLagC(self, sint, fr):
        #------------------------------------------------------
        # get the 1D vectors for the 8 point Lagrange weights
        # inline the constants, and write explicit for loop
        # for the C compilation
        #------------------------------------------------------
        # cdef int n.
        if sint == 'm1q4':
            # define the weights for M1Q4 spline interpolation.
            g = np.zeros(4)
            g[0] = fr * (fr * (-1.0 / 2.0 * fr + 1) - 1.0 / 2.0)
            g[1] = fr**2 * ((3.0 / 2.0) * fr - 5.0 / 2.0) + 1
            g[2] = fr * (fr * (-3.0 / 2.0 * fr + 2) + 1.0 / 2.0)
            g[3] = fr**2 * ((1.0 / 2.0) * fr - 1.0 / 2.0)
        elif sint == 'm2q8':
            # define the weights for M2Q8 spline interpolation.
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
                g  = np.array([0,1.,0,0])
                # weight index.
                w_index = 1
            elif sint == 'lag6':
                wN = [1.,-5.,10.,-10.,5.,-1.]
                g  = np.array([0,0,1.,0,0,0])
                # weight index.
                w_index = 2
            elif sint == 'lag8':
                wN = [1.,-7.,21.,-35.,35.,-21.,7.,-1.]
                g  = np.array([0,0,0,1.,0,0,0,0])
                # weight index.
                w_index = 3

            #----------------------------
            # calculate weights if fr>0, and insert into gg
            #----------------------------
            if (fr>0):
                num_points = len(g)

                s = 0
                for n in range(num_points):
                    g[n] = wN[n] / (fr - n + w_index)
                    s += g[n]

                for n in range(num_points):
                    g[n] = g[n] / s

        return g
    
    def interpLagL(self, p, u):
        #--------------------------------------------------------
        # p is an np.array(3) containing the three coordinates
        #---------------------------------------------------------
        # get the coefficients
        #----------------------    
        if self.sint != 'none':
            # spatial interpolation methods.
            ix = p.astype(np.int32)
            fr = p - ix
            gx = self.LW[int(self.NB * fr[0])]
            gy = self.LW[int(self.NB * fr[1])]
            gz = self.LW[int(self.NB * fr[2])]
            #------------------------------------
            # create the 3D kernel from the
            # outer product of the 1d kernels
            #------------------------------------
            gk = np.einsum('i,j,k', gz, gy, gx)

            return np.einsum('ijk,ijkl->l', gk, u)
        else:
            # 'none' spatial interpolation.
            ix = np.floor(p + 0.5).astype(np.int32)
            
            return np.array(u[ix[2], ix[1], ix[0], :])
        
    """
    common functions.
    """
    def get_file_for_mortoncode(self, cornercode):
        """
        querying the cached SQL metadata for the file for the specified morton code.
        """
        # query the cached SQL metadata for the user-specified grid point.
        t = self.cache[(self.cache['minLim'] <= cornercode) & (self.cache['maxLim'] >= cornercode)]
        t = t.iloc[0]
        f = self.filepaths[f'{t.ProductionDatabaseName}']
        return f, t.minLim, t.maxLim
    
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
        except FileNotFoundError:
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
        # set the byte order for reading the data from the files.
        dt = np.dtype(np.float32)
        dt = dt.newbyteorder('<')
        
        # the collection of local output data that will be returned to fill the complete output_data array.
        local_output_data = []
        # iterate over the database files to read the data from.
        for db_file in user_single_db_boxes_disk_data:
            zm = zarr.open(db_file + '_' + str(self.timepoint) + '.zarr' + os.sep + self.var_name, dtype = dt, mode = 'r')
            
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
        
        # newline character.
        nl = '\r\n'
        
        with open(self.output_path.joinpath(output_filename + '.xmf'), 'w') as tf:
            print(f"<?xml version=\"1.0\" ?>{nl}", file = tf)
            print(f"<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>{nl}", file = tf)
            print(f"<Xdmf Version=\"2.0\">{nl}", file = tf)
            print(f"  <Domain>{nl}", file = tf)
            print(f"      <Grid Name=\"Structured Grid\" GridType=\"Uniform\">{nl}", file = tf)
            print(f"        <Time Value=\"{self.timepoint + 1}\" />{nl}", file = tf)
            print(f"        <Topology TopologyType=\"3DRectMesh\" NumberOfElements=\"{shape[2]} {shape[1]} {shape[0]}\"/>{nl}", file = tf)
            print(f"        <Geometry GeometryType=\"VXVYVZ\">{nl}", file = tf)
            print(f"          <DataItem Name=\"Xcoor\" Dimensions=\"{shape[0]}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">{nl}", file = tf)
            print(f"            {output_filename}.h5:/xcoor{nl}", file = tf)
            print(f"          </DataItem>{nl}", file = tf)
            print(f"          <DataItem Name=\"Ycoor\" Dimensions=\"{shape[1]}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">{nl}", file = tf)
            print(f"            {output_filename}.h5:/ycoor{nl}", file = tf)
            print(f"          </DataItem>{nl}", file = tf)
            print(f"          <DataItem Name=\"Zcoor\" Dimensions=\"{shape[2]}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">{nl}", file = tf)
            print(f"            {output_filename}.h5:/zcoor{nl}", file = tf)
            print(f"          </DataItem>{nl}", file = tf)
            print(f"        </Geometry>{nl}", file = tf)
            print(f"{nl}", file = tf)
            print(f"        <Attribute Name=\"{h5_var_name}\" AttributeType=\"{h5_attribute_type}\" Center=\"Node\">{nl}", file = tf)
            print(f"          <DataItem Dimensions=\"{shape[2]} {shape[1]} {shape[0]} {self.num_values_per_datapoint}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">{nl}", file = tf)
            print(f"            {output_filename}.h5:/{h5_dataset_name}{nl}", file = tf)
            print(f"          </DataItem>{nl}", file = tf)
            print(f"        </Attribute>{nl}", file = tf)
            print(f"      </Grid>{nl}", file = tf)
            print(f"{nl}", file = tf)
            print(f"  </Domain>{nl}", file = tf)
            print(f"</Xdmf>{nl}", file = tf)
            
    """
    getVariable functions.
    """
    def identify_database_file_points(self, points):
        # vectorize the mortoncurve.pack function.
        v_morton_pack = np.vectorize(self.mortoncurve.pack, otypes = [int])
        
        # convert the points to the center point position within their own bucket.
        center_points = (points / self.dx % 1) + self.cube_min_index
        # convert the points to gridded datapoints.
        datapoints = np.floor(points / self.dx).astype(int) % self.N
        # calculate the minimum and maximum chunk (x, y, z) corner point for each point in datapoints.
        chunk_min_xyzs = ((datapoints - self.cube_min_index) - ((datapoints - self.cube_min_index) % self.chunk_size)) % self.N
        chunk_max_xyzs = ((datapoints + self.cube_max_index) + (self.chunk_size - ((datapoints + self.cube_max_index) % self.chunk_size) - 1)) % self.N
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
        # zip the data.
        zipped_data = sorted(zip(chunk_min_mortons, chunk_keys, db_min_files, db_max_files, points, datapoints, center_points,
                                 chunk_min_xyzs, chunk_max_xyzs, chunk_min_mods, chunk_max_mods, original_points_indices), key = lambda x: (x[0], x[1]))
        
        # map the native bucket points to their respective db files and chunks.
        db_native_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # store an array of visitor bucket points.
        db_visitor_map = []
        
        for chunk_min_morton, chunk_key, db_min_file, db_max_file, point, datapoint, center_point, \
            chunk_min_xyz, chunk_max_xyz, chunk_min_mod, chunk_max_mod, original_point_index in zipped_data:
            # update the database file info if the morton code is outside of the previous database fil maximum morton limit.
            db_disk = db_min_file.split(os.sep)[self.database_file_disk_index]
            
            if db_min_file == db_max_file:
                # assign to native map.
                if chunk_key not in db_native_map[db_disk][db_min_file]:
                    db_native_map[db_disk][db_min_file][chunk_key].append((chunk_min_xyz, chunk_max_xyz, chunk_min_mod, chunk_max_mod))
    
                db_native_map[db_disk][db_min_file][chunk_key].append((point, datapoint, center_point, original_point_index))
            else:
                # assign to the visitor map.
                db_visitor_map.append((point, datapoint, center_point, original_point_index))
        
        return db_native_map, np.array(db_visitor_map, dtype = 'object')
    
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
            except FileNotFoundError:
                # update the distributed_cluster flag to False.
                distributed_cluster = False

                # using a local cluster if there is no premade distributed cluster.
                cluster = LocalCluster(n_workers = self.dask_maximum_processes, processes = True)
                client = Client(cluster)
    
            # available workers.
            workers = list(client.scheduler_info()['workers'])
            num_workers = len(workers)
            
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
        except FileNotFoundError:
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
        # set the byte order for reading the data from the files.
        dt = np.dtype(np.float32)
        dt = dt.newbyteorder('<')
        
        # the collection of local output data that will be returned to fill the complete output_data array.
        local_output_data = []
        # iterate over the database files and morton sub-boxes to read the data from.
        for db_file in db_native_map_data:
            zs = zarr.open(db_file + '_' + str(self.timepoint) + '.zarr' + os.sep + self.var_name, dtype = dt, mode = 'r')
            
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
                    local_output_data.append((original_point_index, (point, self.interpLagL(center_point, bucket))))
        
        return local_output_data
    
    def get_iso_points_variable_visitor(self, visitor_data, verbose = False):
            """
            reads and interpolates the user-requested visitor points.
            """
            # set the byte order for reading the data from the files.
            dt = np.dtype(np.float32)
            dt = dt.newbyteorder('<')

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
                            zs = zarr.open(db_file + '_' + str(self.timepoint) + '.zarr' + os.sep + self.var_name, dtype = dt, mode = 'r')

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
                local_output_data.append((point_data[3], (point_data[0], self.interpLagL(point_data[2], bucket))))

            return local_output_data
    