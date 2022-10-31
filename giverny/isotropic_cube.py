import os
import sys
import dill
import glob
import h5py
import math
import mmap
import shutil
import pathlib
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
        # create the output directory if it does not already exist.
        create_output_folder(self.output_path)
        
        # set the directory for saving and reading the pickled files.
        self.pickle_dir = pathlib.Path(f'/home/idies/workspace/Temporary/{user}/scratch/turbulence_pickled')
        # create the pickled directory if it does not already exist.
        create_output_folder(self.pickle_dir)
        
        # get map of the filepaths for all of the dataset binary files.
        self.init_filepaths()
        
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
    
    def init_filepaths(self):
        # pickled file for saving the globbed filepaths.
        pickle_file = self.pickle_dir.joinpath(self.dataset_title + '_database_filepaths.pickle')
        
        try:
            # try reading the pickled file.
            with open(pickle_file, 'rb') as pickled_filepath:
                self.filepaths = dill.load(pickled_filepath)
        except FileNotFoundError:
            # map the filepaths to the part of each filename that matches "ProductionDatabaseName" in the SQL metadata for this dataset.
            self.filepaths = {}
            
            # get the common filename prefix for all files in this dataset, e.g. "iso8192" for the isotropic8192 dataset.
            dataset_filename_prefix = get_filename_prefix(self.dataset_title)
            # recursively search all sub-directories in the turbulence filedb system for the dataset binary files.
            filepaths = sorted(glob.glob(f'/home/idies/workspace/turb/**/{dataset_filename_prefix}*.bin', recursive = True))

            # map the filepaths to the filenames so that they can be easily retrieved.
            for filepath in filepaths:
                # part of the filenames that exactly matches the "ProductionDatabaseName" column stored in the SQL metadata, plus the variable
                # identifer (e.g. 'vel'), plus the timepoint.
                filename = filepath.split(os.sep)[-1].replace('.bin', '').strip()
                # only add the filepath to the dictionary once since there could be backup copies of the files.
                if filename not in self.filepaths:
                    self.filepaths[filename] = filepath
            
            # save self.filepaths to a pickled file.
            with open(pickle_file, 'wb') as pickled_filepath:
                dill.dump(self.filepaths, pickled_filepath)
                
    def init_cornercode_file_map(self):
        # pickled file for saving the db file cornercodes to filenames map.
        pickle_file = self.pickle_dir.joinpath(self.dataset_title + f'_cornercode_file_map-variable_{self.var}-timepoint_{self.timepoint}.pickle')
        
        try:
            # try reading the pickled file.
            with open(pickle_file, 'rb') as pickled_file_map:
                self.cornercode_file_map = dill.load(pickled_file_map)
        except FileNotFoundError:
            # create a map of the db file cornercodes to filenames for the whole dataset.
            self.cornercode_file_map = {}
            
            cornercode = 0
            while cornercode < self.N ** 3:
                # get the file info for the db file cornercode.
                f, db_minLim, db_maxLim = self.get_file_for_cornercode(cornercode)
                
                self.cornercode_file_map[db_minLim] = (f, db_minLim)
                
                cornercode = db_maxLim + 1
                
            # save self.cornercode_file_map to a pickled file.
            with open(pickle_file, 'wb') as pickled_file_map:
                dill.dump(self.cornercode_file_map, pickled_file_map)
    
    def init_interpolation_lookup_table(self):
        if self.sint != 'none':
            # pickled file for saving the interpolation coefficient lookup table.
            pickle_file = self.pickle_dir.joinpath(self.dataset_title + f'_{self.sint}_lookup_table.pickle')

            # lookup table resolution.
            self.NB = 10**5

            try:
                # try reading the pickled file.
                with open(pickle_file, 'rb') as pickled_lookup_table:
                    self.LW = dill.load(pickled_lookup_table)
            except FileNotFoundError:
                # create the interpolation coefficient lookup table.
                self.LW = self.getLagL()

                # save self.LW to a pickled file.
                with open(pickle_file, 'wb') as pickled_lookup_table:
                    dill.dump(self.LW, pickled_lookup_table)
                
    def init_interpolation_cube_size_lookup_table(self):
        pickle_file = self.pickle_dir.joinpath(self.dataset_title + f'_interpolation_cube_size_lookup_table.pickle')
        
        try:
            # try reading the pickled file.
            with open(pickle_file, 'rb') as pickled_lookup_table:
                self.interp_cube_sizes = dill.load(pickled_lookup_table)
        except FileNotFoundError:
            # create the interpolation cube size lookup table. the first number is the number of points on the left of the integer 
            # interpolation point, and the second number is the number of points on the right.
            self.interp_cube_sizes = {}
            self.interp_cube_sizes['lag4'] = [1, 3]
            self.interp_cube_sizes['m1q4'] = [1, 3]
            self.interp_cube_sizes['lag6'] = [2, 4]
            self.interp_cube_sizes['lag8'] = [3, 5]
            self.interp_cube_sizes['m2q8'] = [3, 5]
            self.interp_cube_sizes['none'] = [0, 0]
            
            # save self.interp_cube_sizes to a pickled file.
            with open(pickle_file, 'wb') as pickled_lookup_table:
                dill.dump(self.interp_cube_sizes, pickled_lookup_table)
                
        # lookup the interpolation cube size indices.
        self.cube_indices = self.interp_cube_sizes[self.sint]
        self.cube_min_index = self.cube_indices[0]
        self.cube_max_index = self.cube_indices[1]
        # store the bucket indices for determining how many voxels need to be read for each point.
        self.bucket_min_index = self.cube_min_index
        self.bucket_max_index = self.cube_max_index - 1
    
    def init_constants(self, var, timepoint, sint,
                       num_values_per_datapoint, bytes_per_datapoint,
                       voxel_side_length, missing_value_placeholder, database_file_disk_index, dask_maximum_processes):
        # create the constants.
        self.var = var
        self.timepoint = timepoint
        self.sint = sint
        self.num_values_per_datapoint = num_values_per_datapoint
        self.bytes_per_datapoint = bytes_per_datapoint
        self.voxel_side_length = voxel_side_length
        self.missing_value_placeholder = missing_value_placeholder
        self.database_file_disk_index = database_file_disk_index
        self.dask_maximum_processes = dask_maximum_processes
        
        # get a map of the files to cornercodes for all of the dataset binary files.
        self.init_cornercode_file_map()
    
    """
    interpolation functions.
    """
    def getLagL(self):
        frac = np.linspace(0, 1 - 1 / self.NB, self.NB)
        LW = []
        for fp in frac:
            LW.append(self.getLagC(fp))

        return LW
    
    #===============================================================================
    # Interpolating functions to compute the kernel, extract subcube and convolve
    #===============================================================================
    def getLagC(self, fr):
        #------------------------------------------------------
        # get the 1D vectors for the 8 point Lagrange weights
        # inline the constants, and write explicit for loop
        # for the C compilation
        #------------------------------------------------------
        # cdef int n.
        if self.sint == 'm1q4':
            # define the weights for M1Q4 spline interpolation.
            g = np.zeros(4)
            g[0] = fr * (fr * (-1.0 / 2.0 * fr + 1) - 1.0 / 2.0)
            g[1] = fr**2 * ((3.0 / 2.0) * fr - 5.0 / 2.0) + 1
            g[2] = fr * (fr * (-3.0 / 2.0 * fr + 2) + 1.0 / 2.0)
            g[3] = fr**2 * ((1.0 / 2.0) * fr - 1.0 / 2.0)
        elif self.sint == 'm2q8':
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
            if self.sint == 'lag4':
                wN = [1.,-3.,3.,-1.]
                g  = np.array([0,1.,0,0])
                # weight index.
                w_index = 1
            elif self.sint == 'lag6':
                wN = [1.,-5.,10.,-10.,5.,-1.]
                g  = np.array([0,0,1.,0,0,0])
                # weight index.
                w_index = 2
            elif self.sint == 'lag8':
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
            #---------------------------------------
            # assemble the 8x8x8 cube and convolve
            #---------------------------------------
            d = u[ix[2] - self.cube_min_index : ix[2] + self.cube_max_index,
                  ix[1] - self.cube_min_index : ix[1] + self.cube_max_index,
                  ix[0] - self.cube_min_index : ix[0] + self.cube_max_index,
                  :]

            ui = np.einsum('ijk,ijkl->l', gk, d)
        else:
            # 'none' spatial interpolation.
            ix = np.floor(p + 0.5).astype(np.int32)
            ui = np.array(u[ix[2], ix[1], ix[0], :])
        
        return ui
    
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
                    min_corner_info = self.get_file_for_point(min_corner_xyz)
                    min_corner_db_file = min_corner_info[0]
                    database_file_disk = min_corner_db_file.split(os.sep)[self.database_file_disk_index]
                    box_minLim = min_corner_info[3]
                    max_corner_xyz = self.mortoncurve.unpack(min_corner_info[4])
                    
                    # calculate the number of periodic cubes that the user-specified box expands into beyond the boundary along each axis.
                    cube_ms = [math.floor(float(min_corner_xyz[q]) / float(self.N)) * self.N for q in range(3)]
                    
                    # specify the box that is fully inside a database file.
                    box = [[min_corner_xyz[i], min(max_corner_xyz[i] + cube_ms[i], box_max_xyz[i])] for i in range(3)]
                    
                    # add the box axes ranges and the minimum morton limit to the map.
                    single_file_boxes[database_file_disk][min_corner_db_file].append((box, box_minLim))

                    # move to the next database file origin point.
                    current_x = max_corner_xyz[0] + cube_ms[0] + 1

                current_y = max_corner_xyz[1] + cube_ms[1] + 1

            current_z = max_corner_xyz[2] + cube_ms[2] + 1
    
        return single_file_boxes
        
    def get_offset(self, datapoint):
        """
        todo: is this code correct for velocity as well?  yes.
        """
        # morton curve index corresponding to the user specified x, y, and z values
        code = self.mortoncurve.pack(datapoint[0], datapoint[1], datapoint[2])
        # always looking at an 8 x 8 x 8 box around the grid point, so the shift is always 9 bits to determine 
        # the bottom left corner of the box. the cornercode (bottom left corner of the 8 x 8 x 8 box) is always 
        # in the same file as the user-specified grid point.
        # equivalent to 512 * (math.floor(code / 512))
        cornercode = (code >> 9) << 9
        corner = np.array(self.mortoncurve.unpack(cornercode))
        # calculates the offset between the grid point and corner of the box and converts it to a 4-byte float.
        offset = np.sum((np.array(datapoint) - corner) * np.array([1, 8, 64]))
        
        return cornercode, offset
    
    def get_file_for_point(self, datapoint):
        """
        querying the cached SQL metadata for the file for the user specified grid point
        """
        # use periodic boundary conditions to adjust the x, y, and z values if they are outside the range of the whole dataset cube.
        datapoint = [point % self.N for point in datapoint]
        
        # query the cached SQL metadata for the user-specified grid point.
        cornercode, offset = self.get_offset(datapoint)
        t = self.cache[(self.cache['minLim'] <= cornercode) & (self.cache['maxLim'] >= cornercode)]
        t = t.iloc[0]
        f = self.filepaths[f'{t.ProductionDatabaseName}_{self.var}_{self.timepoint}']
        return f, cornercode, offset, t.minLim, t.maxLim
        
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
                                                    verbose = False,
                                                    workers = workers))
        
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
        # set the byte order for reading the data from the binary files.
        dt = np.dtype(np.float32)
        dt = dt.newbyteorder('<')
        
        # vectorize the mortoncurve.pack function.
        v_morton_pack = np.vectorize(self.mortoncurve.pack, otypes = [int])
        
        # volume of the voxel cube.
        voxel_cube_size = self.voxel_side_length**3
        
        # the collection of local output data that will be returned to fill the complete output_data array.
        local_output_data = []
        # iterate over the database files and morton sub-boxes to read the data from.
        for db_file in user_single_db_boxes_disk_data:
            # create an open file object of the database file.
            open_db_file = open(db_file, 'rb')
            
            # create a memory map of the database file.
            size_bytes = os.fstat(open_db_file.fileno()).st_size
            mm = mmap.mmap(open_db_file.fileno(), length = size_bytes, access = mmap.ACCESS_READ)
            
            # iterate over the user box ranges corresponding to the morton voxels that will be read from this database file.
            for user_box_data in user_single_db_boxes_disk_data[db_file]:
                # retrieve the axes ranges of sub-box and the minimum morton limit of the database file.
                user_box_ranges = user_box_data[0]
                db_minLim = user_box_data[1]

                # retrieve the minimum and maximum (x, y, z) coordinates of the database file box that is going to be read in.
                min_xyz = np.array([axis_range[0] for axis_range in user_box_ranges])
                max_xyz = np.array([axis_range[1] for axis_range in user_box_ranges])
                xyz_diffs = max_xyz - min_xyz + 1
                
                # expand the boundaries of the box to encompass complete voxels.
                full_min_xyz = [axis_range[0] - (axis_range[0] % self.voxel_side_length) for axis_range in user_box_ranges]
                full_max_xyz = [axis_range[1] + (self.voxel_side_length - (axis_range[1] % self.voxel_side_length) - 1) for axis_range in user_box_ranges]
                
                # origin points of voxels that overlap the user-specified box.
                voxel_origin_points = np.array([[x, y, z]
                                                 for z in range(full_min_xyz[2], full_max_xyz[2] + 1, self.voxel_side_length)
                                                 for y in range(full_min_xyz[1], full_max_xyz[1] + 1, self.voxel_side_length)
                                                 for x in range(full_min_xyz[0], full_max_xyz[0] + 1, self.voxel_side_length)]) % self.N
                
                # vectorized calculation of the morton indices for the voxel origin points.
                morton_mins = v_morton_pack(voxel_origin_points[:, 0], voxel_origin_points[:, 1], voxel_origin_points[:, 2])
                
                # calculate the number of periodic cubes that the user-specified box expands into beyond the boundary along each axis.
                cube_ms = np.array([math.floor(float(min_xyz[q]) / float(self.N)) * self.N for q in range(len(min_xyz))])

                # create copies of min_xyz and max_xyz to vectorize the voxel_mins and voxel_maxs calculations.
                tiled_min_xyz = np.tile(min_xyz, (len(voxel_origin_points), 1))
                tiled_max_xyz = np.tile(max_xyz, (len(voxel_origin_points), 1))
                # determine the minimum and maximum voxel ranges that overlap the user-specified box.
                voxel_mins = np.max([voxel_origin_points + cube_ms, tiled_min_xyz], axis = 0)
                voxel_maxs = np.min([voxel_origin_points + cube_ms + self.voxel_side_length - 1, tiled_max_xyz], axis = 0)
                
                # calculate the modulus of each voxel minimum and maximum and add 1 to voxel_maxs_mod for correctly
                # slicing partially overlapped voxels.
                voxel_mins_mod = voxel_mins % self.voxel_side_length
                voxel_maxs_mod = (voxel_maxs % self.voxel_side_length) + 1
                # adjust voxel_mins and voxel_maxs by min_xyz and add 1 to voxel_maxs for simplified local_output_array slicing.
                voxel_mins = voxel_mins - min_xyz
                voxel_maxs = voxel_maxs - min_xyz + 1
                
                # create the local output array for this box that will be filled and returned.
                local_output_array = np.full((xyz_diffs[2], xyz_diffs[1], xyz_diffs[0],
                                              self.num_values_per_datapoint), fill_value = self.missing_value_placeholder, dtype = np.float32)
                
                # iterates over the voxels and reads them from the memory map of the database file.
                for morton_index_min, voxel_min, voxel_max, voxel_min_mod, voxel_max_mod in \
                    sorted(zip(morton_mins, voxel_mins, voxel_maxs, voxel_mins_mod, voxel_maxs_mod), key = lambda x: x[0]):
                    l = np.frombuffer(mm, dtype = dt,
                                      count = self.num_values_per_datapoint * voxel_cube_size,
                                      offset = self.num_values_per_datapoint * self.bytes_per_datapoint * (morton_index_min - db_minLim))
                    
                    # reshape the data into a 3-d voxel.
                    l = l.reshape(self.voxel_side_length, self.voxel_side_length, self.voxel_side_length, self.num_values_per_datapoint)
                    
                    # put the voxel data into the local array.
                    local_output_array[voxel_min[2] : voxel_max[2],
                                       voxel_min[1] : voxel_max[1],
                                       voxel_min[0] : voxel_max[0]] = l[voxel_min_mod[2] : voxel_max_mod[2],
                                                                        voxel_min_mod[1] : voxel_max_mod[1],
                                                                        voxel_min_mod[0] : voxel_max_mod[0]]
                
                # checks to make sure that data was read in for all points.
                if self.missing_value_placeholder in local_output_array:
                    raise Exception(f'local_output_array was not filled correctly')
                
                # append the filled local_output_array into local_output_data.
                local_output_data.append((local_output_array, min_xyz, max_xyz))
        
            # close the open file object.
            open_db_file.close()
        
        return local_output_data
    
    def write_output_matrix_to_hdf5(self, output_data, output_filename, dataset_name):
        # write output_data to a hdf5 file.
        with h5py.File(self.output_path.joinpath(output_filename + '.h5'), 'w') as h5f:
            h5f.create_dataset(dataset_name, data = output_data)
            
    """
    getVariable functions.
    """
    def get_voxel_origin_groups(self, bucket_min_x, bucket_min_y, bucket_min_z, bucket_max_x, bucket_max_y, bucket_max_z):
        # get arrays of the voxel origin poins for each bucket.
        return np.array([[x, y, z]
                         for z in range(bucket_min_z, (bucket_max_z if bucket_min_z <= bucket_max_z else self.N + bucket_max_z) + 1, 8)
                         for y in range(bucket_min_y, (bucket_max_y if bucket_min_y <= bucket_max_y else self.N + bucket_max_y) + 1, 8)
                         for x in range(bucket_min_x, (bucket_max_x if bucket_min_x <= bucket_max_x else self.N + bucket_max_x) + 1, 8)])
    
    def find_center_points(self, points):
        # find center coordinates of the bucket [e.g, (7.5 7.5 7.5) for the first bucket].
        center_points = points / self.dx % self.voxel_side_length
        center_points = np.where([center_point < self.bucket_min_index for center_point in center_points],
                                 [center_point + 8 for center_point in center_points],
                                 [center_point for center_point in center_points])
        
        return center_points
    
    def identify_database_file_points(self, points):
        # vectorize the mortoncurve.pack function.
        v_morton_pack = np.vectorize(self.mortoncurve.pack, otypes = [int])
        # vectorize the get_voxel_origin_groups function.
        v_get_voxel_origin_groups = np.vectorize(self.get_voxel_origin_groups, otypes = [list])
        
        # convert the points to the center point position within their own bucket.
        center_points = self.find_center_points(points)
        # convert the points to gridded datapoints.
        datapoints = np.floor(points / self.dx).astype(int) % self.N
        # calculate the minimum and maximum bucket (x, y, z) corner point for each point in datapoints.
        bucket_min_xyzs = ((datapoints - self.bucket_min_index) - ((datapoints - self.bucket_min_index) % self.voxel_side_length)) % self.N
        bucket_max_xyzs = ((datapoints + self.bucket_max_index) + (self.voxel_side_length - ((datapoints + self.bucket_max_index) % self.voxel_side_length) - 1)) % self.N
        # calculate the minimum and maximum bucket morton codes for each point in datapoints.
        bucket_min_mortons = v_morton_pack(bucket_min_xyzs[:, 0], bucket_min_xyzs[:, 1], bucket_min_xyzs[:, 2])
        bucket_max_mortons = v_morton_pack(bucket_max_xyzs[:, 0], bucket_max_xyzs[:, 1], bucket_max_xyzs[:, 2])
        # get the origin points for each voxel in the buckets.
        voxel_origin_groups = v_get_voxel_origin_groups(bucket_min_xyzs[:, 0], bucket_min_xyzs[:, 1], bucket_min_xyzs[:, 2],
                                                        bucket_max_xyzs[:, 0], bucket_max_xyzs[:, 1], bucket_max_xyzs[:, 2])
             
        # save the original indices for bucket_min_mortons, which corresponds to the orderering of the user-specified
        # points. these indices will be used for sorting output_data back to the user-specified points ordering.
        original_points_indices = [q for q in range(len(points))]
        # zip the data and sort by the bucket minimum morton codes.
        zipped_data = sorted(zip(bucket_min_mortons, bucket_max_mortons, points, center_points, voxel_origin_groups, original_points_indices), key = lambda x: x[0])
        
        # map the native bucket points to their respective db files and buckets.
        db_native_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # store an array of visitor bucket points.
        db_visitor_map = []
        
        # first morton code.
        bucket_min_morton = zipped_data[0][0]
        
        # database file info for the first morton code.
        file_info = self.get_file_for_cornercode(bucket_min_morton)
        db_file = file_info[0]
        db_disk = db_file.split(os.sep)[-3]
        db_minLim = file_info[1]
        db_maxLim = file_info[2]
        
        for bucket_min_morton, bucket_max_morton, point, center_point, voxel_origin_group, original_point_index in zipped_data:
            # update the database file info if the morton code is outside of the previous database fil maximum morton limit.
            if bucket_min_morton > db_maxLim:
                file_info = self.get_file_for_cornercode(bucket_min_morton)
                db_file = file_info[0]
                db_disk = db_file.split(os.sep)[-3]
                db_minLim = file_info[1]
                db_maxLim = file_info[2]
            
            # assign to native map.
            if ((bucket_max_morton <= db_maxLim) and (bucket_min_morton <= bucket_max_morton)):
                # convert the voxel_origin_group to a unique value that can used as a key in db_native_map.
                bucket_key = voxel_origin_group.tobytes()
                
                # assign to the native map.
                db_native_map[db_disk][db_file, db_minLim][bucket_key].append((point, center_point, original_point_index))
            else:
                # assign to the visitor map.
                db_visitor_map.append((point, center_point, original_point_index))
        
        # reformat db_native_map for distributed processing if the native data is distributed across more than 1 db disk.
        if len(db_native_map) > 1:
            for db_disk in db_native_map:
                db_native_map[db_disk] = [point_pair
                                          for file_key in db_native_map[db_disk]
                                          for bucket_key in db_native_map[db_disk][file_key]
                                          for point_pair in db_native_map[db_disk][file_key][bucket_key]]
        
        return db_native_map, np.array(db_visitor_map, dtype = 'object')
    
    def get_file_for_cornercode(self, cornercode):
        """
        querying the cached SQL metadata for the file for the specified morton cornercode.
        """
        # query the cached SQL metadata for the user-specified grid point.
        t = self.cache[(self.cache['minLim'] <= cornercode) & (self.cache['maxLim'] >= cornercode)]
        t = t.iloc[0]
        f = self.filepaths[f'{t.ProductionDatabaseName}_{self.var}_{self.timepoint}']
        return f, t.minLim, t.maxLim
    
    def read_natives_sequentially_variable(self, db_native_map, native_output_data):
        # native data.
        # iterate over the data volumes that the database files are stored on.
        for database_file_disk in db_native_map:
            # read in the voxel data from all of the database files on this disk.
            native_output_data += self.get_iso_points_variable_one_native_volume(db_native_map[database_file_disk], verbose = False)
            
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
                # scatter the chunk.
                db_map_scatter = client.scatter(db_visitor_map_chunk, broadcast = False, workers = worker)
                # submit the chunk for parallel processing.
                temp_visitor_output_data.append(client.submit(self.get_iso_points_variable_visitor, db_map_scatter, verbose = False, workers = worker))
                
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
                
                # scatter the data.
                db_map_scatter = client.scatter(np.array(db_native_map[database_file_disk], dtype = 'object'), broadcast = False, workers = worker)
                # submit the data for parallel processing.
                result_output_data.append(client.submit(self.get_iso_points_variable, db_map_scatter, verbose = False, workers = worker))
        
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
                # scatter the data.
                db_map_scatter = client.scatter(db_visitor_map_chunk, broadcast = False, workers = worker)
                # submit the data for parallel processing.
                result_output_data.append(client.submit(self.get_iso_points_variable_visitor, db_map_scatter, verbose = False, workers = worker))
        
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
    
    def get_iso_points_variable_one_native_volume(self, db_native_map_data, verbose = False):
        """
        reads and interpolates the user-requested native points in a single database volume.
        """
        # set the byte order for reading the data from the binary files.
        dt = np.dtype(np.float32)
        dt = dt.newbyteorder('<')
        
        # vectorize the mortoncurve.pack function.
        v_morton_pack = np.vectorize(self.mortoncurve.pack, otypes = [int]) 
        
        # volume of the voxel- cube.
        voxel_cube_size = self.voxel_side_length**3
        
        # the collection of local output data that will be returned to fill the complete output_data array.
        local_output_data = []
        
        # empty voxel bucket.
        bucket = np.zeros((16, 16, 16, self.num_values_per_datapoint))
        
        # iterate of the db files.
        for db_file, db_minLim in db_native_map_data:
            # create an open file object of the database file.
            with open(db_file, 'rb') as open_db_file:
                # create a memory map of the database file.
                size_bytes = os.fstat(open_db_file.fileno()).st_size
                mm = mmap.mmap(open_db_file.fileno(), length = size_bytes, access = mmap.ACCESS_READ)

                # build the buckets and interpolate the points.
                db_file_data = db_native_map_data[db_file, db_minLim]
                for bucket_key in db_file_data:                    
                    bucket_data = db_file_data[bucket_key]
                    # only use the first point to calculate the bucket.                     
                    point = bucket_data[0][0]
                    
                    # convert the point to a gridded datapoint.
                    datapoint = np.floor(point / self.dx).astype(int) % self.N
                    # calculate the minimum and maximum bucket (x, y, z) corner point for datapoint.
                    bucket_min_xyz = ((datapoint - self.bucket_min_index) - ((datapoint - self.bucket_min_index) % self.voxel_side_length)) % self.N
                    bucket_max_xyz = ((datapoint + self.bucket_max_index) + (self.voxel_side_length - ((datapoint + self.bucket_max_index) % self.voxel_side_length) - 1)) % self.N
                    # calculate the minimum and maximum bucket morton code for datapoint.
                    bucket_min_morton = self.mortoncurve.pack(bucket_min_xyz[0], bucket_min_xyz[1], bucket_min_xyz[2])
                    bucket_max_morton = self.mortoncurve.pack(bucket_max_xyz[0], bucket_max_xyz[1], bucket_max_xyz[2])
                    # get the origin points for each voxel in the bucket.
                    voxel_origin_group = self.get_voxel_origin_groups(bucket_min_xyz[0], bucket_min_xyz[1], bucket_min_xyz[2],
                                                                      bucket_max_xyz[0], bucket_max_xyz[1], bucket_max_xyz[2])
                                                       
                    # get the voxel origin group inside the dataset domain.
                    voxel_origin_group_boundary = voxel_origin_group % self.N
                    # calculate the morton codes for the minimum point in each voxel of the bucket.
                    morton_mins = v_morton_pack(voxel_origin_group_boundary[:, 0], voxel_origin_group_boundary[:, 1], voxel_origin_group_boundary[:, 2])
                    # adjust the voxel origin points to the domain [0, 15] for filling the empty bucket array.
                    voxel_origin_points = voxel_origin_group - voxel_origin_group[0]
                    
                    # iterates over the voxels and reads them from the memory map of the database file.
                    for morton_min, voxel_origin_point in zip(morton_mins, voxel_origin_points):
                        l = np.frombuffer(mm, dtype = dt,
                                          count = self.num_values_per_datapoint * voxel_cube_size,
                                          offset = self.num_values_per_datapoint * self.bytes_per_datapoint * (morton_min - db_minLim))

                        # reshape the data into a 3-d voxel.
                        l = l.reshape(self.voxel_side_length, self.voxel_side_length, self.voxel_side_length, self.num_values_per_datapoint)

                        # fill the bucket.
                        bucket[voxel_origin_point[2] : voxel_origin_point[2] + 8,
                               voxel_origin_point[1] : voxel_origin_point[1] + 8,
                               voxel_origin_point[0] : voxel_origin_point[0] + 8,
                               :] = l

                    for point, center_point, original_point_index in bucket_data:
                        # interpolate the points and use a lookup table for faster interpolations.
                        local_output_data.append((original_point_index, (point, self.interpLagL(center_point, bucket))))
        
        return local_output_data
    
    def get_iso_points_variable(self, db_native_map_data, verbose = False):
        """
        reads and interpolates the user-requested native points in multiple database volumes.
        """
        # set the byte order for reading the data from the binary files.
        dt = np.dtype(np.float32)
        dt = dt.newbyteorder('<')
        
        # vectorize the mortoncurve.pack function.
        v_morton_pack = np.vectorize(self.mortoncurve.pack, otypes = [int])
        # vectorize the get_voxel_origin_groups function.
        v_get_voxel_origin_groups = np.vectorize(self.get_voxel_origin_groups, otypes = [list])
        
        # volume of the voxel- cube.
        voxel_cube_size = self.voxel_side_length**3
        
        # the collection of local output data that will be returned to fill the complete output_data array.
        local_output_data = []
        
        # empty voxel bucket.
        bucket = np.zeros((16, 16, 16, self.num_values_per_datapoint))
        
        # convert the points to gridded datapoints.
        datapoints = np.array([np.floor(point / self.dx).astype(int) % self.N for point in db_native_map_data[:, 0]])
        # calculate the minimum and maximum bucket (x, y, z) corner points for each datapoint.
        bucket_min_xyzs = ((datapoints - self.bucket_min_index) - ((datapoints - self.bucket_min_index) % self.voxel_side_length)) % self.N
        bucket_max_xyzs = ((datapoints + self.bucket_max_index) + (self.voxel_side_length - ((datapoints + self.bucket_max_index) % self.voxel_side_length) - 1)) % self.N
        # calculate the morton codes for each datapoint.
        mortons = v_morton_pack(datapoints[:, 0], datapoints[:, 1], datapoints[:, 2])
        # calculate the db file cornercodes for each morton code.
        db_cornercodes = (mortons >> 27) << 27

        # get the origin points for each voxel in the buckets.
        voxel_origin_groups = v_get_voxel_origin_groups(bucket_min_xyzs[:, 0], bucket_min_xyzs[:, 1], bucket_min_xyzs[:, 2],
                                                        bucket_max_xyzs[:, 0], bucket_max_xyzs[:, 1], bucket_max_xyzs[:, 2])

        current_file = ''
        current_bucket = ''
        # build the buckets and interpolate the points.
        for point_data, voxel_origin_group, db_cornercode in zip(db_native_map_data, voxel_origin_groups, db_cornercodes):
            db_file, db_minLim = self.cornercode_file_map[db_cornercode]
            
            # create a memory map of the file if the current point is in a different file from the previous point.
            if current_file != db_file:
                # create an open file object of the database file.
                with open(db_file, 'rb') as open_db_file:
                    # create a memory map of the database file.
                    size_bytes = os.fstat(open_db_file.fileno()).st_size
                    mm = mmap.mmap(open_db_file.fileno(), length = size_bytes, access = mmap.ACCESS_READ)

                # update current_file.
                current_file = db_file

            # only update the bucket if the current point is in a different bucket than the previous point.
            bucket_key = voxel_origin_group.tobytes()
            if current_bucket != bucket_key:
                # get the voxel origin group inside the dataset domain.
                voxel_origin_group_boundary = voxel_origin_group % self.N
                # calculate the morton codes for the minimum point in each voxel of the bucket.
                morton_mins = v_morton_pack(voxel_origin_group_boundary[:, 0], voxel_origin_group_boundary[:, 1], voxel_origin_group_boundary[:, 2])
                # adjust the voxel origin points to the domain [0, 15] for filling the empty bucket array.
                voxel_origin_points = voxel_origin_group - voxel_origin_group[0]

                # iterates over the voxels and reads them from the memory map of the database file.
                for morton_min, voxel_origin_point in zip(morton_mins, voxel_origin_points):
                    l = np.frombuffer(mm, dtype = dt,
                                      count = self.num_values_per_datapoint * voxel_cube_size,
                                      offset = self.num_values_per_datapoint * self.bytes_per_datapoint * (morton_min - db_minLim))

                    # reshape the data into a 3-d voxel.
                    l = l.reshape(self.voxel_side_length, self.voxel_side_length, self.voxel_side_length, self.num_values_per_datapoint)

                    # fill the bucket.
                    bucket[voxel_origin_point[2] : voxel_origin_point[2] + 8,
                           voxel_origin_point[1] : voxel_origin_point[1] + 8,
                           voxel_origin_point[0] : voxel_origin_point[0] + 8,
                           :] = l

                current_bucket = bucket_key

            # interpolate the points and use a lookup table for faster interpolations.
            local_output_data.append((point_data[2], (point_data[0], self.interpLagL(point_data[1], bucket))))
        
        return local_output_data
    
    def get_iso_points_variable_visitor(self, visitor_data, verbose = False):
        """
        reads and interpolates the user-requested visitor points.
        """
        # set the byte order for reading the data from the binary files.
        dt = np.dtype(np.float32)
        dt = dt.newbyteorder('<')
        
        # vectorize the mortoncurve.pack function.
        v_morton_pack = np.vectorize(self.mortoncurve.pack, otypes = [int])
        
        # volume of the voxel cube.
        voxel_cube_size = self.voxel_side_length**3
        
        # the collection of local output data that will be returned to fill the complete output_data array.
        local_output_data = []
        
        # empty voxel bucket.
        bucket = np.zeros((16, 16, 16, self.num_values_per_datapoint))
        
        current_bucket = ''
        for point_data in visitor_data:
            # only use the first point to calculate the bucket.
            point = point_data[0]
            
            # convert the point to a gridded datapoint.
            datapoint = np.floor(point / self.dx).astype(int) % self.N
            # calculate the minimum and maximum bucket (x, y, z) corner point for datapoint.
            bucket_min_xyz = ((datapoint - self.bucket_min_index) - ((datapoint - self.bucket_min_index) % self.voxel_side_length)) % self.N
            bucket_max_xyz = ((datapoint + self.bucket_max_index) + (self.voxel_side_length - ((datapoint + self.bucket_max_index) % self.voxel_side_length) - 1)) % self.N
            # get the origin points for each voxel in the bucket.
            voxel_origin_group = self.get_voxel_origin_groups(bucket_min_xyz[0], bucket_min_xyz[1], bucket_min_xyz[2],
                                                              bucket_max_xyz[0], bucket_max_xyz[1], bucket_max_xyz[2])
    
            # only update the bucket if the new point is in a different bucket than the previous point.
            bucket_key = voxel_origin_group.tobytes()
            if current_bucket != bucket_key:
                # get the voxel origin group inside the dataset domain.
                voxel_origin_group_boundary = voxel_origin_group % self.N
                # calculate the morton codes for the minimum point in each voxel of the bucket.
                morton_mins = v_morton_pack(voxel_origin_group_boundary[:, 0], voxel_origin_group_boundary[:, 1], voxel_origin_group_boundary[:, 2])
                # adjust the voxel origin points to the domain [0, 15] for filling the empty bucket array.
                voxel_origin_points = voxel_origin_group - voxel_origin_group[0]

                # calculate the db file cornercodes for each morton code.
                db_cornercodes = (morton_mins >> 27) << 27
                # identify the database files that will need to be read for this bucket.
                db_files = [self.cornercode_file_map[morton_code] for morton_code in db_cornercodes]

                current_file = ''
                # iterate of the db files.
                for db_file_info, morton_min, voxel_origin_point in sorted(zip(db_files, morton_mins, voxel_origin_points), key = lambda x: x[1]):
                    db_file = db_file_info[0]
                    db_minLim = db_file_info[1]

                    # create a memory map of the file if the current point is in a different file from the previous point.
                    if db_file != current_file:
                        # create an open file object of the database file.
                        with open(db_file, 'rb') as open_db_file:
                            # create a memory map of the database file.
                            size_bytes = os.fstat(open_db_file.fileno()).st_size
                            mm = mmap.mmap(open_db_file.fileno(), length = size_bytes, access = mmap.ACCESS_READ)

                        # update current_file.
                        current_file = db_file

                    l = np.frombuffer(mm, dtype = dt,
                                      count = self.num_values_per_datapoint * voxel_cube_size,
                                      offset = self.num_values_per_datapoint * self.bytes_per_datapoint * (morton_min - db_minLim))

                    # reshape the data into a 3-d voxel.
                    l = l.reshape(self.voxel_side_length, self.voxel_side_length, self.voxel_side_length, self.num_values_per_datapoint)

                    # fill the bucket.
                    bucket[voxel_origin_point[2] : voxel_origin_point[2] + 8,
                           voxel_origin_point[1] : voxel_origin_point[1] + 8,
                           voxel_origin_point[0] : voxel_origin_point[0] + 8,
                           :] = l
                    
                # update current_bucket.
                current_bucket = bucket_key
            
            # interpolate the point and use a lookup table for faster interpolation.
            local_output_data.append((point_data[2], (point_data[0], self.interpLagL(point_data[1], bucket))))
        
        return local_output_data
    