import os
import sys
import dill
import glob
import h5py
import math
import mmap
import time
import shutil
import pathlib
import subprocess
import numpy as np
import SciServer.CasJobs as cj
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
    def __init__(self, dataset_title = '', cube_dimensions = 3):
        # check that dataset_title is a valid dataset title.
        check_dataset_title(dataset_title)
        
        # cube size.
        self.N = get_dataset_resolution(dataset_title)
        
        # turbulence dataset name, e.g. "isotropic8192" or "isotropic1024fine".
        self.dataset_title = dataset_title
        
        # setting up Morton curve.
        bits = int(math.log(self.N, 2))
        self.mortoncurve = morton.Morton(dimensions = cube_dimensions, bits = bits)
        
        # get the SciServer user name.
        user = Authentication.getKeystoneUserWithToken(Authentication.getToken()).userName
        # set the directory for saving and reading the pickled files.
        self.pickle_dir = pathlib.Path(f'/home/idies/workspace/Temporary/{user}/scratch/turbulence_pickled')
        # create the pickled directory if it does not already exist.
        create_output_folder(self.pickle_dir)
        
        # get map of the filepaths for all of the dataset binary files.
        self.init_filepaths()
        
        # get a cache of the metadata for the database files.
        self.init_cache()
        
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
    
    def identify_single_database_file_sub_boxes(self, box, var, timepoint, database_file_disk_index):
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
                    min_corner_info = self.get_file_for_point(min_corner_xyz, var, timepoint)
                    min_corner_db_file = min_corner_info[0]
                    database_file_disk = min_corner_db_file.split(os.sep)[database_file_disk_index]
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
    
    def get_file_for_point(self, datapoint, var = 'pr', timepoint = 0):
        """
        querying the cached SQL metadata for the file for the user specified grid point
        """
        # use periodic boundary conditions to adjust the x, y, and z values if they are outside the range of the whole dataset cube.
        datapoint = [point % self.N for point in datapoint]
        
        # query the cached SQL metadata for the user-specified grid point.
        cornercode, offset = self.get_offset(datapoint)
        t = self.cache[(self.cache['minLim'] <= cornercode) & (self.cache['maxLim'] >= cornercode)]
        t = t.iloc[0]
        f = self.filepaths[f'{t.ProductionDatabaseName}_{var}_{timepoint}']
        return f, cornercode, offset, t.minLim, t.maxLim
        
    def read_database_files_sequentially(self, user_single_db_boxes,
                                         num_values_per_datapoint, bytes_per_datapoint, voxel_side_length,
                                         missing_value_placeholder):
        result_output_data = []
        # iterate over the hard disk drives that the database files are stored on.
        for database_file_disk in user_single_db_boxes:
            # read in the voxel data from all of the database files on this disk.
            result_output_data += self.get_iso_points(user_single_db_boxes[database_file_disk],
                                                      num_values_per_datapoint, bytes_per_datapoint, voxel_side_length, missing_value_placeholder,
                                                      verbose = False)
        
        return result_output_data
    
    def read_database_files_in_parallel_with_dask(self, user_single_db_boxes,
                                                  num_values_per_datapoint, bytes_per_datapoint, voxel_side_length,
                                                  missing_value_placeholder, num_processes):
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
            cluster = LocalCluster(n_workers = num_processes, processes = True)
            client = Client(cluster)
        
        # number of hard disks that the database files being read are stored on.
        num_db_disks = len(user_single_db_boxes)
        
        # available workers.
        workers = list(client.scheduler_info()['workers'])
        num_workers = len(workers)
        
        # determine how many workers will be utilized for processing the data.
        utilized_workers = num_workers
        if utilized_workers > num_db_disks:
            utilized_workers = num_db_disks
        
        print(f'Database files are being read in parallel ({utilized_workers} workers utilized)...')
        sys.stdout.flush()
        
        result_output_data = []
        # iterate over the hard disk drives that the database files are stored on.
        for database_file_disk in user_single_db_boxes:
            # read in the voxel data from all of the database files on this disk.
            result_output_data.append(client.submit(self.get_iso_points, user_single_db_boxes[database_file_disk],
                                                    num_values_per_datapoint, bytes_per_datapoint, voxel_side_length, missing_value_placeholder,
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
                       num_values_per_datapoint = 1, bytes_per_datapoint = 4, voxel_side_length = 8, missing_value_placeholder = -999.9,
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
        voxel_cube_size = voxel_side_length**3
        
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
                full_min_xyz = [axis_range[0] - (axis_range[0] % voxel_side_length) for axis_range in user_box_ranges]
                full_max_xyz = [axis_range[1] + (voxel_side_length - (axis_range[1] % voxel_side_length) - 1) for axis_range in user_box_ranges]
                
                # origin points of voxels that overlap the user-specified box.
                voxel_origin_points = np.array([[x, y, z]
                                                 for z in range(full_min_xyz[2], full_max_xyz[2] + 1, voxel_side_length)
                                                 for y in range(full_min_xyz[1], full_max_xyz[1] + 1, voxel_side_length)
                                                 for x in range(full_min_xyz[0], full_max_xyz[0] + 1, voxel_side_length)]) % self.N
                
                # vectorized calculation of the morton indices for the voxel origin points.
                morton_mins = v_morton_pack(voxel_origin_points[:, 0], voxel_origin_points[:, 1], voxel_origin_points[:, 2])
                
                # calculate the number of periodic cubes that the user-specified box expands into beyond the boundary along each axis.
                cube_ms = np.array([math.floor(float(min_xyz[q]) / float(self.N)) * self.N for q in range(len(min_xyz))])

                # create copies of min_xyz and max_xyz to vectorize the voxel_mins and voxel_maxs calculations.
                tiled_min_xyz = np.tile(min_xyz, (len(voxel_origin_points), 1))
                tiled_max_xyz = np.tile(max_xyz, (len(voxel_origin_points), 1))
                # determine the minimum and maximum voxel ranges that overlap the user-specified box.
                voxel_mins = np.max([voxel_origin_points + cube_ms, tiled_min_xyz], axis = 0)
                voxel_maxs = np.min([voxel_origin_points + cube_ms + voxel_side_length - 1, tiled_max_xyz], axis = 0)
                
                # calculate the modulus of each voxel minimum and maximum and add 1 to voxel_maxs_mod for correctly
                # slicing partially overlapped voxels.
                voxel_mins_mod = voxel_mins % voxel_side_length
                voxel_maxs_mod = (voxel_maxs % voxel_side_length) + 1
                # adjust voxel_mins and voxel_maxs by min_xyz and add 1 to voxel_maxs for simplified local_output_array slicing.
                voxel_mins = voxel_mins - min_xyz
                voxel_maxs = voxel_maxs - min_xyz + 1
                
                # create the local output array for this box that will be filled and returned.
                local_output_array = np.full((xyz_diffs[2], xyz_diffs[1], xyz_diffs[0],
                                              num_values_per_datapoint), fill_value = missing_value_placeholder, dtype = np.float32)
                
                # iterates over the voxels and reads them from the memory map of the database file.
                for morton_index_min, voxel_min, voxel_max, voxel_min_mod, voxel_max_mod in \
                    sorted(zip(morton_mins, voxel_mins, voxel_maxs, voxel_mins_mod, voxel_maxs_mod), key = lambda x: x[0]):
                    l = np.frombuffer(mm, dtype = dt,
                                      count = num_values_per_datapoint * voxel_cube_size,
                                      offset = num_values_per_datapoint * bytes_per_datapoint * (morton_index_min - db_minLim))
                    
                    # reshape the data into a 3-d voxel.
                    l = l.reshape(voxel_side_length, voxel_side_length, voxel_side_length, num_values_per_datapoint)
                    
                    # put the voxel data into the local array.
                    local_output_array[voxel_min[2] : voxel_max[2],
                                       voxel_min[1] : voxel_max[1],
                                       voxel_min[0] : voxel_max[0]] = l[voxel_min_mod[2] : voxel_max_mod[2],
                                                                        voxel_min_mod[1] : voxel_max_mod[1],
                                                                        voxel_min_mod[0] : voxel_max_mod[0]]
                
                # checks to make sure that data was read in for all points.
                if missing_value_placeholder in local_output_array:
                    raise Exception(f'local_output_array was not filled correctly')
                
                # append the filled local_output_array into local_output_data.
                local_output_data.append((local_output_array, min_xyz, max_xyz))
        
            # close the open file object.
            open_db_file.close()
        
        return local_output_data
    
    def write_output_matrix_to_hdf5(self, output_data, output_path, output_filename, dataset_name):
        # write output_data to a hdf5 file.
        with h5py.File(output_path.joinpath(output_filename + '.h5'), 'w') as h5f:
            h5f.create_dataset(dataset_name, data = output_data)
            