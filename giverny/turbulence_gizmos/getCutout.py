import os
import sys
import math
import time
import numpy as np
import xarray as xr
from giverny.turbulence_gizmos.basic_gizmos import *

def getCutout_process_data(cube, cube_resolution, cube_title, output_path, \
                           x_range, y_range, z_range, var, timepoint, \
                           x_range_original, y_range_original, z_range_original, var_original, timepoint_original):
    # calculate how much time it takes to run the code.
    start_time = time.perf_counter()

    # data constants.
    # -----
    # index that pulls out the parent folder for a database file when the filepath is split on forward slash ("/"). the parent folder
    # references the hard disk drive that the database file is stored on.
    database_file_disk_index = -3
    # the maximum number of python processes that dask will be allowed to create when using a local cluster.
    dask_maximum_processes = 4
    # placeholder for missing values that will be used to fill the output_data array when it is initialized.
    missing_value_placeholder = -999.9
    # bytes per value associated with a datapoint.
    bytes_per_datapoint = 4
    # maximum data size allowed to be retrieved, in gigabytes (GB).
    max_data_size = 3.0
    # smallest sub-box size to recursively shrink to. if this size box is only partially contained in the user-specified box, then
    # the (X, Y, Z) points outside of the user-specified box will be trimmed.  the value is the length of one side of the cube.
    voxel_side_length = 8

    # the number of values to read per datapoint. for pressure data this value is 1.  for velocity
    # data this value is 3, because there is a velocity measurement along each axis.
    num_values_per_datapoint = get_num_values_per_datapoint(var)

    # used for determining the indices in the output array for each X, Y, Z datapoint.
    x_min = x_range[0]
    y_min = y_range[0]
    z_min = z_range[0]
    
    # number of original datapoints along each axis specified by the user. used for checking that the user did not request
    # too much data.
    x_axis_original_length = x_range_original[1] - x_range_original[0] + 1
    y_axis_original_length = y_range_original[1] - y_range_original[0] + 1
    z_axis_original_length = z_range_original[1] - z_range_original[0] + 1

    # used for creating the 3-D output array using numpy.
    x_axis_length = x_range[1] - x_range[0] + 1
    y_axis_length = y_range[1] - y_range[0] + 1
    z_axis_length = z_range[1] - z_range[0] + 1

    # total number of datapoints, used for checking if the user requested too much data.
    num_datapoints = x_axis_original_length * y_axis_original_length * z_axis_original_length
    # total size of data, in GBs, requested by the user's box.
    requested_data_size = (num_datapoints * bytes_per_datapoint * num_values_per_datapoint) / float(1024**3)
    # maximum number of datapoints that can be read in. currently set to 3 GBs worth of datapoints.
    max_datapoints = int((max_data_size * (1024**3)) / (bytes_per_datapoint * float(num_values_per_datapoint)))
    # approximate max size of a cube representing the maximum data points. this number is rounded down.
    approx_max_cube = int(max_datapoints**(1/3))

    if requested_data_size > max_data_size:
        raise ValueError(f'Please specify a box with fewer than {max_datapoints} data points. This represents an approximate cube size ' + \
                         f'of ({approx_max_cube} x {approx_max_cube} x {approx_max_cube}).')

    # begin processing of data.
    # -----
    print('Note: For larger boxes, e.g. 512-cubed and up, processing will take approximately 1 minute or more...\n' + '-' * 5)
    sys.stdout.flush()
    
    # -----
    # get a map of the database files where all the data points are in.
    print('\nStep 1: Determining which database files the user-specified box is found in...\n' + '-' * 25)
    sys.stdout.flush()
    
    # calculate how much time it takes to run step 1.
    start_time_step1 = time.perf_counter()

    user_single_db_boxes = cube.identify_single_database_file_sub_boxes(x_range, y_range, z_range, var, timepoint)

    print(f'number of database files that the user-specified box is found in:\n{len(user_single_db_boxes)}\n')
    sys.stdout.flush()
    
    # calculate how much time it takes to run step 1.
    end_time_step1 = time.perf_counter()

    print('Successfully completed.\n' + '-' * 5)
    sys.stdout.flush()
    
    # -----
    # recursively break down each single file box into sub-boxes, each of which is exactly one of the sub-divided cubes of the database file.
    print('\nStep 2: Recursively breaking down the portion of the user-specified box in each database file into voxels...\n' + '-' * 25)
    sys.stdout.flush()
    
    # calculate how much time it takes to run step 2.
    start_time_step2 = time.perf_counter()
    
    # iterates over the database files to figure out how many different hard disk drives these database files are stored on. if the number of disks
    # is greater than 1, then processing of the data will be distributed across several python processes using dask to speed up the processing 
    # time. if all of the database files are stored on 1 hard disk drive, then the data will be processed sequentially using base python.
    database_file_disks = set([])
    sub_db_boxes = {}
    for db_file in sorted(user_single_db_boxes, key = lambda x: os.path.basename(x)):
        # the parent folder for the database file corresponds to the hard disk drive that the file is stored on.
        database_file_disk = db_file.split('/')[database_file_disk_index]
        
        # add the folder to the set of folders already identified. this will be used to determine if dask is needed for processing.
        database_file_disks.add(database_file_disk)
        
        # create a new dictionary for all of the database files that are stored on this disk.
        if database_file_disk not in sub_db_boxes:
            sub_db_boxes[database_file_disk] = {}
            # keeping track of the original user-specified box ranges in case the user specifies a box outside of the dataset cube.
            sub_db_boxes[database_file_disk][db_file] = {}
        elif db_file not in sub_db_boxes[database_file_disk]:
            sub_db_boxes[database_file_disk][db_file] = {}
        
        for user_db_box_data in user_single_db_boxes[db_file]:
            user_db_box = user_db_box_data[0]
            user_db_box_minLim = user_db_box_data[1]
            # convert the user_db_box list of lists to a tuple of tuples so that it can be used as a key, along with user_db_box_minLim 
            # in the sub_db_boxes dictionary.
            user_db_box_key = (tuple([tuple(user_db_box_range) for user_db_box_range in user_db_box]), user_db_box_minLim)

            morton_voxels_to_read = cube.identify_sub_boxes_in_file(user_db_box, var, timepoint, voxel_side_length)

            # update sub_db_boxes with the information for reading in the database files.
            sub_db_boxes[database_file_disk][db_file][user_db_box_key] = morton_voxels_to_read
    
    min_file_boxes = np.min([len(sub_db_boxes[database_file_disk][db_file][user_db_box_key]) 
                             for database_file_disk in sub_db_boxes 
                             for db_file in sub_db_boxes[database_file_disk] 
                             for user_db_box_key in sub_db_boxes[database_file_disk][db_file]])
    max_file_boxes = np.max([len(sub_db_boxes[database_file_disk][db_file][user_db_box_key]) 
                             for database_file_disk in sub_db_boxes 
                             for db_file in sub_db_boxes[database_file_disk] 
                             for user_db_box_key in sub_db_boxes[database_file_disk][db_file]])
    
    print('sub-box statistics for the database file(s):\n-')
    print(f'minimum number of sub-boxes to read in a database file:\n{min_file_boxes}')
    print(f'maximum number of sub-boxes to read in a database file:\n{max_file_boxes}\n')
    sys.stdout.flush()
    
    # calculate how much time it takes to run step 2.
    end_time_step2 = time.perf_counter()
    
    print('Successfully completed.\n' + '-' * 5)
    sys.stdout.flush()

    # -----
    # read the data.
    print('\nStep 3: Reading the data from all of the database files and storing the values into a matrix...\n' + '-' * 25)
    sys.stdout.flush()
    
    # calculate how much time it takes to run step 3.
    start_time_step3 = time.perf_counter()
    
    # pre-fill the output data 3-d array that will be filled with the data that is read in. initially the datatype is set to "f" (float)
    # so that the array is filled with the missing placeholder values (-999.9). 
    output_data = np.full((z_axis_length, y_axis_length, x_axis_length, num_values_per_datapoint), fill_value = missing_value_placeholder, dtype = 'f')
    
    # determines if the database files will be read sequentially with base python, or in parallel with dask.
    num_db_disks = len(database_file_disks)
    if num_db_disks == 1:
        # sequential processing.
        print('Database files are being read sequentially...')
        sys.stdout.flush()
        
        result_output_data = cube.read_database_files_sequentially(sub_db_boxes, \
                                                                   x_min, y_min, z_min, x_range, y_range, z_range, \
                                                                   num_values_per_datapoint, bytes_per_datapoint, voxel_side_length, \
                                                                   missing_value_placeholder)
    else:
        # parallel processing.
        # optimizes the number of processes that are used by dask and makes sure that the number of processes does not exceed dask_maximum_processes.
        num_processes = dask_maximum_processes
        if num_db_disks < dask_maximum_processes:
            num_processes = num_db_disks
        
        result_output_data = cube.read_database_files_in_parallel_with_dask(sub_db_boxes, \
                                                                            x_min, y_min, z_min, x_range, y_range, z_range, \
                                                                            num_values_per_datapoint, bytes_per_datapoint, voxel_side_length, \
                                                                            missing_value_placeholder, num_processes)
    
    # iterate over the results to fill output_data.
    for result in result_output_data:
        output_data[result[1][2] - z_min : result[2][2] - z_min + 1, \
                    result[1][1] - y_min : result[2][1] - y_min + 1, \
                    result[1][0] - x_min : result[2][0] - x_min + 1] = result[0]
    
    # determines how many copies of data need to be me made along each axis when the number of datapoints the user specified
    # exceeds cube_resolution. note: no copies of the data values should be made, hence data_value_multiplier equals 1.
    x_axis_multiplier = int(math.ceil(float(x_axis_original_length) / float(cube_resolution)))
    y_axis_multiplier = int(math.ceil(float(y_axis_original_length) / float(cube_resolution)))
    z_axis_multiplier = int(math.ceil(float(z_axis_original_length) / float(cube_resolution)))
    data_value_multiplier = 1
    
    # duplicates the data along the z-, y-, and x-axes of output_data if the the user asked for more datapoints than cube_resolution along any axis.
    if x_axis_multiplier > 1 or y_axis_multiplier > 1 or z_axis_multiplier > 1:
        output_data = np.tile(output_data, (z_axis_multiplier, y_axis_multiplier, x_axis_multiplier, data_value_multiplier))
        # truncate any extra datapoints from the duplicate data outside of the original range of the datapoints specified by the user.
        output_data = np.copy(output_data[0 : z_axis_original_length, 0 : y_axis_original_length, 0 : x_axis_original_length, :])
    
    # convert output_data from a numpy array to a xarray.
    z_coords = range(z_range_original[0], z_range_original[1] + 1)
    y_coords = range(y_range_original[0], y_range_original[1] + 1)
    x_coords = range(x_range_original[0], x_range_original[1] + 1)

    output_data = xr.DataArray(output_data, 
                               coords = {'z':z_coords, 'y':y_coords, 'x':x_coords, 't':timepoint_original}, 
                               dims = ['z', 'y', 'x', 'v'])
    
    # checks to make sure that data was read in for all points.
    if missing_value_placeholder in output_data:
        raise Exception(f'output_data was not filled correctly')
    
    # calculate how much time it takes to run step 3.
    end_time_step3 = time.perf_counter()
    
    print('\nSuccessfully completed.\n' + '-' * 5)
    sys.stdout.flush()
    
    # -----
    # write the output file.
    print('\nStep 4: Writing the output matrix to a hdf5 file...\n' + '-' * 25)
    sys.stdout.flush()
    
    # calculate how much time it takes to run step 4.
    start_time_step4 = time.perf_counter()
    
    # write output_data to a hdf5 file.
    # the output filename specifies the title of the cube, and the x-, y-, and z-ranges so that the file is unique. 1 is added to all of the 
    # ranges, and the timepoint, because python uses 0-based indices, and the output is desired to be 1-based indices.
    output_filename = f'{cube_title}_{var_original}_' + \
                      f't{timepoint_original}_' + \
                      f'z{z_range_original[0]}-{z_range_original[1]}_' + \
                      f'y{y_range_original[0]}-{y_range_original[1]}_' + \
                      f'x{x_range_original[0]}-{x_range_original[1]}'
    
    # formats the dataset name for the hdf5 output file. "untitled" is a placeholder.
    dataset_name = var_original.title()
        
    # adds the timpoint information, formatted with leading zeros out to 1000, to dataset_name. 1 is added to timepoint because python uses
    # 0-based indices, and the output is desired to be 1-based indices.
    dataset_name += '_' + str(timepoint_original).zfill(4)
    
    # writes the output file.
    #cube.write_output_matrix_to_hdf5(output_data, output_path, output_filename, dataset_name)
    
    # calculate how much time it takes to run step 4.
    end_time_step4 = time.perf_counter()
    
    print('\nSuccessfully completed.\n' + '-' * 5)
    sys.stdout.flush()

    end_time = time.perf_counter()
    
    # see how long the program took to run.
    print(f'\nstep 1 time elapsed = {round(end_time_step1 - start_time_step1, 3)} seconds ({round((end_time_step1 - start_time_step1) / 60, 3)} minutes)')
    print(f'step 2 time elapsed = {round(end_time_step2 - start_time_step2, 3)} seconds ({round((end_time_step2 - start_time_step2) / 60, 3)} minutes)')
    print(f'step 3 time elapsed = {round(end_time_step3 - start_time_step3, 3)} seconds ({round((end_time_step3 - start_time_step3) / 60, 3)} minutes)')
    print(f'step 4 time elapsed = {round(end_time_step4 - start_time_step4, 3)} seconds ({round((end_time_step4 - start_time_step4) / 60, 3)} minutes)')
    print(f'total time elapsed = {round(end_time - start_time, 3)} seconds ({round((end_time - start_time) / 60, 3)} minutes)')
    sys.stdout.flush()
    
    print('\nData processing pipeline has completed successfully.\n' + '-' * 5)
    sys.stdout.flush()
    
    return output_data