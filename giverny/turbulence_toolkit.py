import tracemalloc
from giverny.turbulence_gizmos.getCutout import *
from giverny.turbulence_gizmos.basic_gizmos import *

"""
convert indices to 0-based and retrieve a cutout of the isotropic cube.
"""
def getCutout(cube, cube_resolution, cube_title, \
              output_path, \
              x_range_original, y_range_original, z_range_original, var_original, timepoint_original):
    # housekeeping procedures.
    # -----
    # converts the 1-based axes ranges above to 0-based axes ranges, and truncates the ranges if they are longer than cube_resolution since 
    # the boundaries are periodic. output_data will be filled in with the duplicate data for the truncated data points after processing
    # so that the data files are not read redundantly.
    x_range, y_range, z_range = convert_to_0_based_ranges(x_range_original, y_range_original, z_range_original, cube_resolution)

    # convert the variable name from var_original into a variable identifier.
    var_original = var_original.lower()
    var = get_variable_identifier(var_original)
    
    # converts the 1-based timepoint above to a 0-based timepoint.
    timepoint = convert_to_0_based_value(timepoint_original)
    
    # create the output_path folder if it does not already exist.
    create_output_folder(output_path)

    # process the data.
    # -----
    # starting the tracemalloc library.
    #tracemalloc.start()
    # checking the memory usage of the program.
    #tracemem_start = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
    #tracemem_used_start = tracemalloc.get_tracemalloc_memory() / (1024**3)

    # parse the database files, generate the output_data matrix, and write the matrix to an hdf5 file.
    output_data = getCutout_process_data(cube, cube_resolution, cube_title, output_path, \
                                         x_range, y_range, z_range, var, timepoint, \
                                         x_range_original, y_range_original, z_range_original, var_original, timepoint_original)
    
    # memory used during processing as calculated by tracemalloc.
    #tracemem_end = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
    #tracemem_used_end = tracemalloc.get_tracemalloc_memory() / (1024**3)
    # stopping the tracemalloc library.
    #tracemalloc.stop()
    
    # see how much memory was used during processing.
    # memory used at program start.
    #print(f'\nstarting memory used in GBs [current, peak] = {tracemem_start}')
    # memory used by tracemalloc.
    #print(f'starting memory used by tracemalloc in GBs = {tracemem_used_start}')
    # memory used during processing.
    #print(f'ending memory used in GBs [current, peak] = {tracemem_end}')
    # memory used by tracemalloc.
    #print(f'ending memory used by tracemalloc in GBs = {tracemem_used_end}')
    
    return output_data
