import pathlib
import tracemalloc
from giverny.turbulence_gizmos.getCutout import *
from giverny.turbulence_gizmos.basic_gizmos import *

"""
retrieve a cutout of the isotropic cube.
"""
def getCutout(cube, cube_title,
              output_path,
              axes_ranges_original, strides, var_original, timepoint_original,
              trace_memory = False):
    # housekeeping procedures.
    # -----
    output_path, var, axes_ranges, timepoint, cube_resolution = \
        housekeeping_procedures(cube_title, output_path, axes_ranges_original, strides, var_original, timepoint_original)
    
    # process the data.
    # -----
    # starting the tracemalloc library.
    if trace_memory:
        tracemalloc.start()
        # checking the memory usage of the program.
        tracemem_start = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_start = tracemalloc.get_tracemalloc_memory() / (1024**3)

    # parse the database files, generate the output_data matrix, and write the matrix to an hdf5 file.
    output_data = getCutout_process_data(cube, cube_resolution, cube_title, output_path,
                                         axes_ranges, var, timepoint,
                                         axes_ranges_original, strides, var_original, timepoint_original)
    
    # closing the tracemalloc library.
    if trace_memory:
        # memory used during processing as calculated by tracemalloc.
        tracemem_end = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_end = tracemalloc.get_tracemalloc_memory() / (1024**3)
        # stopping the tracemalloc library.
        tracemalloc.stop()

        # see how much memory was used during processing.
        # memory used at program start.
        print(f'\nstarting memory used in GBs [current, peak] = {tracemem_start}')
        # memory used by tracemalloc.
        print(f'starting memory used by tracemalloc in GBs = {tracemem_used_start}')
        # memory used during processing.
        print(f'ending memory used in GBs [current, peak] = {tracemem_end}')
        # memory used by tracemalloc.
        print(f'ending memory used by tracemalloc in GBs = {tracemem_used_end}')
    
    return output_data

"""
complete all of the housekeeping procedures before data processing.
    - convert 1-based axes ranges to 0-based.
    - format the variable name and get the variable identifier.
    - convert 1-based timepoint to 0-based.
    - format the output path using pathlib and create the output folder directory.
"""
def housekeeping_procedures(cube_title,
                            output_path,
                            axes_ranges_original, strides, var_original, timepoint_original):
    # validate user-input.
    # -----
    # check that the user-input variable is a valid variable name.
    check_variable(var_original)
    # check that the user-input timepoint is a valid timepoint for the dataset.
    check_timepoint(timepoint_original, cube_title)
    # check that the user-input x-, y-, and z-axis ranges are all specified correctly as [minimum, maximum] integer values.
    check_axes_ranges(axes_ranges_original)
    # check that the user-input strides are all positive integers.
    check_strides(strides)
    
    # pre-processing steps.
    # -----
    # get the number of datapoints (resolution) along each axis of the isotropic cube.
    cube_resolution = get_dataset_resolution(cube_title)
    
    # converts the 1-based axes ranges above to 0-based axes ranges, and truncates the ranges if they are longer than cube_resolution since 
    # the boundaries are periodic. output_data will be filled in with the duplicate data for the truncated data points after processing
    # so that the data files are not read redundantly.
    axes_ranges = convert_to_0_based_ranges(axes_ranges_original, cube_resolution)

    # convert the variable name from var_original into a variable identifier.
    var = get_variable_identifier(var_original)
    
    # converts the 1-based timepoint above to a 0-based timepoint.
    timepoint = convert_to_0_based_value(timepoint_original)
    
    # create the output_path folder if it does not already exist and make sure output_path is formatted properly.
    output_path = pathlib.Path(output_path)
    create_output_folder(output_path)
    
    return (output_path, var, axes_ranges, timepoint, cube_resolution)
