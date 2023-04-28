import pathlib
import tracemalloc
from giverny.turbulence_gizmos.basic_gizmos import *

"""
retrieve a cutout of the isotropic cube.
"""
def getCutout(cube, var_original, timepoint_original, axes_ranges_original, strides,
              rewrite_metadata = False, trace_memory = False):
    from giverny.turbulence_gizmos.getCutout import getCutout_process_data
    
    # housekeeping procedures.
    # -----
    var, axes_ranges, timepoint = \
        getCutout_housekeeping_procedures(cube, axes_ranges_original, strides, var_original, timepoint_original)
    
    # process the data.
    # -----
    # starting the tracemalloc library.
    if trace_memory:
        tracemalloc.start()
        # checking the memory usage of the program.
        tracemem_start = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_start = tracemalloc.get_tracemalloc_memory() / (1024**3)

    # parse the database files, generate the output_data matrix, and write the matrix to an hdf5 file.
    output_data = getCutout_process_data(cube, axes_ranges, var, timepoint,
                                         axes_ranges_original, strides, var_original, timepoint_original,
                                         rewrite_metadata = rewrite_metadata)
    
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
complete all of the getCutout housekeeping procedures before data processing.
    - convert 1-based axes ranges to 0-based.
    - format the variable name and get the variable identifier.
    - convert 1-based timepoint to 0-based.
"""
def getCutout_housekeeping_procedures(cube, axes_ranges_original, strides, var_original, timepoint_original):
    # validate user-input.
    # -----
    # check that the user-input variable is a valid variable name.
    check_variable(var_original)
    # check that the user-input timepoint is a valid timepoint for the dataset.
    check_timepoint(timepoint_original, cube.dataset_title)
    # check that the user-input x-, y-, and z-axis ranges are all specified correctly as [minimum, maximum] integer values.
    check_axes_ranges(axes_ranges_original)
    # check that the user-input strides are all positive integers.
    check_strides(strides)
    
    # pre-processing steps.
    # -----
    # converts the 1-based axes ranges above to 0-based axes ranges, and truncates the ranges if they are longer than 
    # the cube resolution (cube.N) since the boundaries are periodic. output_data will be filled in with the duplicate data 
    # for the truncated data points after processing so that the data files are not read redundantly.
    axes_ranges = convert_to_0_based_ranges(axes_ranges_original, cube.N)

    # convert the variable name from var_original into a variable identifier.
    var = get_variable_identifier(var_original)
    
    # converts the 1-based timepoint above to a 0-based timepoint.
    timepoint = convert_to_0_based_value(timepoint_original)
    
    return (var, axes_ranges, timepoint)

"""
interpolate pressures for points of the isotropic cube.
"""
def getPressure(cube, timepoint_original, sint, points,
                rewrite_metadata = False, trace_memory = False):
    
    print('\n' + '-' * 25 + '\ngetPressure is processing...')
    sys.stdout.flush()
    
    # call the driver function for interpolating the data for different variables.
    output_data = getVariable(cube, points, sint, timepoint_original, rewrite_metadata, trace_memory, var_original = 'pressure')
    
    return output_data

"""
interpolate velocities for points of the isotropic cube.
"""
def getVelocity(cube, timepoint_original, sint, points,
                rewrite_metadata = False, trace_memory = False):
    
    print('\n' + '-' * 25 + '\ngetVelocity is processing...')
    sys.stdout.flush()
    
    # call the driver function for interpolating the data for different variables.
    output_data = getVariable(cube, points, sint, timepoint_original, rewrite_metadata, trace_memory, var_original = 'velocity')
    
    return output_data

"""
interpolate the variable for points of the isotropic cube.
"""
def getVariable(cube, points, sint, timepoint_original,
                rewrite_metadata = False, trace_memory = False, var_original = 'pressure'):
    from giverny.turbulence_gizmos.getVariable import getVariable_process_data
    
    # housekeeping procedures. will handle multiple variables, e.g. 'pressure' and 'velocity'.
    # -----
    sint, var, timepoint = getVariable_housekeeping_procedures(cube, sint, var_original, timepoint_original)
    
    # process the data.
    # -----
    # starting the tracemalloc library.
    if trace_memory:
        tracemalloc.start()
        # checking the memory usage of the program.
        tracemem_start = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_start = tracemalloc.get_tracemalloc_memory() / (1024**3)

    # parse the database files, generate the output_data array, and write the array to an hdf5 file.
    output_data = getVariable_process_data(cube, points, sint, var, timepoint,
                                           var_original, timepoint_original,
                                           rewrite_metadata = rewrite_metadata)
    
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
    - format the variable name and get the variable identifier.
    - convert 1-based timepoint to 0-based.
    - format the output path using pathlib and create the output folder directory.
"""
def getVariable_housekeeping_procedures(cube, sint, var_original, timepoint_original):
    # validate user-input.
    # -----
    # check that the user-input variable is a valid variable name.
    check_variable(var_original)
    # check that the user-input timepoint is a valid timepoint for the dataset.
    check_timepoint(timepoint_original, cube.dataset_title)
    # check that the user-input spatial interpolation (sint) is a valid spatial interpolation method.
    sint = sint.strip().lower()
    check_spatial_interpolation(sint)
    
    # pre-processing steps.
    # -----
    # convert the variable name from var_original into a variable identifier.
    var = get_variable_identifier(var_original)
    
    # converts the 1-based timepoint above to a 0-based timepoint.
    timepoint = convert_to_0_based_value(timepoint_original)
    
    return (sint, var, timepoint)
