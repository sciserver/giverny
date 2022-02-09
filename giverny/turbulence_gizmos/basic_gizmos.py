import os
import math
import numpy as np
import matplotlib.pyplot as plt

"""
mapping gizmos.
"""
def get_value_names():
    # map of the value names for each variable, e.g. "ux" is the x-component of the velocity. 
    value_name_map = {}
    value_name_map['vel'] = {1:'ux', 2:'uy', 3:'uz'}
    value_name_map['pr'] = {1:'p'}
    
    return value_name_map

def get_num_values_per_datapoint(variable_id):
    # get the number of values per datapoint for the user-specified variable.
    datapoint_values_map = {}
    datapoint_values_map['vel'] = 3
    datapoint_values_map['pr'] = 1
    
    num_values_per_datapoint = datapoint_values_map[variable_id]
    
    return num_values_per_datapoint

def get_variable_identifier(variable_name):
    # convert the variable name to its corresponding identifier, e.g. convert "velocity" to "vel".
    variable_map = {}
    variable_map['velocity'] = 'vel'
    variable_map['pressure'] = 'pr'
    
    variable_name = variable_name.lower()
    
    if variable_name not in variable_map:
        raise Exception(f"'{variable_name}' is not a valid variable: {list(variable_map.keys())}")
    
    variable_id = variable_map[variable_name]
    
    return variable_id

"""
processing gizmos.
"""
def assemble_axis_ranges(x_range, y_range, z_range):
    # assemble all of the axis ranges together into one numpy array.
    axes_ranges = np.array([x_range, y_range, z_range], dtype = np.ndarray)
    
    return axes_ranges

def convert_to_0_based_value(value):
    # convert the 1-based value to a 0-based value.
    updated_value = value - 1
    
    return updated_value

def check_axis_range_num_datapoints(axis_range):
    # number of datapoints in the axis range.
    axis_length = axis_range[1] - axis_range[0] + 1
    
    return axis_length

def update_axis_range(axis_range, axis_length, cube_resolution):
    # convert the 1-based axis range to a 0-based axis range.
    updated_axis_range = list(np.array(axis_range) - 1)
    
    # truncate the axis range if necessary.
    if axis_length > cube_resolution:
        updated_axis_range = [updated_axis_range[0], updated_axis_range[0] + cube_resolution - 1]
    
    return updated_axis_range

def convert_to_0_based_ranges(x_range, y_range, z_range, cube_resolution):
    # calculate the number of datapoints along each axis range.
    x_axis_length = check_axis_range_num_datapoints(x_range)
    y_axis_length = check_axis_range_num_datapoints(y_range)
    z_axis_length = check_axis_range_num_datapoints(z_range)
    
    # convert the 1-based axes ranges to 0-based axes ranges and truncate the axis range if necessary.
    updated_x_range = update_axis_range(x_range, x_axis_length, cube_resolution)
    updated_y_range = update_axis_range(y_range, y_axis_length, cube_resolution)
    updated_z_range = update_axis_range(z_range, z_axis_length, cube_resolution)
    
    return updated_x_range, updated_y_range, updated_z_range

def convert_negative_ranges_to_1_based(axes_ranges):
    # converts the negative axis ranges to 1-based indices such that they can be used for plotting xarray data.
    updated_axes_ranges = np.array([axes_range - axes_range[0] + 1 if axes_range[0] < 0 else axes_range for axes_range in axes_ranges], 
                                   dtype = np.ndarray)
    
    return updated_axes_ranges

"""
output folder gizmos.
"""
def update_folderpath_end_oblique(folderpath):
    # updates folderpath if it does not already end in an oblique ("/") character.
    if folderpath[-1] != '/':
        folderpath += '/'
    
    return folderpath

def create_output_folder(output_path):
    # update output_path with an oblique ("/") character at the end of the folder path if the last character is not already an oblique character.
    output_path = update_folderpath_end_oblique(output_path)
    # create the output folder directory if it does not already exist.
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
"""
user-interfacing gizmos.
"""
def create_contour_plot(value_index_original, variable, cutout_data, plot_ranges, axes_ranges, \
                        output_path, output_filename, \
                        colormap = 'viridis'):
    # constants and dictionaries.
    # -----
    # number of datapoints in each axis range. always 2 datapoints, corresponding to minimum and maximum values.
    axis_range_dimensions = 2
    # turbulence datasets are all in 3 dimensions (x, y, z).
    cube_dimensions = 3
    # variable identifier, e.g. "vel" for "velocity".
    variable_id = get_variable_identifier(variable)
    # names for each value, e.g. value index 0 for velocity data corresponds to the x-component of the velocity ("ux").
    value_name_map = get_value_names()
    
    # exception handling.
    # -----
    # check that the user specified a valid value index.
    if value_index_original not in value_name_map[variable_id]:
        raise Exception(f"{value_index_original} is not a valid value_index: {list(value_name_map[variable_id].keys())}")
        
    # transposed minimum and maximum arrays for both plot_ranges and axes_ranges.
    plot_ranges_min, plot_ranges_max = plot_ranges.T.reshape(axis_range_dimensions, cube_dimensions)
    axes_min, axes_max = axes_ranges.T.reshape(axis_range_dimensions, cube_dimensions)
    
    # raise exception if all of the plot datapoints are not inside the bounds of the user box volume.
    if not(np.all(axes_min <= plot_ranges_min) and np.all(plot_ranges_max <= axes_max)):
        raise Exception(f'the specified plot ranges are not all bounded by the box volume defined by:\n{axes_ranges}')

    # generate the plot.
    # -----
    # convert the 1-based value_index_original to a 0-based index for python.
    value_index = convert_to_0_based_value(value_index_original)
    # update output_path with an oblique ("/") character at the end of the folder path if the last character is not already an oblique character.
    output_path = update_folderpath_end_oblique(output_path)
    
    # name of the value that is being plotted.
    value_name = value_name_map[variable_id][value_index_original]
    
    # convert any negative indices to 1-based indices.
    corrected_plot_ranges = convert_negative_ranges_to_1_based(plot_ranges)
    
    # specify the subset (or full) axes ranges to use for plotting. cutout_data is of the format [z-range, y-range, x-range, output value index]. the 
    # ranges are converted to 0-based indices.
    plot_data = cutout_data[corrected_plot_ranges[2][0] - 1:corrected_plot_ranges[2][1], \
                            corrected_plot_ranges[1][0] - 1:corrected_plot_ranges[1][1], \
                            corrected_plot_ranges[0][0] - 1:corrected_plot_ranges[0][1], \
                            value_index]
    
    # create the figure.
    fig = plt.figure(figsize = (11, 8.5), dpi = 300)
    ax = fig.add_subplot(111)
    cf = plot_data.plot(ax = ax, cmap = colormap, center = False)

    # plot labels.
    title_str = f'plane {variable} ({value_name}) contour plot ({ax.get_title()})'
    plt.title(title_str, fontsize = 16, weight = 'bold')
    plt.xlabel(ax.get_xlabel(), fontsize = 14, weight = 'bold')
    plt.ylabel(ax.get_ylabel(), fontsize = 14, weight = 'bold')
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xticks([math.ceil(xlims[0]), math.floor(xlims[1])])
    plt.yticks([math.ceil(ylims[0]), math.floor(ylims[1])])

    # save the figure.
    #plt.show()
    plt.savefig(output_path + output_filename)
    plt.close()

def retrieve_data_for_point(x, y, z, output_data, axes_ranges):
    # constants.
    # -----
    # number of datapoints in each axis range. always 2 datapoints, corresponding to minimum and maximum values.
    axis_range_dimensions = 2
    # turbulence datasets are all in 3 dimensions (x, y, z).
    cube_dimensions = 3
    
    # -----
    # minimum and maximum endpoints along each axis for the points the user requested.
    endpoints = np.array([[np.min(x), np.max(x)], \
                          [np.min(y), np.max(y)], \
                          [np.min(z), np.max(z)]], dtype = np.ndarray)

    # transposed minimum and maximum arrays for both endpoints and axes_ranges.
    endpoints_min, endpoints_max = endpoints.T.reshape(axis_range_dimensions, cube_dimensions)
    axes_min, axes_max = axes_ranges.T.reshape(axis_range_dimensions, cube_dimensions)

    # raise exception if all of the user requested datapoints are not inside the bounds of the user box volume.
    if not(np.all(axes_min <= endpoints_min) and np.all(endpoints_max <= axes_max)):
        raise Exception(f'the specified point(s) are not all bounded by the box volume defined by:\n{axes_ranges}')

    # value(s) corresponding to the specified (x, y, z) datapoint(s).
    output_val = output_data.sel(x = x, y = y, z = z)

    return output_val