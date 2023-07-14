import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

"""
user-input checking gizmos.
"""
def check_dataset_title(dataset_title):
    # check that dataset_title is a valid dataset title.
    valid_dataset_titles = ['isotropic4096', 'isotropic8192']
    
    if dataset_title not in valid_dataset_titles:
        raise Exception(f"'{dataset_title}' (case-sensitive) is not a valid dataset title: {valid_dataset_titles}")
        
    return

def check_variable(variable):
    # check that variable is a valid variable name.
    valid_variables = ['pressure', 'velocity']
    
    if variable not in valid_variables:
        raise Exception(f"'{variable}' (case-sensitive) is not a valid variable: {valid_variables}")
        
    return

def check_timepoint(timepoint, dataset_title):
    # check that timepoint is a valid timepoint for the dataset.
    valid_timepoints = {'isotropic4096':range(1, 1 + 1), 'isotropic8192':range(1, 6 + 1)}
    
    if timepoint not in valid_timepoints[dataset_title]:
        raise Exception(f'{timepoint} is not a valid timepoint: must be in the inclusive range of ' +
                        f'[{valid_timepoints[dataset_title][0]}, {valid_timepoints[dataset_title][-1]}]')
        
    return

def check_spatial_interpolation(sint):
    valid_sints = ['none', 'lag4', 'lag6', 'lag8', 'm1q4', 'm2q8']
    
    if sint not in valid_sints:
        raise Exception(f"'{sint}' (case-sensitive) is not a valid sint: {valid_sints}")
        
    return

def check_gradient_interpolation(sint):
    valid_sints = ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm1q4', 'm2q8']
    
    if sint not in valid_sints:
        raise Exception(f"'{sint}' (case-sensitive) is not a valid sint: {valid_sints}")
        
    return

def check_hessian_interpolation(sint):
    valid_sints = ['fd4noint', 'fd6noint', 'fd8noint', 'm2q8']
    
    if sint not in valid_sints:
        raise Exception(f"'{sint}' (case-sensitive) is not a valid sint: {valid_sints}")
        
    return

def check_laplacian_interpolation(sint):
    valid_sints = ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4']
    
    if sint not in valid_sints:
        raise Exception(f"'{sint}' (case-sensitive) is not a valid sint: {valid_sints}")
        
    return

def check_axes_ranges(axes_ranges):
    # check that the axis ranges are all specified as minimum and maximum integer values.
    for axis_range in axes_ranges:
        if len(axis_range) != 2:
            raise Exception(f'axis range, {axis_range}, is not correctly specified as [minimum, maximum]')
            
        for val in axis_range:
            if type(val) != int:
                raise Exception(f'{val} in axis range, {list(axis_range)}, is not an integer')
                
def check_strides(strides):
    # check that the strides are all positive integer values.
    for stride in strides:
        if type(stride) != int or stride < 1:
            raise Exception(f'stride, {stride}, is not an integer value >= 1')

"""
mapping gizmos.
"""
def get_dataset_resolution(dataset_title):
    # get the number of datapoints (resolution) along each axis of the isotropic dataset.
    return {
        'isotropic4096':4096,
        'isotropic8192':8192
    }[dataset_title]

def get_filename_prefix(dataset_title):
    # get the common filename prefix for each database file in the dataset.
    return {
        'isotropic4096':'iso4096',
        'isotropic8192':'iso8192'
    }[dataset_title]

def get_value_names():
    # map of the value names for each variable, e.g. "ux" is the x-component of the velocity.
    return {
        'vel':{1:'ux', 2:'uy', 3:'uz'},
        'pr':{1:'p'}
    }

def get_num_values_per_datapoint(variable_id):
    # get the number of values per datapoint for the user-specified variable.
    return {
        'vel':3,
        'pr':1
    }[variable_id]

def get_variable_identifier(variable_name):
    # convert the variable name to its corresponding identifier, e.g. convert "velocity" to "vel".
    return {
        'velocity':'vel',
        'pressure':'pr'
    }[variable_name]

def get_output_variable_name(variable_name):
    # format the variable name string for the HDF5 output file dataset name.
    return {
        'velocity':'Velocity',
        'pressure':'Pressure'
    }[variable_name]

def get_attribute_type(variable_name):
    # get the attribute type of the output data.
    return {
        'velocity':'Vector',
        'pressure':'Scalar'
    }[variable_name]

"""
processing gizmos.
"""
def assemble_axis_data(axes_data):
    # assemble all of the axis data together into one numpy array.
    return np.array(axes_data, dtype = np.ndarray)

def convert_to_0_based_value(value):
    # convert the 1-based value to a 0-based value.
    return value - 1

def get_axes_ranges_num_datapoints(axes_ranges):
    # number of datapoints along each axis.
    return axes_ranges[:, 1] - axes_ranges[:, 0] + 1

def convert_to_0_based_ranges(axes_ranges, cube_resolution):
    # calculate the number of datapoints along each axis range.
    axes_lengths = get_axes_ranges_num_datapoints(axes_ranges)
    
    # convert the 1-based axes ranges to 0-based axes ranges.
    axes_ranges = axes_ranges - 1
    
    # truncate the axes range if necessary.
    for axis_index, axis_range in enumerate(axes_ranges):
        if axes_lengths[axis_index] > cube_resolution:
            axes_ranges[axis_index, 1] = axes_ranges[axis_index, 0] + cube_resolution - 1
    
    return axes_ranges
    
"""
output folder gizmos.
"""
def create_output_folder(output_path):
    # create the output folder directory.
    os.makedirs(output_path, exist_ok = True)
        
"""
user-interfacing gizmos.
"""
def contourPlot(cube, value_index_original, cutout_data, plot_ranges, axes_ranges, strides, output_filename,
                colormap = 'viridis'):
    # dictionaries.
    # -----
    variable = cube.var_name
    
    # variable identifier, e.g. "vel" for "velocity".
    variable_id = get_variable_identifier(variable)
    # names for each value, e.g. value index 0 for velocity data corresponds to the x-component of the velocity ("ux").
    value_name_map = get_value_names()
    
    # create the output folder if it does not already exist..
    create_output_folder(cube.output_path)
    
    # exception handling.
    # -----
    # check that the user-input x-, y-, and z-axis plot ranges are all specified correctly as [minimum, maximum] integer values.
    check_axes_ranges(plot_ranges)
    
    # check that the user specified a valid value index.
    if value_index_original not in value_name_map[variable_id]:
        raise Exception(f"{value_index_original} is not a valid value_index: {list(value_name_map[variable_id].keys())}")
        
    # transposed minimum and maximum arrays for both plot_ranges and axes_ranges.
    plot_ranges_min = plot_ranges[:, 0]
    plot_ranges_max = plot_ranges[:, 1]
    
    axes_min = axes_ranges[:, 0]
    axes_max = axes_ranges[:, 1]
    
    # raise exception if all of the plot datapoints are not inside the bounds of the user box volume.
    if not(np.all(axes_min <= plot_ranges_min) and np.all(plot_ranges_max <= axes_max)):
        raise Exception(f'the specified plot ranges are not all bounded by the box volume defined by:\n{axes_ranges}')
        
    # determine how many of the axis minimum values are equal to their corresponding axis maximum value.
    num_axes_equal_min_max = plot_ranges_min == plot_ranges_max
    # raise exception if the data being plotted is not 2-dimensional.
    if np.count_nonzero(num_axes_equal_min_max == True) != 1:
        raise Exception(f'only one axis (x, y, or z) should be specified as a single point, e.g. z_plot_range = [3, 3], to create a contour plot')
        
    # convert the requested plot ranges to the domain [0, 2 * pi].
    xcoor_values = np.around(np.arange(plot_ranges_min[0] - 1, plot_ranges_max[0], strides[0], dtype = np.float32) * cube.dx, cube.decimals)
    ycoor_values = np.around(np.arange(plot_ranges_min[1] - 1, plot_ranges_max[1], strides[1], dtype = np.float32) * cube.dx, cube.decimals)
    zcoor_values = np.around(np.arange(plot_ranges_min[2] - 1, plot_ranges_max[2], strides[2], dtype = np.float32) * cube.dx, cube.decimals)

    # generate the plot.
    print('Generating contour plot...')
    print('-----')
    sys.stdout.flush()
    
    # -----
    # convert the 1-based value_index_original to a 0-based index for python.
    value_index = convert_to_0_based_value(value_index_original)
    
    # name of the value that is being plotted.
    value_name = value_name_map[variable_id][value_index_original]
    
    # specify the subset (or full) axes ranges to use for plotting. cutout_data is of the format [z-range, y-range, x-range, output value index].
    plot_data = cutout_data[cube.dataset_name].sel(xcoor = xcoor_values, 
                                                   ycoor = ycoor_values, 
                                                   zcoor = zcoor_values,
                                                   values = value_index)
    
    # raise exception if only one point is going to be plotted along more than 1 axis. a contour plot requires more 
    # than 1 point along 2 axes. this check is required in case the user specifies a stride along an axis that 
    # is >= number of datapoints along that axis.
    if plot_data.shape.count(1) > 1:
        raise Exception('the contour plot could not be created because more than 1 axis only had 1 datapoint')
    
    # create the figure.
    fig = plt.figure(figsize = (11, 8.5), dpi = 300)
    fig.set_facecolor('white')
    ax = fig.add_subplot(111)
    cf = plot_data.plot(ax = ax, cmap = colormap, center = False,
                        cbar_kwargs = {'label': None})

    # plot labels.
    # convert the [0, 2 * pi] domain plot label to the integer index in the [1, 8192] domain.
    original_axis_title = ax.get_title().replace('coor', '').split('=')
    plane_axis = original_axis_title[0].strip()
    plane_point = int(float(original_axis_title[-1].strip()) / cube.dx) + 1
    axis_title = plane_axis + ' = ' + str(plane_point) + ', t = ' + str(cutout_data.attrs['t_start'])
    title_str = f'plane {variable} ({value_name}) contour plot ({axis_title})'
    # get the x-axis and y-axis variable names (e.g. 'x' and 'y') before the axis labels are appended to.
    x_axis_variable = ax.get_xlabel()
    y_axis_variable = ax.get_ylabel()
    x_label = str(x_axis_variable).split('coor')[0].strip()
    y_label = str(y_axis_variable).split('coor')[0].strip()
    x_axis_stride = cutout_data.attrs[f'{x_label}_step']
    y_axis_stride = cutout_data.attrs[f'{y_label}_step']
    plt.title(title_str, fontsize = 16, weight = 'bold')
    plt.xlabel(f'{x_label} (stride = {x_axis_stride})', fontsize = 14, weight = 'bold')
    plt.ylabel(f'{y_label} (stride = {y_axis_stride})', fontsize = 14, weight = 'bold')
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    # adjust the axis ticks to the center of each datapoint.
    x_ticks = [xlims[0] + ((x_axis_stride * cube.dx) / 2), xlims[1] - ((x_axis_stride * cube.dx) / 2)]
    y_ticks = [ylims[0] + ((y_axis_stride * cube.dx) / 2), ylims[1] - ((y_axis_stride * cube.dx) / 2)]
    # axis datapoints.
    x_axis_points = np.rint(plot_data.coords[x_axis_variable].values / cube.dx) + 1
    y_axis_points = np.rint(plot_data.coords[y_axis_variable].values / cube.dx) + 1
    plt.xticks(x_ticks, [int(x_axis_points[0]), int(x_axis_points[-1])])
    plt.yticks(y_ticks, [int(y_axis_points[0]), int(y_axis_points[-1])])
    
    # save the figure.
    plt.tight_layout()
    plt.savefig(cube.output_path.joinpath(output_filename))
    
    # show the figure in the notebook, and shrinks the dpi to make it easily visible.
    fig.set_dpi(67)
    plt.tight_layout()
    plt.show()
    
    # close the figure.
    plt.close()
    
    print('-----')
    print('Contour plot created successfully.')

def dataValues(cube, x, y, z, output_data, axes_ranges, strides):
    # retrieve data values for all of the specified points.
    
    # -----
    # minimum and maximum endpoints along each axis for the points the user requested.
    endpoints_min = np.array([np.min(x), np.min(y), np.min(z)], dtype = np.int32)
    endpoints_max = np.array([np.max(x), np.max(y), np.max(z)], dtype = np.int32)
    
    # exception_handling.
    # -----
    # raise exception if all of the user requested datapoints are not inside the bounds of the user box volume.
    if not(np.all(axes_ranges[:, 0] <= endpoints_min) and np.all(endpoints_max <= axes_ranges[:, 1])):
        raise Exception(f'the specified point(s) are not all bounded by the box volume defined by:\n{axes_ranges}')
    
    # convert the requested plot ranges to 0-based indices and then to their corresponding values in the domain [0, 2 * pi].
    xcoor_values = np.around(np.arange(endpoints_min[0] - 1, endpoints_max[0], strides[0], dtype = np.float32) * cube.dx, cube.decimals)
    ycoor_values = np.around(np.arange(endpoints_min[1] - 1, endpoints_max[1], strides[1], dtype = np.float32) * cube.dx, cube.decimals)
    zcoor_values = np.around(np.arange(endpoints_min[2] - 1, endpoints_max[2], strides[2], dtype = np.float32) * cube.dx, cube.decimals)

    # value(s) corresponding to the specified (x, y, z) datapoint(s).
    return output_data[cube.dataset_name].sel(xcoor = xcoor_values,
                                              ycoor = ycoor_values,
                                              zcoor = zcoor_values)
